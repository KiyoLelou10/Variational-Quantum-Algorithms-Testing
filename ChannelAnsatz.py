# -*- coding: utf-8 -*-
"""
8-class Hadamard-sim, 16 channels -> 4-qubit stage-2, overlap loss (swap-test equivalent).
Runs MNIST, KMNIST, FashionMNIST sequentially with identical hyperparams.

Fixes:
- NaN-safe normalizations (safe_normalize)
- P cleaned & renormalized
- pin_memory only when CUDA
- (optional) parameter clamp after each step to prevent blow-ups

Matches best trial params:
M=8, activation='silu', Uw_pattern='cw_ccw_alternating',
optimizer=RMSprop, lr=9.751630997371441e-04,
weight_decay=4.074541806255375e-04, batch_size=512,
scheduler=OneCycleLR with pct_start=0.23715573331861967,
div_factor=10.805871407081492, final_div_factor=859.687134651531.
"""

import time, math, gc, numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST, KMNIST, FashionMNIST

# ---------------- Device & seeds ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
torch.manual_seed(42); np.random.seed(42)

# ---------------- Helpers ----------------
def safe_normalize(x, dim=1, eps=1e-6):
    # Clean infinities/NaNs first, then divide by a clamped norm.
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    n = torch.norm(x, dim=dim, keepdim=True).clamp_min(eps)
    return x / n

def ry_matrix(a):
    c = torch.cos(a/2); s = torch.sin(a/2)
    return torch.stack([torch.stack([c, -s], -1),
                        torch.stack([s,  c], -1)], -2)

CNOT_2 = torch.tensor([[1,0,0,0],
                       [0,1,0,0],
                       [0,0,0,1],
                       [0,0,1,0]], dtype=torch.float32, device=device)

def apply_single(S, n, q, G2):
    b = S.shape[0]; T = S.view(b, *([2]*n)); ax = 1+q
    axes = [0] + [a for a in range(1,n+1) if a!=ax] + [ax]
    TP = T.permute(*axes).contiguous()
    rest = int(np.prod(TP.shape[1:-1])) if TP.dim()>2 else 1
    out = torch.einsum('bvi,ij->bvj', TP.view(b,rest,2), G2.to(TP.device))
    inv = [0]*(n+1)
    for i,a in enumerate(axes): inv[a]=i
    return out.view(*TP.shape).permute(*inv).contiguous().view(b,-1)

def apply_two(S, n, q1, q2, G4):
    b=S.shape[0]; T=S.view(b,*([2]*n)); ax1=1+q1; ax2=1+q2
    axes=[0]+[a for a in range(1,n+1) if a not in (ax1,ax2)]+[ax1,ax2]
    TP=T.permute(*axes).contiguous()
    lead=TP.shape[1:-2]; V=1
    for d0 in lead: V*=d0
    out=torch.einsum('bvf,fg->bvg', TP.view(b,V,4), G4.to(TP.device))
    inv=[0]*(n+1)
    for i,a in enumerate(axes): inv[a]=i
    return out.view(b,*lead,2,2).permute(*inv).contiguous().view(b,-1)

def ry_layer(S, n, params):
    for q in range(params.shape[0]):
        S = apply_single(S, n, q, ry_matrix(params[q]))
    return S

def circular_cnot_dir(S, n, nv, forward=True):
    for q in range(nv):
        tgt = (q+1) % nv if forward else (q-1) % nv
        S = apply_two(S, n, q, tgt, CNOT_2)
    return S

def ansatz_block(S, n, nv, theta, M, ent_pattern="cw_ccw_alternating"):
    p=0
    for m in range(M):
        S = ry_layer(S, n, theta[p:p+nv]); p+=nv
        if ent_pattern == "cw_ccw_alternating":
            S = circular_cnot_dir(S, n, nv, forward=(m % 2 == 0))
        elif ent_pattern == "cw_fixed":
            S = circular_cnot_dir(S, n, nv, forward=True)
        else:
            raise ValueError(f"Unknown entanglement pattern: {ent_pattern}")
    return ry_layer(S, n, theta[p:p+nv])

# ---------------- Model ----------------
class HadamardSimMultiChannel8_16ch4q(nn.Module):
    """
    Stage-1: real amplitude encoding on SAME 8-qubit register.
      For each channel i: |psi_wi> = Uw_i|0^8|, z_i = <psi_wi | psi_x>.
    Stage-2: pack 16 features -> 4-qubit state, apply Uw(4q),
      marginalize first qubit, last 3 are the 8-class distribution P.
    """
    def __init__(self, n_AB=8, num_channels=16, n_stage2=4, l=3, M=8,
                 activation="silu", beta0=0.47548822599780394,
                 ent_pattern_stage1="cw_ccw_alternating",
                 ent_pattern_stage2="cw_ccw_alternating"):
        super().__init__()
        self.n_AB=n_AB; self.num_channels=num_channels
        self.n_stage2=n_stage2; self.l=l; self.M=M
        self.activation_name=activation
        self.ent_pattern_stage1=ent_pattern_stage1
        self.ent_pattern_stage2=ent_pattern_stage2

        # Channel params (8 qubits)
        self.theta_channels = nn.Parameter(
            torch.randn(num_channels, (M+1)*n_AB, device=device)*0.08
        )
        # Stage-2 params (4 qubits)
        self.theta_stage2 = nn.Parameter(
            torch.randn((M+1)*n_stage2, device=device)*0.08
        )
        # kept for completeness (unused by overlap loss)
        self.beta = nn.Parameter(torch.tensor(beta0, dtype=torch.float32, device=device))
        self.logit_bias = nn.Parameter(torch.zeros(2**l, dtype=torch.float32, device=device))

    def zero_state(self, b, n):
        S=torch.zeros(b, 1<<n, dtype=torch.float32, device=device); S[:,0]=1.0; return S

    def build_w_state(self, theta_vec):
        S0=self.zero_state(1, self.n_AB)
        return ansatz_block(S0, n=self.n_AB, nv=self.n_AB,
                            theta=theta_vec, M=self.M,
                            ent_pattern=self.ent_pattern_stage1)  # [1,256]

    @staticmethod
    def hadamard_overlap_sim(Sw, Sx):
        return torch.einsum('bd,ad->b', Sx, Sw.expand(Sx.shape[0], -1))  # [B]

    def f_of_z(self, z):
        name=self.activation_name
        if name=="silu": return F.silu(z)
        if name=="gelu": return F.gelu(z)
        if name=="elu":  return F.elu(z, alpha=1.0)
        if name=="leaky":return F.leaky_relu(z, negative_slope=0.02)
        if name=="tanh": return torch.tanh(z)
        if name=="relu": return z.clamp_min(0.0)
        if name=="tanh_deg5": return z - (z**3)/3.0 + (2.0/15.0)*(z**5)
        if name=="relu_deg5": return (0.0585966775 + 0.5*z + 0.820271503*(z**2) - 0.410094752*(z**4))
        if name=="softsign": return z/(1.0+torch.abs(z))
        if name=="softplus": return F.softplus(z)
        return z

    def forward(self, x):
        b=x.shape[0]
        Sx = x.to(device)  # L2-normalized (16x16 -> 256)

        # 16-channel overlaps
        z_list=[]
        for i in range(self.num_channels):
            Sw=self.build_w_state(self.theta_channels[i])   # [1,256]
            zi=self.hadamard_overlap_sim(Sw, Sx)            # [B]
            z_list.append(zi)
        Z=torch.stack(z_list, dim=1)                        # [B,16]

        # activation per channel
        Fch=self.f_of_z(Z)                                  # [B,16]

        # normalize to valid 4-qubit statevector (len 16)
        Fch = safe_normalize(Fch, dim=1, eps=1e-6)

        # Stage-2 HEA on 4 qubits
        S4=ansatz_block(Fch, n=self.n_stage2, nv=self.n_stage2,
                        theta=self.theta_stage2, M=self.M,
                        ent_pattern=self.ent_pattern_stage2)  # [B,16]

        # Normalize again for safety
        S4 = safe_normalize(S4, dim=1, eps=1e-6)

        # Marginalize first qubit -> last 3 qubits are labels (8 classes)
        T=S4.view(b,2,2,2,2)
        P=(T**2).sum(dim=1).contiguous().view(b, 2**self.l)  # [B,8], sums to 1 ideally

        # Clean + renormalize to exactly sum to 1
        P = torch.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        P = P / P.sum(dim=1, keepdim=True).clamp_min(1e-6)

        logp=torch.log(P.clamp_min(1e-12))  # handy for monitoring if needed
        return logp, P

# ---------------- Data utils ----------------
def preprocess_16x16_L2(imgs_np):
    N=imgs_np.shape[0]
    im=imgs_np.reshape(-1,28,28)[:,6:22,6:22]   # center crop 16x16
    flat=im.reshape(N,-1).astype(np.float32)
    n=np.linalg.norm(flat, axis=1, keepdims=True)
    n[n==0]=1.0
    return flat/n

def make_dataloaders(ds_ctor, batch_size=512, max_per_class=2000, classes=list(range(8))):
    train_raw=ds_ctor(root='./data', train=True, download=True)
    test_raw =ds_ctor(root='./data', train=False, download=True)

    Xtr_raw=train_raw.data.numpy().reshape(-1,28*28); ytr=train_raw.targets.numpy()
    Xte_raw=test_raw.data.numpy().reshape(-1,28*28); yte=test_raw.targets.numpy()

    mask_tr=np.isin(ytr, classes); mask_te=np.isin(yte, classes)
    Xtr_raw,ytr=Xtr_raw[mask_tr], ytr[mask_tr]
    Xte_raw,yte=Xte_raw[mask_te], yte[mask_te]

    rng=np.random.default_rng(42)
    idx=[]
    for c in classes:
        where=np.where(ytr==c)[0]
        k=min(max_per_class, where.size)
        idx.append(rng.choice(where, k, replace=False))
    idx=np.concatenate(idx)
    Xtr_raw,ytr=Xtr_raw[idx], ytr[idx]

    Xtr=preprocess_16x16_L2(Xtr_raw)
    Xte=preprocess_16x16_L2(Xte_raw)

    train_ds=TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                           torch.tensor(ytr, dtype=torch.long))
    test_ds =TensorDataset(torch.tensor(Xte, dtype=torch.float32),
                           torch.tensor(yte, dtype=torch.long))

    pin = (device.type == "cuda")
    train_loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True,  pin_memory=pin)
    test_loader =DataLoader(test_ds,  batch_size=batch_size, shuffle=False, pin_memory=pin)
    return train_loader, test_loader

# ---------------- Overlap loss (swap-test equivalent) ----------------
def overlap_loss(P, y):
    # maximize P[true]; equivalently minimize negative mean overlap
    return -(P[torch.arange(y.size(0), device=P.device), y].clamp_min(1e-12)).mean()

# ---------------- Train/eval one dataset ----------------
def run_one_dataset(ds_name, ds_ctor,
                    M=8,
                    activation="silu",
                    ent_pattern="cw_ccw_alternating",
                    batch_size=512,
                    lr=9.751630997371441e-04,
                    weight_decay=4.074541806255375e-04,
                    pct_start=0.23715573331861967,
                    div_factor=10.805871407081492,
                    final_div_factor=859.687134651531,
                    epochs=100,
                    max_per_class=2000,
                    clamp_params=True):
    print(f"\n=== {ds_name} (classes 0..7), M={M}, act={activation}, ent={ent_pattern} ===")
    train_loader, test_loader = make_dataloaders(ds_ctor, batch_size=batch_size, max_per_class=max_per_class)

    model = HadamardSimMultiChannel8_16ch4q(
        n_AB=8, num_channels=16, n_stage2=4, l=3, M=M,
        activation=activation,
        beta0=0.47548822599780394,
        ent_pattern_stage1=ent_pattern,
        ent_pattern_stage2=ent_pattern
    ).to(device)

    optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,  # use provided lr as max_lr (as in your BO)
        total_steps=total_steps,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor
    )

    best_acc = 0.0
    for epoch in range(epochs):
        model.train(); t0=time.time(); losses=[]
        for xb, yb in train_loader:
            xb=xb.to(device); yb=yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, P = model(xb)
            loss = overlap_loss(P, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if clamp_params:
                with torch.no_grad():
                    for p in model.parameters():
                        p.clamp_(-100.0, 100.0)
            scheduler.step()
            losses.append(loss.item())

        # Eval
        model.eval(); corr=0; tot=0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb=xb.to(device); yb=yb.to(device)
                _, P = model(xb)
                pred = P.argmax(1)
                corr += (pred==yb).sum().item()
                tot  += yb.size(0)
        acc = 100.0 * corr / max(1, tot)
        best_acc = max(best_acc, acc)

        print(f"Epoch {epoch+1:3d}/{epochs} | loss={np.mean(losses):.4f} | acc={acc:.2f}% | best={best_acc:.2f}% | time={time.time()-t0:.1f}s")

        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return best_acc

if __name__ == "__main__":
    params = dict(
        M=8,
        activation="silu",
        ent_pattern="cw_ccw_alternating",
        batch_size=512,
        lr=9.751630997371441e-04,
        weight_decay=4.074541806255375e-04,
        pct_start=0.23715573331861967,
        div_factor=10.805871407081492,
        final_div_factor=859.687134651531,
        epochs=100,
        max_per_class=2000,
        clamp_params=True
    )

    results = {}
    results["MNIST"]        = run_one_dataset("MNIST",        MNIST,        **params)
    results["KMNIST"]       = run_one_dataset("KMNIST",       KMNIST,       **params)
    results["FashionMNIST"] = run_one_dataset("FashionMNIST", FashionMNIST, **params)

    print("\n=== Summary (best accuracies) ===")
    for k,v in results.items():
        print(f"{k}: {v:.2f}%")
