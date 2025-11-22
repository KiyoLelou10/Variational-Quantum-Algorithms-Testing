# qae_mnist_best.py
import time, math, gc, os, numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST, KMNIST

# ---------------- Device & seeds ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
torch.manual_seed(42)
np.random.seed(42)

# ---------------- Fixed configs ----------------
n_AB = 8           # total A+B qubits (2**8 = 256 amplitudes)
l = 3              # B size (8 classes)
n_ref = l          # kept for layout (not used)
use_anc = False
anc = 1 if use_anc else 0
k = n_AB - l       # A size (=5)
assert k >= 0
beta = 1.5         # unused (kept for compatibility)

# numerical guards
eps = 1e-8
grad_clip = 5.0

# total qubits: AB + ref + anc
n_qubits = n_AB + n_ref + anc
print(f"n_AB={n_AB} (A={k}, B={l}), n_ref={n_ref}, anc={anc}, total_qubits={n_qubits}")

# ---------------- Gates ----------------
def ry_matrix(angle):
    c = torch.cos(angle / 2.0)
    s = torch.sin(angle / 2.0)
    return torch.stack([torch.stack([c, -s], dim=-1),
                        torch.stack([s,  c], dim=-1)], dim=-2)

CNOT_2 = torch.tensor([[1,0,0,0],
                       [0,1,0,0],
                       [0,0,0,1],
                       [0,0,1,0]], dtype=torch.float32, device=device)

def apply_single_qubit_gate(state, n, qubit_idx, gate_2x2):
    batch = state.shape[0]
    t = state.view(batch, *([2]*n))
    axis = 1 + qubit_idx
    axes = [0] + [a for a in range(1, n+1) if a != axis] + [axis]
    tperm = t.permute(*axes).contiguous()
    rest = int(np.prod(tperm.shape[1:-1])) if tperm.dim()>2 else 1
    tflat = tperm.view(batch, rest, 2)
    out = torch.einsum('bvi,ij->bvj', tflat, gate_2x2.to(tflat.device))
    out = out.view(*tperm.shape)
    inv_perm = [0] * (n+1)
    for i, a in enumerate(axes):
        inv_perm[a] = i
    out = out.permute(*inv_perm).contiguous()
    return out.view(batch, -1)

def apply_two_qubit_gate(state, n, q1, q2, gate4):
    batch = state.shape[0]
    t = state.view(batch, *([2]*n))
    ax1 = 1 + q1; ax2 = 1 + q2
    axes = [0] + [a for a in range(1, n+1) if a not in (ax1, ax2)] + [ax1, ax2]
    tperm = t.permute(*axes).contiguous()
    leading = tperm.shape[1:-2]
    V = 1
    for d in leading: V *= d
    tflat = tperm.view(batch, V, 4)
    out = torch.einsum('bvf,fg->bvg', tflat, gate4.to(tflat.device))
    out = out.view(batch, *leading, 2, 2)
    inv_perm = [0] * (n+1)
    for i, a in enumerate(axes):
        inv_perm[a] = i
    out = out.permute(*inv_perm).contiguous()
    return out.view(batch, -1)

def circular_cnot_dir(state, n, nv, forward=True):
    """Ring of CNOTs over qubits [0..nv-1]; direction can be cw (forward=True) or ccw (False)."""
    s = state
    for q in range(nv):
        control = q
        target = (q + 1) % nv if forward else (q - 1) % nv
        s = apply_two_qubit_gate(s, n, control, target, CNOT_2)
    return s

def ry_layer(state, n, params):
    s = state
    for q in range(params.shape[0]):
        R = ry_matrix(params[q])
        s = apply_single_qubit_gate(s, n, q, R)
    return s

# ---------------- Model ----------------
class QAEModel(nn.Module):
    def __init__(self, n_AB, l, n_ref=0, anc=0, M=20, ent_pattern="cw_fixed", device=device):
        super().__init__()
        self.device = device
        self.n_AB = n_AB; self.l = l; self.k = n_AB - l
        self.n_ref = n_ref; self.anc = anc; self.M = M
        self.ent_pattern = ent_pattern  # 'cw_fixed' or 'cw_ccw_alternating'

        self.nv = self.n_AB
        self.n_qubits = self.n_AB + self.n_ref + self.anc

        total = (M + 1) * self.nv
        self.theta = nn.Parameter(torch.randn(total, dtype=torch.float32, device=self.device) * 0.08)
        print(f"QAEModel: n_qubits={self.n_qubits}, nv={self.nv}, params={total}, ent={self.ent_pattern}, M={self.M}")

    def amplitude_encode_AB(self, x):
        batch = x.shape[0]
        dim = 2 ** self.n_qubits
        out = torch.zeros(batch, dim, dtype=torch.float32, device=self.device)
        shift = self.n_ref + self.anc
        inds = (torch.arange(0, 2**self.n_AB, device=self.device).unsqueeze(0) << shift)
        out.scatter_(1, inds.repeat(batch, 1), x.to(self.device))
        return out

    def forward(self, x):
        batch = x.shape[0]
        state = self.amplitude_encode_AB(x)

        # RY + ring-CNOT with optional cw/ccw alternation
        pidx = 0
        for m in range(self.M):
            thetas = self.theta[pidx:pidx + self.nv]
            state = ry_layer(state, self.n_qubits, thetas)
            pidx += self.nv
            forward_dir = True if self.ent_pattern == "cw_fixed" else (m % 2 == 0)
            state = circular_cnot_dir(state, self.n_qubits, self.nv, forward=forward_dir)
        thetas = self.theta[pidx:pidx + self.nv]
        state = ry_layer(state, self.n_qubits, thetas)

        # probs over B (size 2**l = 8) by marginalizing A, ref, anc
        tensor = state.view(batch, *([2] * self.n_qubits))
        A_axes  = list(range(1, 1 + self.k))
        B_axes  = list(range(1 + self.k, 1 + self.k + self.l))
        Bp_axes = list(range(1 + self.k + self.l, 1 + self.k + self.l + self.n_ref))
        anc_axes= list(range(1 + self.k + self.l + self.n_ref, 1 + self.n_qubits)) if self.anc == 1 else []
        tperm = tensor.permute(*([0] + A_axes + B_axes + Bp_axes + anc_axes)).contiguous()

        A_dim  = 2 ** self.k
        B_dim  = 2 ** self.l
        Bp_dim = 2 ** self.n_ref
        anc_dim= 2 if self.anc == 1 else 1

        tperm = tperm.view(batch, A_dim, B_dim, (Bp_dim if Bp_dim>0 else 1), anc_dim)
        probs_B = (tperm ** 2).sum(dim=(1, 3, 4))  # [batch, 2**l]
        return probs_B  # Pr(B=j) in computational basis

# ---------------- Data ----------------
def preprocess_images_to_32(imgs):
    # center crop 16x16, L2-normalize to 256-dim real amplitudes
    N = imgs.shape[0]
    imgs = imgs.reshape(-1, 28, 28)
    imgs = imgs[:, 6:22, 6:22]
    flat = imgs.reshape(N, -1).astype(np.float32)
    norm = np.linalg.norm(flat, axis=1, keepdims=True)
    norm[norm==0] = 1.0
    flat = flat / norm
    return flat

def make_dataloaders(batch_size=512, max_per_class=2000, classes=list(range(8))):
    train_raw = KMNIST(root='./data', train=True, download=True)
    test_raw  = KMNIST(root='./data', train=False, download=True)
    Xtr_raw = train_raw.data.numpy().reshape(-1,28*28); ytr = train_raw.targets.numpy()
    Xte_raw = test_raw.data.numpy().reshape(-1,28*28); yte = test_raw.targets.numpy()
    mask_tr = np.isin(ytr, classes); mask_te = np.isin(yte, classes)
    Xtr_raw, ytr = Xtr_raw[mask_tr], ytr[mask_tr]
    Xte_raw, yte = Xte_raw[mask_te], yte[mask_te]
    rng = np.random.default_rng(42)
    idx = np.concatenate([rng.choice(np.where(ytr==c)[0], min(max_per_class,(ytr==c).sum()), replace=False) for c in classes])
    Xtr_raw, ytr = Xtr_raw[idx], ytr[idx]
    Xtr = preprocess_images_to_32(Xtr_raw); Xte = preprocess_images_to_32(Xte_raw)
    train_ds = TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long))
    test_ds  = TensorDataset(torch.tensor(Xte, dtype=torch.float32), torch.tensor(yte, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader

# ---------------- STE & loss ----------------
def ste(noisy, clean):
    return clean + (noisy - clean).detach()

def overlap_loss(Fy_eff):
    Fy_eff = torch.clamp(Fy_eff, eps, 1.0)
    return -torch.log(Fy_eff).mean()

# ---------------- Train/Eval ----------------
def train_eval():
    # === Best trial hyperparams ===
    M = 80
    ent_pattern = 'cw_alternating'
    batch_size = 512
    shots_label = 3247

    lr = 0.0037600731932887215
    weight_decay = 0.000249762197126322
    optimizer_name = 'RMSprop'
    scheduler_name = 'CosineWarmRestarts'
    T_0, T_mult, eta_min = 400, 2, 3.932514087662826e-06

    epochs = 100
    classes = list(range(8))

    # ---- Data ----
    train_loader, test_loader = make_dataloaders(batch_size=batch_size, max_per_class=2000, classes=classes)

    # ---- Model ----
    model = QAEModel(n_AB=n_AB, l=l, n_ref=n_ref, anc=anc, M=M, ent_pattern=ent_pattern, device=device).to(device)

    # ---- Optimizer & Scheduler ----
    optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)

    print(f"\n--- Using best hyperparams ---\n"
          f"M={M}, ent_pattern={ent_pattern}, batch_size={batch_size}\n"
          f"optimizer={optimizer_name}(lr={lr}, weight_decay={weight_decay})\n"
          f"scheduler={scheduler_name}(T_0={T_0}, T_mult={T_mult}, eta_min={eta_min})\n"
          f"shots_label={shots_label}, epochs={epochs}\n")

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        losses = []

        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            P = model(xb)  # [B, 8] clean class probabilities
            Fy_clean = P.gather(1, yb.view(-1,1)).squeeze(1)

            # Label shot noise + STE (swap-test ancilla-1 prob p1=(1-Fy)/2)
            with torch.no_grad():
                p1 = torch.clamp((1.0 - Fy_clean) * 0.5, 0.0, 1.0)
                k1 = torch.distributions.Binomial(total_count=shots_label, probs=p1).sample()
                p1_hat = k1 / shots_label
                Fy_hat = torch.clamp(1.0 - 2.0 * p1_hat, 0.0, 1.0)
            Fy_eff = ste(Fy_hat, Fy_clean)

            loss = overlap_loss(Fy_eff)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # Per-iteration scheduler step (CosineWarmRestarts supports epoch steps; we keep it per-epoch below)
            losses.append(loss.item())

        # Epoch-level scheduler step for CosineWarmRestarts
        scheduler.step(epoch + 1)

        # Validation (CLEAN eval: no shots)
        model.eval(); corr=0; tot=0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device); yb = yb.to(device)
                P = model(xb)
                pred = P.argmax(1)
                corr += (pred == yb).sum().item()
                tot  += yb.size(0)
        acc = 100.0 * corr / max(1, tot)
        best_acc = max(best_acc, acc)

        print(f"Epoch {epoch+1:3d}/{epochs} | loss={np.mean(losses):.4f} | acc={acc:.2f}% "
              f"| best={best_acc:.2f}% | time={time.time()-t0:.1f}s")

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\nFinal best acc: {best_acc:.2f}%")
    return best_acc

if __name__ == "__main__":
    train_eval()
