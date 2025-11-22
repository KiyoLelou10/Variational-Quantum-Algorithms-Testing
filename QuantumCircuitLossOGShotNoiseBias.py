#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KMNIST — AB=8 (A=5,B=3), M=50, RZ every 5 layers, plus one ancilla in |+>
After final RY: apply CRY from ancilla → the 3 readout qubits (B).
Train 100 epochs; report final best train/eval accuracy.

Variants:
  VARIANT=1: three independent CRY angles (no ancilla RZ).
  VARIANT=2: one shared CRY angle for all three B qubits + a trainable RZ on ancilla before CRY.

Updated:
- Adds shot-noise + STE and an entropy penalty using P_eff during training.
- Evaluation remains CLEAN (uses P_clean).
"""

import os, math, gc, json, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import KMNIST

# ---------------- Variant switch ----------------
VARIANT = 1   # <-- set to 2 to enable (shared CRY + ancilla RZ) variant

# ---------- Device & seed ----------
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, "| VARIANT:", VARIANT)

# ---------- Hyperparams (from your BO where useful) ----------
BEST_LR = 0.0049915219748095
BEST_WD = 9.764760471842949e-05
BEST_BS = 128
BEST_PCT_START = 0.27221018153060467
BEST_DIV = 11.194702721840567
BEST_FINAL_DIV = 960.7644900306943
GRAD_CLIP = 5.0

# ----- Shot-noise + entropy penalty knobs -----
SHOTS_LABEL = 4087
BEST_LAMBDA_H = 0.029314339884667882
EPS_ENT = 1e-12

def ste(noisy: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    return clean + (noisy - clean).detach()

@torch.no_grad()
def apply_shot_noise_to_probs(P: torch.Tensor, shots: int) -> torch.Tensor:
    # Clamp to valid probability range
    P = torch.clamp(P, 0.0, 1.0)
    # Symmetric Bernoulli model per class
    p1 = torch.clamp((1.0 - P) * 0.5, 0.0, 1.0)
    k = torch.distributions.Binomial(total_count=shots, probs=p1).sample()
    p1_hat = k / shots
    # Map back to class probability proxy
    P_hat = torch.clamp(1.0 - 2.0 * p1_hat, 0.0, 1.0)
    return P_hat

def entropy_of_distribution(P: torch.Tensor, eps: float = EPS_ENT) -> torch.Tensor:
    Z = P.sum(dim=1, keepdim=True).clamp_min(eps)
    Pn = (P / Z).clamp_min(eps)
    return -(Pn * Pn.log()).sum(dim=1)

# ---------- Layout ----------
n_AB = 8     # A+B
l = 3        # B size => classes = 8
k = n_AB - l # A size = 5
anc = 1      # <- we add ONE ancilla qubit
n_qubits = n_AB + anc  # (no ref qubits)
M = 50

# ---------- Gates ----------
def ry_matrix(angle):
    c = torch.cos(angle / 2.0)
    s = torch.sin(angle / 2.0)
    return torch.stack([torch.stack([c, -s], dim=-1),
                        torch.stack([s,  c], dim=-1)], dim=-2)

def rz_matrix(angle):
    half = angle / 2.0
    c0 = torch.exp(-1j * half)
    c1 = torch.exp( 1j * half)
    z = torch.zeros_like(c0, dtype=torch.complex64)
    return torch.stack([
        torch.stack([c0.to(torch.complex64), z], dim=-1),
        torch.stack([z, c1.to(torch.complex64)], dim=-1)
    ], dim=-2)

# Controlled-RY (control first, target second) 4x4
def cry_matrix(angle):
    R = ry_matrix(angle).to(torch.complex64)
    M4 = torch.zeros(4,4, dtype=torch.complex64)
    # |00>, |01> unchanged
    M4[0,0] = 1.0
    M4[1,1] = 1.0
    # control=1 → apply RY on target subspace {|10>,|11>}
    M4[2,2] = R[0,0]; M4[2,3] = R[0,1]
    M4[3,2] = R[1,0]; M4[3,3] = R[1,1]
    return M4

CNOT_2 = torch.tensor([[1,0,0,0],
                       [0,1,0,0],
                       [0,0,0,1],
                       [0,0,1,0]], dtype=torch.float32, device=device)

# ---------- Tensor helpers ----------
def apply_single_qubit_gate(state, n, qubit_idx, gate_2x2):
    B = state.shape[0]
    t = state.view(B, *([2]*n))
    axis = 1 + qubit_idx
    axes = [0] + [a for a in range(1, n+1) if a != axis] + [axis]
    tperm = t.permute(*axes).contiguous()
    rest = int(np.prod(tperm.shape[1:-1])) if tperm.dim()>2 else 1
    tflat = tperm.view(B, rest, 2)
    G = gate_2x2.to(dtype=tflat.dtype, device=tflat.device)
    out = torch.einsum('bvi,ij->bvj', tflat, G)
    out = out.view(*tperm.shape)
    inv_perm = [0]*(n+1)
    for i,a in enumerate(axes): inv_perm[a] = i
    return out.permute(*inv_perm).contiguous().view(B, -1)

def apply_two_qubit_gate(state, n, q1, q2, gate4):
    B = state.shape[0]
    t = state.view(B, *([2]*n))
    ax1 = 1 + q1; ax2 = 1 + q2
    axes = [0] + [a for a in range(1, n+1) if a not in (ax1, ax2)] + [ax1, ax2]
    tperm = t.permute(*axes).contiguous()
    leading = tperm.shape[1:-2]
    V = 1
    for d in leading: V *= d
    tflat = tperm.view(B, V, 4)
    G = gate4.to(dtype=tflat.dtype, device=tflat.device)
    out = torch.einsum('bvf,fg->bvg', tflat, G)
    out = out.view(B, *leading, 2, 2)
    inv_perm = [0]*(n+1)
    for i,a in enumerate(axes): inv_perm[a] = i
    return out.permute(*inv_perm).contiguous().view(B, -1)

def circular_cnot_dir(state, n, nv, forward=True):
    s = state
    for q in range(nv):
        control = q
        target = (q + 1) % nv if forward else (q - 1) % nv
        s = apply_two_qubit_gate(s, n, control, target, CNOT_2)
    return s

def ry_layer(state, n, params):
    s = state
    for q in range(params.shape[0]):
        R = ry_matrix(params[q]).to(torch.complex64)
        s = apply_single_qubit_gate(s, n, q, R)
    return s

def rz_layer(state, n, params):
    s = state
    for q in range(params.shape[0]):
        Z = rz_matrix(params[q])
        s = apply_single_qubit_gate(s, n, q, Z)
    return s

# ---------- Model with ancilla |+> and post-final CRY to B ----------
class QAEPlusAncCRY(nn.Module):
    """
    Variational block on A+B (nv = n_AB) with RZ every 5. Ancilla is NOT touched until the end.
    After final RY over A+B:
        VARIANT=1: apply three CRY with independent angles (anc → B[5,6,7]).
        VARIANT=2: apply an RZ on the ancilla first, then three CRY sharing one angle.
    """
    def __init__(self, n_AB=8, l=3, M=50, ent_pattern="cw_ccw_alternating", variant=1, device=device):
        super().__init__()
        self.device = device
        self.n_AB = n_AB; self.l = l; self.k = n_AB - l  # A=5,B=3
        self.M = M; self.ent_pattern = ent_pattern
        self.nv = n_AB
        self.anc_idx = n_AB              # ancilla is the last qubit
        self.n_qubits = n_AB + 1         # no ref qubits
        self.variant = variant

        # Params for the main ansatz
        self.n_ry_blocks = M + 1
        self.n_rz_blocks = M // 5
        self.theta_ry = nn.Parameter(torch.randn(self.n_ry_blocks * self.nv, dtype=torch.float32, device=device) * 0.08)
        self.theta_rz = nn.Parameter(torch.randn(self.n_rz_blocks * self.nv, dtype=torch.float32, device=device) * 0.03)

        # Post-final control stage params
        if self.variant == 1:
            # Three independent CRY angles
            self.theta_cry = nn.Parameter(torch.zeros(3, dtype=torch.float32, device=device))
            self.theta_anc_rz = None
        else:
            # One shared CRY angle + ancilla RZ
            self.theta_cry_shared = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))
            self.theta_anc_rz = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))

        # Softmax head
        self.beta = nn.Parameter(torch.tensor(1.5, dtype=torch.float32, device=device))
        self.logit_bias = nn.Parameter(torch.zeros(2**self.l, dtype=torch.float32, device=device))

    def amplitude_encode_AB_with_plus_anc(self, x):
        """
        x: [B, 256] real, L2-normalized.
        Prepare |ψ> = (|0>_anc + |1>_anc)/√2 ⊗ |x>_{AB} (no ref qubits).
        """
        B = x.shape[0]
        dim = 2 ** self.n_qubits
        out = torch.zeros(B, dim, dtype=torch.complex64, device=device)
        base_inds = torch.arange(0, 2**self.n_AB, device=device).unsqueeze(0)  # [1, 256]
        inds_anc0 = base_inds
        inds_anc1 = base_inds + (1 << self.n_AB)  # anc is qubit index n_AB (MSB)
        amp = (x.to(device, dtype=torch.float32) / math.sqrt(2.0)).to(torch.complex64)
        out = out.scatter(1, inds_anc0.repeat(B,1), amp) \
                 .scatter(1, inds_anc1.repeat(B,1), amp)
        return out

    def forward_state(self, x):
        state = self.amplitude_encode_AB_with_plus_anc(x)
        idx_ry = 0; idx_rz = 0
        for m in range(self.M):
            # RY on A+B only
            thetas_ry = self.theta_ry[idx_ry:idx_ry + self.nv]
            state = ry_layer(state, self.n_qubits, thetas_ry)
            idx_ry += self.nv
            # RZ every 5 (A+B only)
            if (m + 1) % 5 == 0:
                thetas_rz = self.theta_rz[idx_rz:idx_rz + self.nv]
                state = rz_layer(state, self.n_qubits, thetas_rz)
                idx_rz += self.nv
            # CNOT ring over A+B
            forward_dir = (m % 2 == 0)
            state = circular_cnot_dir(state, self.n_qubits, self.nv, forward=forward_dir)

        # Final RY over A+B
        thetas_ry = self.theta_ry[idx_ry:idx_ry + self.nv]
        state = ry_layer(state, self.n_qubits, thetas_ry)

        # ---- Variant-specific post-final control stage ----
        b_indices = [self.k + i for i in range(self.l)]  # [5,6,7]
        if self.variant == 1:
            # Independent CRY angles
            for j, bq in enumerate(b_indices):
                G = cry_matrix(self.theta_cry[j])
                state = apply_two_qubit_gate(state, self.n_qubits, self.anc_idx, bq, G)
        else:
            # RZ on ancilla first, then shared CRY
            Z = rz_matrix(self.theta_anc_rz)
            state = apply_single_qubit_gate(state, self.n_qubits, self.anc_idx, Z)
            Gshared = cry_matrix(self.theta_cry_shared)
            for bq in b_indices:
                state = apply_two_qubit_gate(state, self.n_qubits, self.anc_idx, bq, Gshared)

        return state

    def probs_over_B(self, x):
        state = self.forward_state(x)                   # [B, 2^n] complex
        Bsz = state.shape[0]
        t = state.view(Bsz, *([2]*self.n_qubits))
        A_axes = list(range(1, 1 + self.k))
        B_axes = list(range(1 + self.k, 1 + self.k + self.l))
        anc_axes = [1 + self.k + self.l]  # anc is last
        tperm = t.permute(*([0] + A_axes + B_axes + anc_axes)).contiguous()
        A_dim = 2**self.k; B_dim = 2**self.l
        tperm = tperm.view(Bsz, A_dim, B_dim, 2)       # [B, A, B, anc]
        P_B = (tperm.abs()**2).sum(dim=(1,3)).real     # marginalize A and anc
        return P_B

    def forward(self, x):
        # Clean logits on P_clean (used by eval); training will call probs_over_B and build P_eff
        P = self.probs_over_B(x)
        logits = self.beta * (2.0 * P - 1.0) + self.logit_bias
        return logits

# ---------- Data ----------
def preprocess_to_256_center_crop(imgs):
    N = imgs.shape[0]
    imgs = imgs.reshape(-1, 28, 28)[:, 6:22, 6:22]  # 16x16
    flat = imgs.reshape(N, -1).astype(np.float32)
    nrm = np.linalg.norm(flat, axis=1, keepdims=True); nrm[nrm==0]=1.0
    return flat / nrm

def make_kmnist_loaders(batch_size=128, max_per_class=2000, classes=list(range(8))):
    train_raw = KMNIST(root='./data', train=True, download=True)
    test_raw  = KMNIST(root='./data', train=False, download=True)
    Xtr_raw = train_raw.data.numpy().reshape(-1, 28*28); ytr = train_raw.targets.numpy()
    Xte_raw = test_raw.data.numpy().reshape(-1, 28*28); yte = test_raw.targets.numpy()
    mask_tr = np.isin(ytr, classes); mask_te = np.isin(yte, classes)
    Xtr_raw, ytr = Xtr_raw[mask_tr], ytr[mask_tr]
    Xte_raw, yte = Xte_raw[mask_te], yte[mask_te]
    rng = np.random.default_rng(SEED)
    idx = np.concatenate([rng.choice(np.where(ytr==c)[0], min(max_per_class, (ytr==c).sum()), replace=False)
                          for c in classes])
    Xtr_raw, ytr = Xtr_raw[idx], ytr[idx]
    Xtr = preprocess_to_256_center_crop(Xtr_raw)
    Xte = preprocess_to_256_center_crop(Xte_raw)
    train_ds = TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long))
    test_ds  = TensorDataset(torch.tensor(Xte, dtype=torch.float32), torch.tensor(yte, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader

# ---------- Optim & sched ----------
def make_optimizer(params, lr, wd): return optim.RMSprop(params, lr=lr, weight_decay=wd)
def make_scheduler(optimizer, total_steps, pct_start, div_factor, final_div):
    return optim.lr_scheduler.OneCycleLR(optimizer,
        max_lr=optimizer.param_groups[0]['lr'],
        total_steps=total_steps, pct_start=pct_start,
        div_factor=div_factor, final_div_factor=final_div)

@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    total, corr = 0, 0
    losses = []
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        # CLEAN evaluation: build logits from clean probabilities
        P_clean = model.probs_over_B(xb)
        logits = model.beta * (2.0 * P_clean - 1.0) + model.logit_bias
        logp = F.log_softmax(logits, dim=1)
        losses.append(F.nll_loss(logp, yb).item())
        corr += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)
    return {'acc': 100.0 * corr / max(1,total),
            'loss': float(np.mean(losses)) if losses else float('nan')}

def main():
    train_loader, test_loader = make_kmnist_loaders(batch_size=BEST_BS)
    model = QAEPlusAncCRY(n_AB=n_AB, l=l, M=M, ent_pattern="cw_ccw_alternating",
                          variant=VARIANT, device=device).to(device)
    optimizer = make_optimizer(model.parameters(), BEST_LR, BEST_WD)
    steps_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, total_steps=100*steps_per_epoch,
                               pct_start=BEST_PCT_START, div_factor=BEST_DIV, final_div=BEST_FINAL_DIV)
    best_train, best_eval = 0.0, 0.0
    for epoch in range(1, 101):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            # Clean probabilities from circuit
            P_clean = model.probs_over_B(xb)

            # Shot-noisy effective probabilities via STE
            with torch.no_grad():
                P_noisy = apply_shot_noise_to_probs(P_clean, shots=SHOTS_LABEL)
            P_eff = ste(P_noisy, P_clean)

            # Softmax head on P_eff
            logits = model.beta * (2.0 * P_eff - 1.0) + model.logit_bias
            logp = F.log_softmax(logits, dim=1)

            # Base NLL plus entropy reward (subtract entropy term)
            base = F.nll_loss(logp, yb)
            H = entropy_of_distribution(P_eff).mean()
            loss = base - BEST_LAMBDA_H * H

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step(); scheduler.step()

        tr = eval_epoch(model, train_loader); ev = eval_epoch(model, test_loader)
        best_train = max(best_train, tr['acc']); best_eval = max(best_eval, ev['acc'])
        tag = f"+anc-CRY(v{VARIANT})"
        print(f"[KMNIST AB=8 M=50 {tag}] Epoch {epoch:3d} | "
              f"train_acc={tr['acc']:.2f} eval_acc={ev['acc']:.2f} | "
              f"best_train={best_train:.2f} best_eval={best_eval:.2f}")
        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()
    print(f"\nBest train acc: {best_train:.2f}%   Best eval acc: {best_eval:.2f}%")

if __name__ == "__main__":
    main()
