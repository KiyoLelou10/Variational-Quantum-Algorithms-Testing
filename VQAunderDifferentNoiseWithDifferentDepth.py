# -*- coding: utf-8 -*-
"""
Noise-depth evaluation (real-only), UPDATED with Fix #2 (shot-accurate per-replica multinomials):
- Idle dephasing is Z-only.
- No .real anywhere (states are real).
- Angle wrapping moved out of the forward; we wrap params after optimizer.step().
- Noise ops are differentiable (no torch.no_grad); random masks are treated as constants,
  so gradients flow through noisy gates (Monte-Carlo estimator of d E[loss] / dθ).
- Shot realism: for each batch we run K independent noisy circuits, sample multinomial shots
  PER replica, sum counts across replicas, then use STE through the K-mean probabilities.
"""

import os, json, math, time, gc, numpy as np
from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import KMNIST

# ---------------- Config ----------------
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Try Lion optimizer
try:
    from lion_pytorch import Lion as LionOpt
except Exception:
    LionOpt = None

# Fixed QAE layout (real-only path)
n_AB = 8
l = 3
k = n_AB - l
assert k >= 0
n_ref = l
nv = n_AB  # number of data qubits (A+B). No ancilla.

# Training/eval settings
BATCH_SIZE = 512
EPOCHS = 25
MAX_PER_CLASS = 2000
CLASSES = list(range(8))
ENT_PATTERN = "cw_alternating"
LR = 0.002
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0
LAMBDA_ENTROPY = 5e-3
SHOTS_LABEL = 6000

# NEW: K replicas for shot realism
SHOT_REPL_K_TRAIN = 3   # e.g., 2–4
SHOT_REPL_K_EVAL  = 3   # can use more for evaluation if desired

RESULTS_JSON = "noise_depth_sweep_results.json"

# ---------------- Helper (real-only) ----------------
TWO_PI = 2.0 * math.pi
PI = math.pi
RDTYPE = torch.float32  # real-only amplitudes

def wrap_to_pi_inplace_(param: torch.Tensor):
    # Wrap in-place to (-pi, pi]; smooth forward avoids modulo kinks
    with torch.no_grad():
        param.add_(PI).remainder_(TWO_PI).sub_(PI)
    return param

def ry_matrix(angle: torch.Tensor):
    # 2x2 real rotation around Y (NO internal wrap; smooth forward)
    c = torch.cos(angle / 2.0)
    s = torch.sin(angle / 2.0)
    G = torch.stack([torch.stack([c, -s], dim=-1),
                     torch.stack([s,  c], dim=-1)], dim=-2)  # [2,2]
    return G.to(RDTYPE)

# Real Pauli matrices (no Y)
X_2 = torch.tensor([[0,1],[1,0]], dtype=RDTYPE, device=device)
Z_2 = torch.tensor([[1,0],[0,-1]], dtype=RDTYPE, device=device)
# CNOT (control first, target second)
CNOT_2 = torch.tensor([[1,0,0,0],
                       [0,1,0,0],
                       [0,0,0,1],
                       [0,0,1,0]], dtype=RDTYPE, device=device)
# Precompute 2q Paulis
XX_4 = torch.kron(X_2, X_2).to(device)
ZZ_4 = torch.kron(Z_2, Z_2).to(device)

def apply_single_qubit_gate(state, n, q, G2):
    # state: [B, 2^n], real
    B = state.shape[0]
    t = state.view(B, *([2]*n))
    ax = 1 + q
    axes = [0] + [a for a in range(1, n+1) if a != ax] + [ax]
    tperm = t.permute(*axes).contiguous()
    rest = int(np.prod(tperm.shape[1:-1])) if tperm.dim()>2 else 1
    tflat = tperm.view(B, rest, 2)
    out = torch.einsum('bvi,ij->bvj', tflat, G2.to(tflat.device))
    out = out.view(*tperm.shape)
    inv = [0]*(n+1)
    for i,a in enumerate(axes):
        inv[a] = i
    out = out.permute(*inv).contiguous()
    return out.view(B, -1)

def apply_two_qubit_gate(state, n, q1, q2, G4):
    B = state.shape[0]
    t = state.view(B, *([2]*n))
    ax1 = 1 + q1; ax2 = 1 + q2
    axes = [0] + [a for a in range(1,n+1) if a not in (ax1,ax2)] + [ax1,ax2]
    tperm = t.permute(*axes).contiguous()
    leading = tperm.shape[1:-2]
    V = 1
    for d in leading: V *= d
    tflat = tperm.view(B, V, 4)
    out = torch.einsum('bvf,fg->bvg', tflat, G4.to(tflat.device))
    out = out.view(B, *leading, 2, 2)
    inv = [0]*(n+1)
    for i,a in enumerate(axes):
        inv[a] = i
    out = out.permute(*inv).contiguous()
    return out.view(B, -1)

# ---------------- Dataset ----------------
def preprocess_images_to_32(imgs_np: np.ndarray) -> np.ndarray:
    N = imgs_np.shape[0]
    imgs = imgs_np.reshape(-1,28,28)[:, 6:22, 6:22]  # center crop 16x16
    flat = imgs.reshape(N, -1).astype(np.float32)
    norm = np.linalg.norm(flat, axis=1, keepdims=True)
    norm[norm==0] = 1.0
    flat = flat / norm
    return flat

def make_dataloaders(batch_size=BATCH_SIZE, max_per_class=MAX_PER_CLASS, classes=CLASSES):
    train_raw = KMNIST(root='./data', train=True, download=True)
    test_raw  = KMNIST(root='./data', train=False, download=True)
    Xtr_raw = train_raw.data.numpy().reshape(-1,28*28); ytr = train_raw.targets.numpy()
    Xte_raw = test_raw.data.numpy().reshape(-1,28*28); yte = test_raw.targets.numpy()
    mask_tr = np.isin(ytr, classes); mask_te = np.isin(yte, classes)
    Xtr_raw, ytr = Xtr_raw[mask_tr], ytr[mask_tr]
    Xte_raw, yte = Xte_raw[mask_te], yte[mask_te]
    rng = np.random.default_rng(SEED)
    idx = np.concatenate([rng.choice(np.where(ytr==c)[0], min(max_per_class,(ytr==c).sum()), replace=False) for c in classes])
    Xtr_raw, ytr = Xtr_raw[idx], ytr[idx]
    Xtr = preprocess_images_to_32(Xtr_raw); Xte = preprocess_images_to_32(Xte_raw)
    train_ds = TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long))
    test_ds  = TensorDataset(torch.tensor(Xte, dtype=torch.float32), torch.tensor(yte, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader

# ---------------- Noise (real-only, per-gate) ----------------
class NoiseLevel:
    """
    Real-only per-gate noise parameters (level can be < 0 to simulate future-lower noise):
      - p1q_xz: after each RY on that qubit, apply X or Z with small prob (split evenly).
      - p2q_1q_xz: after each CNOT, extra 1q X/Z on control and target.
      - p2q_pair: after each CNOT, correlated pair error: with small prob apply XX or ZZ (chosen uniformly).
      - p_idle_z: per-block idle/entanglement Z flips on all AB qubits (models T2-ish).
      - overrot_sigma: coherent over-rotation noise added to each RY angle.
      - readout_p: symmetric readout bit-flip on B.
      - final_b_depolar_p: final depolarizing on B distribution before readout flips.
    """
    def __init__(self, level: float):
        self.level = float(level)

        def lerp(a, b, t):  # allow extrapolation
            return a + (b - a) * t

        # Base ranges: "very light" (t=0) → "high-medium" (t=1)
        p1q_xz            = lerp(0.0005, 0.0050, self.level)
        p2q_1q_xz         = lerp(0.0015, 0.0100, self.level)
        p2q_pair          = lerp(0.0008, 0.0060, self.level)
        p_idle_z          = lerp(0.0003, 0.0030, self.level)
        overrot_sigma     = lerp(0.0050, 0.0200, self.level)
        readout_p         = lerp(0.0050, 0.0200, self.level)
        final_b_depolar_p = lerp(0.0100, 0.0400, self.level)

        # Clamp to physically valid probabilities / magnitudes
        self.p1q_xz            = max(0.0, p1q_xz)
        self.p2q_1q_xz         = max(0.0, p2q_1q_xz)
        self.p2q_pair          = max(0.0, p2q_pair)
        self.p_idle_z          = max(0.0, p_idle_z)
        self.overrot_sigma     = max(0.0, overrot_sigma)
        self.readout_p         = max(0.0, readout_p)
        self.final_b_depolar_p = max(0.0, final_b_depolar_p)

# -------- Vectorized, differentiable noise helpers --------
def _apply_masked_1q(state, n, q, mask_bool, G2):
    """
    Apply 1q gate G2 to those batch items where mask_bool==True, in a branchless way.
    """
    applied = apply_single_qubit_gate(state, n, q, G2)
    mask = mask_bool.view(-1, 1).to(state.dtype)  # [B,1] float mask {0,1}
    return applied * mask + state * (1.0 - mask)

def _apply_masked_2q(state, n, q1, q2, mask_bool, G4):
    applied = apply_two_qubit_gate(state, n, q1, q2, G4)
    mask = mask_bool.view(-1, 1).to(state.dtype)
    return applied * mask + state * (1.0 - mask)

def one_qubit_xz_channel(state, n, q, p_xz, rng=None):
    """
    Apply X or Z (equiprobable) with total probability p_xz. Differentiable.
    """
    if p_xz <= 0:
        return state
    B = state.shape[0]
    r = torch.rand(B, device=state.device) if rng is None else rng.uniform_(torch.empty(B, device=state.device))
    m = (r < p_xz)
    # Split into X vs Z
    rx = torch.rand(B, device=state.device) if rng is None else rng.uniform_(torch.empty(B, device=state.device))
    mx = m & (rx < 0.5)
    mz = m & (~mx)
    if mx.any():
        state = _apply_masked_1q(state, n, q, mx, X_2)
    if mz.any():
        state = _apply_masked_1q(state, n, q, mz, Z_2)
    return state

def two_qubit_pair_channel(state, n, q1, q2, p_pair, rng=None):
    """
    Correlated 2q error: with prob p_pair, apply XX or ZZ (50/50) on the pair. Differentiable.
    """
    if p_pair <= 0:
        return state
    B = state.shape[0]
    r = torch.rand(B, device=state.device) if rng is None else rng.uniform_(torch.empty(B, device=state.device))
    m = (r < p_pair)
    rx = torch.rand(B, device=state.device) if rng is None else rng.uniform_(torch.empty(B, device=state.device))
    mx = m & (rx < 0.5)
    mz = m & (~mx)
    if mx.any():
        state = _apply_masked_2q(state, n, q1, q2, mx, XX_4)
    if mz.any():
        state = _apply_masked_2q(state, n, q1, q2, mz, ZZ_4)
    return state

def idle_dephase_layer(state, n, qubits: List[int], p_idle_z: float, rng=None):
    """
    Z-only idle dephasing. Differentiable (no no_grad, no in-place).
    """
    if p_idle_z <= 0:
        return state
    for q in qubits:
        # Z-only flips with prob p_idle_z
        B = state.shape[0]
        r = torch.rand(B, device=state.device) if rng is None else rng.uniform_(torch.empty(B, device=state.device))
        mz = (r < p_idle_z)
        if mz.any():
            state = _apply_masked_1q(state, n, q, mz, Z_2)
    return state

def apply_readout_error_B(probs_B, l, p_bitflip):
    if p_bitflip <= 0: return probs_B
    B_dim = probs_B.shape[1]
    probs = probs_B
    for b in range(l):
        idx = torch.arange(B_dim, device=probs.device)
        flip_idx = idx ^ (1 << b)
        probs = (1.0 - p_bitflip) * probs + p_bitflip * probs[:, flip_idx]
    return probs

def apply_final_b_depolarizing(probs_B, p):
    if p <= 0: return probs_B
    B_dim = probs_B.shape[1]
    uniform = torch.full_like(probs_B, 1.0 / B_dim)
    return (1.0 - p) * probs_B + p * uniform

# ---------------- Model ----------------
class QAEReal(nn.Module):
    """
    Real-only QAE with per-gate noise hooks (differentiable).
    """
    def __init__(self, n_AB, l, M, ent_pattern, noise: NoiseLevel, device=device):
        super().__init__()
        self.device = device
        self.n_AB = n_AB; self.l = l; self.k = n_AB - l
        self.M = M; self.ent_pattern = ent_pattern
        self.n_qubits = n_AB + n_ref  # ref copies exist but are marginalized; we don't gate on them
        self.noise = noise

        # Parameters: RY on data qubits only; (M+1) RY stacks like before
        self.theta_ry = nn.Parameter(torch.randn(M + 1, self.n_AB, dtype=torch.float32, device=device) * 0.08)

        # Softmax head params
        self.beta = nn.Parameter(torch.tensor(1.5, dtype=torch.float32, device=device))
        self.logit_bias = nn.Parameter(torch.zeros(2**self.l, dtype=torch.float32, device=device))

    def amplitude_encode_AB(self, x):
        batch = x.shape[0]
        dim = 2 ** self.n_qubits
        out = torch.zeros(batch, dim, dtype=RDTYPE, device=self.device)
        shift = self.l  # n_ref=l; no ancilla
        inds = (torch.arange(0, 2**self.n_AB, device=self.device).unsqueeze(0) << shift)
        out.scatter_(1, inds.repeat(batch, 1), x.to(self.device).to(RDTYPE))
        return out

    def _apply_ry_with_noise(self, state, q, angle):
        # coherent over-rotation per gate (differentiable)
        if self.noise.overrot_sigma > 0:
            angle = angle + torch.randn((), device=state.device) * self.noise.overrot_sigma
        G = ry_matrix(angle)
        state = apply_single_qubit_gate(state, self.n_qubits, q, G)
        # per-gate 1q X/Z channel (differentiable)
        state = one_qubit_xz_channel(state, self.n_qubits, q, self.noise.p1q_xz)
        return state

    def _apply_cnot_with_noise(self, state, c, t):
        state = apply_two_qubit_gate(state, self.n_qubits, c, t, CNOT_2)
        # stronger 1q X/Z on both acted qubits
        state = one_qubit_xz_channel(state, self.n_qubits, c, self.noise.p2q_1q_xz)
        state = one_qubit_xz_channel(state, self.n_qubits, t, self.noise.p2q_1q_xz)
        # correlated pair error (XX or ZZ)
        state = two_qubit_pair_channel(state, self.n_qubits, c, t, self.noise.p2q_pair)
        return state

    def _cnot_ring(self, state, forward=True):
        # ring over the first n_AB data qubits
        nq = self.n_AB
        for q in range(nq):
            c = q
            t = (q + 1) % nq if forward else (q - 1) % nq
            state = self._apply_cnot_with_noise(state, c, t)
        return state

    def forward(self, x):
        B = x.shape[0]
        state = self.amplitude_encode_AB(x)

        # Blocks: [RY layer] → [CNOT ring] with alternating direction; idle dephase per block
        for m in range(self.M):
            # RY layer over data qubits
            angles = self.theta_ry[m]  # [n_AB]
            for q in range(self.n_AB):
                state = self._apply_ry_with_noise(state, q, angles[q])

            # Idle/entanglement Z-dephasing (all data qubits) — Z-only
            state = idle_dephase_layer(state, self.n_qubits, list(range(self.n_AB)), self.noise.p_idle_z)

            # CNOT ring
            forward_dir = True if self.ent_pattern == "cw_fixed" else (m % 2 == 0)
            state = self._cnot_ring(state, forward=forward_dir)

        # Closing RY
        angles = self.theta_ry[self.M]
        for q in range(self.n_AB):
            state = self._apply_ry_with_noise(state, q, angles[q])

        # Measure/marginalize to B (sum out A and ref)
        tensor = state.view(B, *([2]*self.n_qubits))
        A_axes  = list(range(1, 1 + self.k))
        B_axes  = list(range(1 + self.k, 1 + self.k + self.l))
        R_axes  = list(range(1 + self.k + self.l, 1 + self.k + self.l + n_ref))
        tperm = tensor.permute(*([0] + A_axes + B_axes + R_axes)).contiguous()

        A_dim = 2 ** self.k
        B_dim = 2 ** self.l
        R_dim = 2 ** n_ref

        tperm = tperm.view(B, A_dim, B_dim, (R_dim if R_dim>0 else 1))
        probs_B = (tperm.square()).sum(dim=(1,3))  # [B, 2**l], real

        # Final B depolarizing + readout bit-flips (differentiable, deterministic expectations)
        probs_B = apply_final_b_depolarizing(probs_B, self.noise.final_b_depolar_p)
        probs_B = apply_readout_error_B(probs_B, self.l, self.noise.readout_p)
        return probs_B

# ---------------- Shots, STE, and K-replica helpers ----------------
def ste(noisy, clean):
    return clean + (noisy - clean).detach()

def _renorm_rows(P: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    Z = P.sum(dim=1, keepdim=True).clamp_min(eps)
    return (P / Z).clamp_min(eps)

@torch.no_grad()
def sample_counts_from_K(Ps: torch.Tensor, shots: int) -> torch.Tensor:
    """
    Ps: [K, B, C] probabilities from K independent noisy circuit draws.
    Returns summed counts over K replicas: [B, C], with total 'shots' per row.
    Shots are split as evenly as possible across replicas.
    """
    K, B, C = Ps.shape
    counts = torch.zeros(B, C, device=Ps.device, dtype=torch.float32)

    shots_k = shots // K
    extra = shots - shots_k * K

    for k in range(K):
        n_k = shots_k + (1 if k < extra else 0)
        if n_k <= 0:
            continue
        Pk = _renorm_rows(Ps[k])
        # Multinomial over classes for each batch row independently
        # torch.distributions.Multinomial supports batched 'probs' of shape [B, C]
        m = torch.distributions.Multinomial(total_count=n_k, probs=Pk)
        counts_k = m.sample()  # [B, C]
        counts.add_(counts_k)

    return counts  # [B, C]

def entropy_of_distribution(P, eps=1e-12):
    Z = P.sum(dim=1, keepdim=True).clamp_min(eps)
    Pn = (P / Z).clamp_min(eps)
    return -(Pn * Pn.log()).sum(dim=1)

# ---------------- Optimizer & Scheduler ----------------
def make_optimizer(params):
    if LionOpt is not None:
        print("Using Lion optimizer.")
        return LionOpt(params, lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.99))
    print("Lion not installed — falling back to AdamW.")
    return optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

def make_scheduler(optimizer):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, threshold=5e-4, min_lr=1e-6
    )

@torch.no_grad()
def wrap_model_angles_(model: nn.Module):
    if hasattr(model, "theta_ry"):
        wrap_to_pi_inplace_(model.theta_ry)

# ---------------- Runner ----------------
def run_one(M: int, noise_level_idx: int, noise_level: NoiseLevel,
            train_loader, test_loader) -> Dict[str, Any]:
    model = QAEReal(n_AB=n_AB, l=l, M=M, ent_pattern=ENT_PATTERN, noise=noise_level, device=device).to(device)
    optimzr = make_optimizer(model.parameters())
    sched = make_scheduler(optimzr)

    best = dict(train_acc=0.0, train_epoch=-1, eval_acc=0.0, eval_epoch=-1)

    for epoch in range(1, EPOCHS+1):
        t0 = time.time()
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            optimzr.zero_grad(set_to_none=True)

            # ----- K replicas (training): run K times, then sample counts per replica -----
            Ps_list = [model(xb) for _ in range(SHOT_REPL_K_TRAIN)]        # each re-samples gate noise
            Ps = torch.stack(Ps_list, dim=0)                                # [K,B,C]
            P_bar = Ps.mean(dim=0)                                          # [B,C] (clean path for grads)

            with torch.no_grad():
                counts = sample_counts_from_K(Ps, shots=SHOTS_LABEL)        # [B,C] summed counts
                P_hat = counts / float(SHOTS_LABEL)                         # empirical probs

            # STE: loss on P_hat, backprop through P_bar
            P_eff = ste(P_hat, P_bar)
            logits = model.beta * (2.0 * P_eff - 1.0) + model.logit_bias
            logp = F.log_softmax(logits, dim=1)
            base = F.nll_loss(logp, yb)
            H = entropy_of_distribution(P_eff).mean()
            loss = base - LAMBDA_ENTROPY * H

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimzr.step()
            wrap_model_angles_(model)
            losses.append(loss.item())

        # ---------------- Train noisy accuracy (shot-accurate, K replicas) ----------------
        model.eval()
        with torch.no_grad():
            corr_tr = 0; tot_tr = 0
            for xb, yb in train_loader:
                xb = xb.to(device); yb = yb.to(device)
                Ps_list = [model(xb) for _ in range(SHOT_REPL_K_EVAL)]
                Ps = torch.stack(Ps_list, dim=0)                             # [K,B,C]
                counts = sample_counts_from_K(Ps, shots=SHOTS_LABEL)
                P_hat = counts / float(SHOTS_LABEL)
                logits_noisy = model.beta * (2.0 * P_hat - 1.0) + model.logit_bias
                pred = logits_noisy.argmax(1)
                corr_tr += (pred == yb).sum().item()
                tot_tr  += yb.size(0)
            train_acc = 100.0 * corr_tr / max(1, tot_tr)

        # ---------------- Eval noisy accuracy (shot-accurate, K replicas) ----------------
        with torch.no_grad():
            corr_te = 0; tot_te = 0
            for xb, yb in test_loader:
                xb = xb.to(device); yb = yb.to(device)
                Ps_list = [model(xb) for _ in range(SHOT_REPL_K_EVAL)]
                Ps = torch.stack(Ps_list, dim=0)                             # [K,B,C]
                counts = sample_counts_from_K(Ps, shots=SHOTS_LABEL)
                P_hat = counts / float(SHOTS_LABEL)
                logits_noisy = model.beta * (2.0 * P_hat - 1.0) + model.logit_bias
                pred = logits_noisy.argmax(1)
                corr_te += (pred == yb).sum().item()
                tot_te  += yb.size(0)
            eval_acc = 100.0 * corr_te / max(1, tot_te)

        sched.step(eval_acc)

        if train_acc > best["train_acc"] + 1e-12:
            best["train_acc"] = train_acc; best["train_epoch"] = epoch
        if eval_acc > best["eval_acc"] + 1e-12:
            best["eval_acc"] = eval_acc; best["eval_epoch"] = epoch

        print(f"[M={M:2d} | lvl_idx={noise_level_idx:02d}] "
              f"epoch {epoch:03d}/{EPOCHS} | loss={np.mean(losses):.4f} "
              f"| train={train_acc:5.2f}% (best {best['train_acc']:5.2f}%@{best['train_epoch']:02d}) "
              f"| eval={eval_acc:5.2f}% (best {best['eval_acc']:5.2f}%@{best['eval_epoch']:02d}) "
              f"| {time.time()-t0:.1f}s")

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Package results
    return {
        "M": M,
        "noise_level_index": noise_level_idx,
        "best_train_noisy_acc": best["train_acc"],
        "best_train_epoch": best["train_epoch"],
        "best_eval_noisy_acc": best["eval_acc"],
        "best_eval_epoch": best["eval_epoch"]
    }

# ---------------- Main sweep ----------------
if __name__ == "__main__":
    train_loader, test_loader = make_dataloaders()

    M_list = [10,20,30,40,50]

    # 12-step noise ladder: two extra levels below the old 0.0, then the original 20 steps
    levels = np.linspace(0.0, 1.0, 4, dtype=np.float32)
    print("Severity levels (22):", [f"{lv:.3f}" for lv in levels.tolist()])

    all_results: List[Dict[str, Any]] = []
    t_global = time.time()

    for i_level, lvl in enumerate(levels):
        nl = NoiseLevel(level=float(lvl))
        for M in reversed(M_list):
            res = run_one(M, i_level, nl, train_loader, test_loader)
            # Add concrete noise params to the result (for traceability)
            res["noise_params"] = {
                "level": float(lvl),
                "p1q_xz": nl.p1q_xz,
                "p2q_1q_xz": nl.p2q_1q_xz,
                "p2q_pair": nl.p2q_pair,
                "p_idle_z": nl.p_idle_z,
                "overrot_sigma": nl.overrot_sigma,
                "readout_p": nl.readout_p,
                "final_b_depolar_p": nl.final_b_depolar_p,
            }
            all_results.append(res)

            # Save incrementally for safety
            with open(RESULTS_JSON, "w") as f:
                json.dump(all_results, f, indent=2)

    print(f"\nSaved {len(all_results)} runs to {RESULTS_JSON}. Total time: {time.time()-t_global:.1f}s")
    # Quick summary line per level
    for i_level in range(len(levels)):
        subset = [r for r in all_results if r["noise_level_index"] == i_level]
        best_eval = max(subset, key=lambda r: r["best_eval_noisy_acc"])
        print(f"[lvl_idx {i_level:02d} | sev={levels[i_level]:.3f}] best EVAL {best_eval['best_eval_noisy_acc']:.2f}% "
              f"at M={best_eval['M']} (epoch {best_eval['best_eval_epoch']})")
