# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 03:20:37 2025

@author: AndrejSumShik
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 21:38:19 2025

@author: AndrejSumShik
"""

#!/usr/bin/env python3
# qae_mnist_best_metrics.py
import os, math, json, time, gc, argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST, KMNIST, FashionMNIST

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

# numerical guards
eps = 1e-8
grad_clip = 5.0

# total qubits: AB + ref + anc
n_qubits = n_AB + n_ref + anc
print(f"n_AB={n_AB} (A={k}, B={l}), n_ref={n_ref}, anc={anc}, total_qubits={n_qubits}")

# ---- Sampling helpers ----
def take_first_n_from_loader(dataset, batch_size, n, shuffle=True):
    """Return a tensor of up to n samples (x only) from a dataset via a DataLoader."""
    if n is None or n <= 0:
        raise ValueError("n must be positive for sampling.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
    xs = []
    cnt = 0
    for xb, _ in loader:
        xs.append(xb.to(device))
        cnt += xb.size(0)
        if cnt >= n:
            break
    x = torch.cat(xs, dim=0)
    return x[:n]

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
        self.ent_pattern = ent_pattern  # 'cw_fixed' or anything else -> alternate

        self.nv = self.n_AB
        self.n_qubits = self.n_AB + self.n_ref + self.anc

        total = (M + 1) * self.nv
        self.theta = nn.Parameter(torch.randn(total, dtype=torch.float32, device=self.device) * 0.08)
        print(f"QAEModel: n_qubits={self.n_qubits}, nv={self.nv}, params={total}, ent={self.ent_pattern}, M={self.M}")

    def amplitude_encode_AB(self, x):
        # x: [B, 256] normalized real amplitudes for AB; embed into (AB + ref + anc)
        batch = x.shape[0]
        dim = 2 ** self.n_qubits
        out = torch.zeros(batch, dim, dtype=torch.float32, device=self.device)
        shift = self.n_ref + self.anc
        inds = (torch.arange(0, 2**self.n_AB, device=self.device).unsqueeze(0) << shift)  # place ref/anc at |0...0>
        out.scatter_(1, inds.repeat(batch, 1), x.to(self.device))
        return out

    def evolve_AB(self, state):
        # Apply RY + ring-CNOT on the AB register only (qubits 0..nv-1), leaving ref/anc untouched.
        pidx = 0
        for m in range(self.M):
            thetas = self.theta[pidx:pidx + self.nv]
            state = ry_layer(state, self.n_qubits, thetas)
            pidx += self.nv
            forward_dir = True if self.ent_pattern == "cw_fixed" else (m % 2 == 0)
            state = circular_cnot_dir(state, self.n_qubits, self.nv, forward=forward_dir)
        thetas = self.theta[pidx:pidx + self.nv]
        state = ry_layer(state, self.n_qubits, thetas)
        return state

    def forward_with_state(self, x):
        """Return class probabilities over B and the AB state vector (BATCHED) before measurement."""
        batch = x.shape[0]
        state = self.amplitude_encode_AB(x)         # [B, 2^(AB+ref+anc)]
        state = self.evolve_AB(state)               # operations affect only AB subspace

        # Slice out the AB amplitudes (ref and anc assumed in |0..0>)
        shift = self.n_ref + self.anc
        idxs = (torch.arange(0, 2**self.n_AB, device=self.device) << shift)
        ab_state = state.index_select(dim=1, index=idxs)   # [B, 2^n_AB]
        # Now compute probs over B by marginalizing A from ab_state
        tensor = ab_state.view(batch, *([2] * self.n_AB))  # only AB
        A_axes = list(range(1, 1 + self.k))
        B_axes = list(range(1 + self.k, 1 + self.k + self.l))
        tperm = tensor.permute(*([0] + A_axes + B_axes)).contiguous()
        A_dim  = 2 ** self.k
        B_dim  = 2 ** self.l
        tperm = tperm.view(batch, A_dim, B_dim)
        probs_B = (tperm ** 2).sum(dim=1)  # [batch, 2**l]
        return probs_B, ab_state  # probabilities on B, and AB pure state amplitudes

    def forward(self, x):
        probs_B, _ = self.forward_with_state(x)
        return probs_B

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

def make_dataloaders(which="KMNIST", batch_size=512, max_per_class=2000, classes=list(range(8))):
    if which == "MNIST":
        Train = MNIST; Test = MNIST
    elif which == "FASHION":
        Train = FashionMNIST; Test = FashionMNIST
    else:
        Train = KMNIST; Test = KMNIST

    train_raw = Train(root='./data', train=True, download=True)
    test_raw  = Test(root='./data',  train=False, download=True)
    Xtr_raw = train_raw.data.numpy().reshape(-1,28*28); ytr = train_raw.targets.numpy()
    Xte_raw = test_raw.data.numpy().reshape(-1,28*28);  yte = test_raw.targets.numpy()
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

# ---------------- Loss & STE ----------------
def ste(noisy, clean): return clean + (noisy - clean).detach()
def overlap_loss(Fy_eff):
    Fy_eff = torch.clamp(Fy_eff, eps, 1.0)
    return -torch.log(Fy_eff).mean()

# ---------------- Inits ----------------
def reinit_params(model: QAEModel, how: str):
    """Different init distributions over theta to average pre-train metrics."""
    with torch.no_grad():
        if how == "normal_0.08":
            model.theta.normal_(0.0, 0.08)
        elif how == "normal_0.3":
            model.theta.normal_(0.0, 0.3)
        elif how == "uniform_0.2":
            a = 0.2
            model.theta.uniform_(-a, a)
        elif how == "uniform_pi":
            a = math.pi
            model.theta.uniform_(-a, a)
        else:  # fallback
            model.theta.normal_(0.0, 0.08)

# ---------------- Metric helpers ----------------
def flatten_params(params: List[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat([p.view(-1) for p in params])

def zero_grads(model: nn.Module):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

def empirical_fim(model: QAEModel, loader: DataLoader, max_batches=2) -> torch.Tensor:
    """Empirical Fisher: F = (1/N) Σ g g^T with g = ∂ log P(y|x)/∂θ at current θ."""
    model.eval()
    P = sum(p.numel() for p in model.parameters())
    F = torch.zeros((P, P), dtype=torch.float64, device=device)
    n = 0
    batches = 0
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        with torch.set_grad_enabled(True):
            probs, _ = model.forward_with_state(xb)
            Fy = probs.gather(1, yb.view(-1,1)).squeeze(1)
            logFy = torch.log(torch.clamp(Fy, eps, 1.0))

            for i in range(logFy.shape[0]):
                zero_grads(model)
                logFy[i].backward(retain_graph=True)
                g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in model.parameters()]).to(torch.float64)
                F += torch.outer(g, g)
                n += 1
        batches += 1
        if batches >= max_batches: break

    if n > 0:
        F /= n
    return F.detach()

def qfim_from_state_autograd(model: QAEModel, xb: torch.Tensor, max_state_components: int = None) -> torch.Tensor:
    """Quantum Fisher for pure states averaged over the batch:
       F = 4 Re( J^T J - a a^T ), where J_{k,i} = ∂ ψ_k / ∂ θ_i, a_i = Σ_k ψ_k ∂ψ_k/∂θ_i.
       Uses exact autograd on the AB state (real here), looping over state components."""
    model.eval()
    P = sum(p.numel() for p in model.parameters())
    F_total = torch.zeros((P, P), dtype=torch.float64, device=device)
    B = xb.shape[0]

    probs, ab_state = model.forward_with_state(xb)  # ab_state: [B, 256], real
    D = ab_state.shape[1]
    idxs = list(range(D))
    if (max_state_components is not None) and (max_state_components < D):
        # sample fixed subset (deterministic for reproducibility)
        idxs = idxs[:max_state_components]

    for b in range(B):
        s = ab_state[b]  # [D]
        # Gram and "a" for this sample
        G = torch.zeros((P, P), dtype=torch.float64, device=device)
        a = torch.zeros((P,), dtype=torch.float64, device=device)

        for k in idxs:
            zero_grads(model)
            s[k].backward(retain_graph=True)
            g_list = [ (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in model.parameters() ]
            g = torch.cat(g_list).to(torch.float64)
            G += torch.outer(g, g)
            a += s[k].detach().to(torch.float64) * g

        F_sample = 4.0 * (G - torch.outer(a, a))
        F_total += F_sample

    F_total /= B
    return F_total.detach()

def eigen_spectrum_stats(F: torch.Tensor) -> Dict[str, object]:
    """Return eigenvalues and several effective-rank/dimension summaries."""
    # numerical symmetrization
    F = 0.5 * (F + F.T)
    # safe to CPU for eigvalsh
    evals = torch.linalg.eigvalsh(F.to(torch.float64)).clamp(min=0).cpu().numpy()
    tr = float(evals.sum())
    l2 = float((evals**2).sum())
    # participation ratio (a.k.a. quadratic effective rank)
    erank_quadratic = (tr*tr) / l2 if l2 > 0 else 0.0
    # entropy-based effective rank
    p = evals / (tr + 1e-20)
    p = p[p > 0]
    H = float(-(p * np.log(p)).sum())
    erank_entropy = float(np.exp(H))
    # effective dimension curve: deff(gamma) = Σ λ/(λ+γ)
    gammas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    deff = { f"{g:.0e}": float(((evals)/(evals+g)).sum()) for g in gammas }
    return {
        "trace": tr,
        "sum_sq": l2,
        "erank_quadratic": erank_quadratic,
        "erank_entropy": erank_entropy,
        "deff": deff,
        "eigs": evals.tolist()
    }

def entanglement_entropy_bits(ab_state_vec: torch.Tensor, n_total=8, nA=4) -> Tuple[float, float]:
    """Von Neumann entanglement entropy S(ρ_A): returns (nats, bits) for a single AB state vector."""
    # reshape |ψ⟩ into (2^nA, 2^(nB)) matrix and take Schmidt values
    A = nA; B = n_total - nA
    psi = ab_state_vec.view(2**A, 2**B)  # real matrix (here)
    # SVD
    S = torch.linalg.svdvals(psi)
    p = (S**2)
    p = p / (p.sum() + 1e-20)
    p = p.clamp(min=1e-20)
    S_nats = float(-(p * torch.log(p)).sum().item())
    S_bits = S_nats / math.log(2.0)
    return S_nats, S_bits

def porter_thomas_w1_kl(prob_vectors: torch.Tensor, bins=200) -> Dict[str, float]:
    """Compare empirical measurement probabilities vs Haar (Porter-Thomas).
       We aggregate all probabilities, scale by d, and compare to Exp(1).
       W1 is approximated via mean |Q_emp(u) - Q_exp(u)| over u in [0,1].
       KL is binned KL with analytic Exp(1) bin masses."""
    d = prob_vectors.shape[1]
    t = (prob_vectors.reshape(-1) * d).detach().cpu().numpy()  # should follow Exp(1) under Haar
    t = t[t > 0]
    # W1 via quantiles
    u = np.linspace(1e-4, 1-1e-4, 2000)
    q_emp = np.quantile(t, u)
    q_the = -np.log(1.0 - u)  # Exp(1) quantile
    W1 = float(np.mean(np.abs(q_emp - q_the)))

    # KL with finite bins up to 99.9% quantile
    max_x = float(np.quantile(t, 0.999))
    edges = np.linspace(0.0, max_x, bins+1)
    hist, _ = np.histogram(t, bins=edges, density=True)   # density
    widths = np.diff(edges)
    p = hist * widths                                   # empirical mass per bin
    # Theoretical mass for Exp(1): ∫_a^b e^{-x} dx = e^{-a} - e^{-b}
    F = lambda x: 1.0 - np.exp(-x)
    q = F(edges[1:]) - F(edges[:-1])
    # smooth
    eps_kl = 1e-12
    p = np.clip(p, eps_kl, None)
    q = np.clip(q, eps_kl, None)
    KL = float(np.sum(p * (np.log(p) - np.log(q))))
    return {"W1": W1, "KL": KL}

# ---------------- Training (your best hyperparams) ----------------
BEST = dict(
    M=48,
    ent_pattern='cw_alternating',  # alternates since != "cw_fixed"
    batch_size=512,
    shots_label=3247,
    lr=0.0037600731932887215,
    weight_decay=0.000249762197126322,
    optimizer_name='RMSprop',
    scheduler_name='CosineWarmRestarts',
    T_0=400, T_mult=2, eta_min=3.932514087662826e-06,
    epochs=100,
    classes=list(range(8)),
)

# ---------------- Orchestration ----------------
@dataclass
class MetricBundle:
    pre_avg_fim: Dict[str, object]
    pre_avg_qfim: Dict[str, object]
    pre_avg_entropies: Dict[str, float]
    pre_haar: Dict[str, float]
    post_fim: Dict[str, object]
    post_qfim: Dict[str, object]
    post_entropies: Dict[str, float]
    test_acc: float
    best_acc: float

def compute_pre_metrics_avg(
    dataset_name: str,
    init_modes: List[str],
    eval_loader: DataLoader,
    build_model_fn,
    qfim_max_batches:int=1,                # kept for signature
    qfim_max_samples:int=32,               # ↑ (moderate)
    qfim_max_state_components:int=256,     # full 8-qubit state
    fim_max_batches:int=4,                 # ↑ (moderate)
    ent_max_samples:int=128,               # decoupled; actually used
    haar_max_samples:int=1024              # ↑ (moderate)
) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, float], Dict[str, float]]:
    """Average pre-train metrics across multiple initializations."""
    fim_eigs = []
    qfim_eigs = []
    ent_44_list = []
    ent_3v5_list = []
    haar_probs_collect = []

    eval_dataset = eval_loader.dataset

    for mode in init_modes:
        model = build_model_fn().to(device)
        reinit_params(model, mode)

        # FIM (empirical): use more batches
        F = empirical_fim(model, eval_loader, max_batches=fim_max_batches)
        fim_eigs.append(eigen_spectrum_stats(F))

        # QFIM: randomized subset
        xb_qfim = take_first_n_from_loader(
            eval_dataset, batch_size=BEST["batch_size"], n=qfim_max_samples, shuffle=True
        )
        QF = qfim_from_state_autograd(model, xb_qfim, max_state_components=qfim_max_state_components)
        qfim_eigs.append(eigen_spectrum_stats(QF))

        # Entanglement (decoupled from QFIM)
        with torch.no_grad():
            xb_ent = take_first_n_from_loader(
                eval_dataset, batch_size=BEST["batch_size"], n=ent_max_samples, shuffle=True
            )
            _, ab_states = model.forward_with_state(xb_ent)
            for i in range(ab_states.shape[0]):
                s = ab_states[i]
                _, Sb4 = entanglement_entropy_bits(s, n_total=8, nA=4)  # 4|4
                _, Sb3 = entanglement_entropy_bits(s, n_total=8, nA=3)  # 3|5
                ent_44_list.append(Sb4)
                ent_3v5_list.append(Sb3)

            # Haar / Porter–Thomas: aggregate over more samples
            probs_all = []
            taken = 0
            for xb, _ in DataLoader(eval_dataset, batch_size=BEST["batch_size"], shuffle=True, pin_memory=True):
                xb = xb.to(device)
                probs, ab = model.forward_with_state(xb)
                ab_probs = (ab ** 2)  # [B, 256]
                probs_all.append(ab_probs.detach())
                taken += xb.size(0)
                if taken >= haar_max_samples:
                    break
            PAB = torch.cat(probs_all, dim=0)[:haar_max_samples]
            haar_stats = porter_thomas_w1_kl(PAB)
            haar_probs_collect.append(haar_stats)

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # average dictionaries
    def avg_spec(specs: List[Dict[str, object]]) -> Dict[str, object]:
        if not specs: return {}
        keys = ["trace","sum_sq","erank_quadratic","erank_entropy"]
        out = {k: float(np.mean([s[k] for s in specs])) for k in keys}
        gammas = list(specs[0]["deff"].keys())
        out["deff"] = {g: float(np.mean([s["deff"][g] for s in specs])) for g in gammas}
        out["eigs"] = specs[0]["eigs"]
        return out

    pre_avg_fim   = avg_spec(fim_eigs)
    pre_avg_qfim  = avg_spec(qfim_eigs)
    pre_avg_ents  = {
        "S_bits_4v4_mean": float(np.mean(ent_44_list)) if ent_44_list else 0.0,
        "S_bits_3v5_mean": float(np.mean(ent_3v5_list)) if ent_3v5_list else 0.0,
    }
    pre_avg_haar  = {
        "W1_mean": float(np.mean([h["W1"] for h in haar_probs_collect])) if haar_probs_collect else 0.0,
        "KL_mean": float(np.mean([h["KL"] for h in haar_probs_collect])) if haar_probs_collect else 0.0,
    }
    return pre_avg_fim, pre_avg_qfim, pre_avg_ents, pre_avg_haar

def compute_post_metrics(
    model: QAEModel,
    eval_loader: DataLoader,
    qfim_max_samples:int=32,               # ↑
    qfim_max_state_components:int=256,
    fim_max_batches:int=4,                 # ↑
    ent_max_samples:int=128                # decoupled
) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, float]]:
    eval_dataset = eval_loader.dataset

    # FIM
    F = empirical_fim(model, eval_loader, max_batches=fim_max_batches)
    post_fim = eigen_spectrum_stats(F)

    # QFIM
    xb_qfim = take_first_n_from_loader(
        eval_dataset, batch_size=BEST["batch_size"], n=qfim_max_samples, shuffle=True
    )
    QF = qfim_from_state_autograd(model, xb_qfim, max_state_components=qfim_max_state_components)
    post_qfim = eigen_spectrum_stats(QF)

    # Entanglement (decoupled)
    with torch.no_grad():
        xb_ent = take_first_n_from_loader(
            eval_dataset, batch_size=BEST["batch_size"], n=ent_max_samples, shuffle=True
        )
        _, ab_states = model.forward_with_state(xb_ent)
        ent_44 = []; ent_3v5=[]
        for i in range(ab_states.shape[0]):
            s = ab_states[i]
            _, Sb4 = entanglement_entropy_bits(s, n_total=8, nA=4)
            _, Sb3 = entanglement_entropy_bits(s, n_total=8, nA=3)
            ent_44.append(Sb4); ent_3v5.append(Sb3)

    post_ents = {
        "S_bits_4v4_mean": float(np.mean(ent_44)) if ent_44 else 0.0,
        "S_bits_3v5_mean": float(np.mean(ent_3v5)) if ent_3v5 else 0.0,
    }
    return post_fim, post_qfim, post_ents

def train_once(dataset_name: str) -> MetricBundle:
    # ---- Data ----
    train_loader, test_loader = make_dataloaders(
        which=dataset_name,
        batch_size=BEST["batch_size"],
        max_per_class=2000,
        classes=BEST["classes"]
    )
    # ---- Build model ----
    def build_model():
        return QAEModel(n_AB=n_AB, l=l, n_ref=n_ref, anc=anc,
                        M=BEST["M"], ent_pattern=BEST["ent_pattern"], device=device)
    # ---- PRE: metrics averaged over multiple inits ----
    init_modes = ["normal_0.08", "normal_0.3", "uniform_0.2", "uniform_pi"]
    pre_fim, pre_qfim, pre_ents, pre_haar = compute_pre_metrics_avg(
        dataset_name,
        init_modes,
        test_loader,  # evaluation loader is fine for pre-metrics
        build_model_fn=build_model,
        qfim_max_batches=1,
        qfim_max_samples=32,
        qfim_max_state_components=256,
        fim_max_batches=4,
        ent_max_samples=128,
        haar_max_samples=1024
    )

    # ---- Model for training (use your best init std=0.08) ----
    model = build_model().to(device)
    reinit_params(model, "normal_0.08")

    # ---- Optimizer & Scheduler ----
    optimizer = optim.RMSprop(model.parameters(), lr=BEST["lr"], weight_decay=BEST["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=BEST["T_0"], T_mult=BEST["T_mult"], eta_min=BEST["eta_min"]
    )

    print(f"\n--- Training {dataset_name} with best hyperparams ---")
    print(f"M={BEST['M']}, ent={BEST['ent_pattern']}, batch={BEST['batch_size']}, epochs={BEST['epochs']}")
    best_acc = 0.0
    for epoch in range(BEST["epochs"]):
        model.train(); t0=time.time(); losses=[]
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            P, _ = model.forward_with_state(xb)  # clean class probabilities
            Fy_clean = P.gather(1, yb.view(-1,1)).squeeze(1)

            # Label shot noise + STE (ancilla-1 prob p1=(1-Fy)/2)
            with torch.no_grad():
                p1 = torch.clamp((1.0 - Fy_clean) * 0.5, 0.0, 1.0)
                k1 = torch.distributions.Binomial(total_count=BEST["shots_label"], probs=p1).sample()
                p1_hat = k1 / BEST["shots_label"]
                Fy_hat = torch.clamp(1.0 - 2.0 * p1_hat, 0.0, 1.0)
            Fy_eff = ste(Fy_hat, Fy_clean)

            loss = overlap_loss(Fy_eff)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step(epoch + 1)

        # Validation (CLEAN)
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
        print(f"Epoch {epoch+1:3d}/{BEST['epochs']} | loss={np.mean(losses):.4f} | acc={acc:.2f}% | best={best_acc:.2f}% | t={time.time()-t0:.1f}s")

        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()

    # Final accuracy
    test_acc = acc

    # ---- POST: metrics on trained weights ----
    post_fim, post_qfim, post_ents = compute_post_metrics(
        model, test_loader,
        qfim_max_samples=32,
        qfim_max_state_components=256,
        fim_max_batches=4,
        ent_max_samples=128
    )

    return MetricBundle(
        pre_avg_fim=pre_fim,
        pre_avg_qfim=pre_qfim,
        pre_avg_entropies=pre_ents,
        pre_haar=pre_haar,
        post_fim=post_fim,
        post_qfim=post_qfim,
        post_entropies=post_ents,
        test_acc=float(test_acc),
        best_acc=float(best_acc),
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["KMNIST","MNIST","FASHION"],
                        help="Which datasets to run: KMNIST MNIST FASHION")
    parser.add_argument("--outdir", type=str, default=".")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    all_results = {}

    for ds in args.datasets:
        mb = train_once(ds)
        out = asdict(mb)
        all_results[ds] = out
        out_path = os.path.join(args.outdir, f"results2_{ds}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved metrics -> {out_path}")

    # also dump a combined file
    combo = os.path.join(args.outdir, "results2_ALL.json")
    with open(combo, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved combined metrics -> {combo}")

if __name__ == "__main__":
    main()
