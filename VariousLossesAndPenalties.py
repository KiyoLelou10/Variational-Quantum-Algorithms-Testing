# qae_biggrid_runner.py
# Experiment driver: 10 losses × penalties (with λ grid) → 10-epoch pilot → select top-2 per loss → 50-epoch finals.

import os, json, math, time, gc, random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import KMNIST

# ------------------------ Global Config ------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

OUTDIR = Path("./results_grid")
OUTDIR.mkdir(parents=True, exist_ok=True)

# -------- Loss selection switch --------
# 1 => ONLY multinomial-bucket losses: lbb, brier, ot, softmax, hinge, softmargin, infonce
# 2 => ONLY binomial-bucket losses: overlap, hellinger, tsallis_*
TYPES_OF_LOSSES = 2  # <<<<<< default as requested

# Data/task
N_AB = 8      # data qubits (A+B)
L = 3         # B qubits => 8 classes
N_REF = 0
USE_ANC = False
ANC = 1 if USE_ANC else 0
K = N_AB - L
assert K >= 0
N_CLASSES = 2**L

# Architecture
M = 40
ENT_PATTERN = "cw_ccw_alternating"

# Optimization & schedule
BATCH = 512
EPOCHS_PILOT = 10
EPOCHS_FINAL = 50
LR = 0.0037600731932887215
WD = 0.000249762197126322
T0, T_MULT, ETA_MIN = 400, 2, 3.932514087662826e-06

# Shots & STE
LABEL_SHOTS = 3247
LAYER_PROBE_SHOTS = 512
PROBED_LAYERS_PER_MINIBATCH = 2
EMA_DECAY = 0.1
PENALTY_STE = True   # use STE for per-layer penalty signals

# Numerical guards
EPS = 1e-12
GRAD_CLIP = 5.0

# Penalty λ grids
LAMBDA_GRID = {
    "entropy": [0.005, 0.02, 0.05],
    "com_maggrad": [0.001, 0.005, 0.02],
    "com_fisher": [0.001, 0.005, 0.02],
    "facility": [0.001, 0.003, 0.01],
    "coverage": [0.001, 0.003, 0.01],
}

# Concave-over-modular mix coefficient β (fixed)
COM_BETA = 0.5  # ai = (1-β)*||θ|| + β*EMA_gradnorm

# Tsallis αs
TSALLIS_ALPHAS = [1.2, 1.5, 2.0]

# Soft-margin & InfoNCE temperatures / Hinge margin
TAU_SOFTMARGIN = 0.1
TAU_INFONCE = 0.1
HINGE_MARGIN = 0.05

# OT Sinkhorn hyperparams
OT_EPS = 0.05
OT_ITERS = 80

# Force entropy penalty to use multinomial shots regardless of primary loss
ENTROPY_ALWAYS_MULTINOMIAL = True

# ------------------------ Utilities ------------------------
def hamming3_cost_matrix(Kclasses=8, device=DEVICE):
    C = torch.zeros(Kclasses, Kclasses, dtype=torch.float32, device=device)
    for i in range(Kclasses):
        for j in range(Kclasses):
            bi = [(i>>b)&1 for b in range(3)]
            bj = [(j>>b)&1 for b in range(3)]
            C[i,j] = sum(1 if bi[b]!=bj[b] else 0 for b in range(3))
    return C

def preprocess_16x16_L2(x28x28: np.ndarray):
    N = x28x28.shape[0]
    imgs = x28x28.reshape(N, 28, 28)
    imgs = imgs[:, 6:22, 6:22]
    flat = imgs.reshape(N, -1).astype(np.float32)
    norm = np.linalg.norm(flat, axis=1, keepdims=True)
    norm[norm==0] = 1.0
    flat = flat / norm
    return flat

def make_kmnist_loaders(batch=BATCH, max_per_class=2000, classes=list(range(8))):
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
    Xtr = preprocess_16x16_L2(Xtr_raw); Xte = preprocess_16x16_L2(Xte_raw)
    train_ds = TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long))
    test_ds  = TensorDataset(torch.tensor(Xte, dtype=torch.float32), torch.tensor(yte, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,  pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch, shuffle=False, pin_memory=True)
    return train_loader, test_loader

# ------------------------ Quantum Core ------------------------
def ry_matrix(angle):
    c = torch.cos(angle / 2.0)
    s = torch.sin(angle / 2.0)
    return torch.stack([torch.stack([c, -s], dim=-1),
                        torch.stack([s,  c], dim=-1)], dim=-2)

CNOT_2 = torch.tensor([[1,0,0,0],
                       [0,1,0,0],
                       [0,0,0,1],
                       [0,0,1,0]], dtype=torch.float32, device=DEVICE)

def apply_single_qubit_gate(state, n, q, G2):
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
    for i,a in enumerate(axes): inv[a]=i
    return out.permute(*inv).contiguous().view(B,-1)

def apply_two_qubit_gate(state, n, q1, q2, G4):
    B = state.shape[0]
    t = state.view(B, *([2]*n))
    ax1 = 1 + q1; ax2 = 1 + q2
    axes = [0] + [a for a in range(1, n+1) if a not in (ax1, ax2)] + [ax1, ax2]
    tperm = t.permute(*axes).contiguous()
    lead = tperm.shape[1:-2]
    V = 1
    for d in lead: V *= d
    tflat = tperm.view(B, V, 4)
    out = torch.einsum('bvf,fg->bvg', tflat, G4.to(tflat.device))
    out = out.view(B, *lead, 2, 2)
    inv = [0]*(n+1)
    for i,a in enumerate(axes): inv[a]=i
    return out.permute(*inv).contiguous().view(B,-1)

def ring_cnot_dir(state, n, nv, forward=True):
    s = state
    for q in range(nv):
        ctrl = q
        tgt = (q+1)%nv if forward else (q-1)%nv
        s = apply_two_qubit_gate(s, n, ctrl, tgt, CNOT_2)
    return s

def ry_layer(state, n, params):
    s = state
    for q in range(params.shape[0]):
        s = apply_single_qubit_gate(s, n, q, ry_matrix(params[q]))
    return s

class QAEModel(nn.Module):
    def __init__(self, n_ab=N_AB, l=L, n_ref=N_REF, anc=ANC, M=M, ent_pattern=ENT_PATTERN, device=DEVICE):
        super().__init__()
        self.device = device
        self.n_AB = n_ab; self.l = l; self.k = n_ab - l
        self.n_ref = n_ref; self.anc = anc; self.M = M
        self.ent_pattern = ent_pattern
        self.nv = self.n_AB
        self.n_qubits = self.n_AB + self.n_ref + self.anc

        total = (M+1)*self.nv
        self.theta = nn.Parameter(torch.randn(total, dtype=torch.float32, device=self.device)*0.08)

        # Shared score head (used ONLY by score-based losses)
        self.beta = nn.Parameter(torch.tensor(1.5, dtype=torch.float32, device=self.device))
        self.logit_bias = nn.Parameter(torch.zeros(2**self.l, dtype=torch.float32, device=self.device))

        # Buffers for per-group gradient EMA (for concave-over-modular)
        self.register_buffer("grad_ema", torch.zeros_like(self.theta))
        self.register_buffer("grad2_ema", torch.zeros_like(self.theta))

        print(f"QAEModel: n_qubits={self.n_qubits}, nv={self.nv}, params={(M+1)*self.nv}, ent={self.ent_pattern}, M={self.M}")

    def amplitude_encode_AB(self, x):
        B = x.shape[0]
        dim = 2**self.n_qubits
        out = torch.zeros(B, dim, dtype=torch.float32, device=self.device)
        shift = self.n_ref + self.anc
        inds = (torch.arange(0, 2**self.n_AB, device=self.device).unsqueeze(0) << shift)
        out.scatter_(1, inds.repeat(B,1), x.to(self.device))
        return out

    def _forward_state_until(self, x, upto_layer_inclusive: int):
        state = self.amplitude_encode_AB(x)
        pidx = 0
        for m in range(self.M):
            thetas = self.theta[pidx:pidx+self.nv]
            state = ry_layer(state, self.n_qubits, thetas)
            pidx += self.nv
            if m == upto_layer_inclusive:
                return state
            forward_dir = True if self.ent_pattern=="cw_fixed" else (m % 2 == 0)
            state = ring_cnot_dir(state, self.n_qubits, self.nv, forward=forward_dir)
        thetas = self.theta[pidx:pidx+self.nv]
        state = ry_layer(state, self.n_qubits, thetas)
        return state

    def probs_after_layer(self, x, g: int):
        state = self._forward_state_until(x, g)
        return self._marginalize_B(state)

    def _marginalize_B(self, state):
        B = state.shape[0]
        t = state.view(B, *([2]*self.n_qubits))
        A_axes  = list(range(1, 1+self.k))
        B_axes  = list(range(1+self.k, 1+self.k+self.l))
        Bp_axes = list(range(1+self.k+self.l, 1+self.k+self.l+self.n_ref))
        anc_axes= list(range(1+self.k+self.l+self.n_ref, 1+self.n_qubits)) if self.anc==1 else []
        tperm = t.permute(*([0] + A_axes + B_axes + Bp_axes + anc_axes)).contiguous()
        A_dim  = 2**self.k
        B_dim  = 2**self.l
        Bp_dim = 2**self.n_ref
        anc_dim = 2 if self.anc==1 else 1
        tperm = tperm.view(B, A_dim, B_dim, (Bp_dim if Bp_dim>0 else 1), anc_dim)
        P_B = (tperm**2).sum(dim=(1,3,4))  # [B, 2**l]
        return P_B

    def forward(self, x):
        state = self._forward_state_until(x, self.M)
        return self._marginalize_B(state)

    def grouped_param_views(self) -> List[torch.Tensor]:
        groups = []
        pidx = 0
        for _ in range(self.M):
            groups.append(self.theta[pidx:pidx+self.nv])
            pidx += self.nv
        groups.append(self.theta[pidx:pidx+self.nv])
        return groups

# ------------------------ Shot noise & STE ------------------------
@torch.no_grad()
def apply_shot_noise_multinomial(P, shots: int, eps: float = EPS):
    """Full-vector multinomial: rows sum exactly to 1."""
    Pn = P.clamp_min(0.0)
    Z = Pn.sum(dim=1, keepdim=True).clamp_min(eps)
    Pn = Pn / Z
    counts = torch.distributions.Multinomial(total_count=shots, probs=Pn).sample()
    return counts / shots

@torch.no_grad()
def apply_shot_noise_binomial_fullvec(P, shots: int):
    """
    Independent binomial per class (your 'old' version).
      p1_j = (1 - P_j)/2
      k_j ~ Binomial(shots, p1_j)
      P_hat_j = 1 - 2*(k_j/shots)
    Note: rows need NOT sum to 1.
    """
    P = torch.clamp(P, 0.0, 1.0)
    p1 = torch.clamp((1.0 - P) * 0.5, 0.0, 1.0)
    k = torch.distributions.Binomial(total_count=shots, probs=p1).sample()
    p1_hat = k / shots
    P_hat = torch.clamp(1.0 - 2.0 * p1_hat, 0.0, 1.0)
    return P_hat

def ste(noisy, clean):
    return clean + (noisy - clean).detach()

def renorm(P):
    Z = P.sum(dim=1, keepdim=True).clamp_min(EPS)
    return (P / Z).clamp_min(EPS)

# Choose shot model per loss
def noisy_probs_for_loss(loss_name: str, P_clean: torch.Tensor, shots: int) -> torch.Tensor:
    if (loss_name == "overlap") or (loss_name == "hellinger") or loss_name.startswith("tsallis_"):
        return apply_shot_noise_binomial_fullvec(P_clean, shots)
    else:
        return apply_shot_noise_multinomial(P_clean, shots)

# ------------------------ Sinkhorn OT ------------------------
def sinkhorn_loss(P, Q, C, eps=OT_EPS, iters=OT_ITERS):
    Kmat = torch.exp(-C / eps)  # [K,K]
    B, Kc = P.shape
    u = torch.ones(B, Kc, device=P.device) / Kc
    v = torch.ones(B, Kc, device=P.device) / Kc
    for _ in range(iters):
        u = P / (v @ Kmat.T + EPS)
        v = Q / (u @ Kmat + EPS)
    Tplan = u.unsqueeze(2) * Kmat.unsqueeze(0) * v.unsqueeze(1)  # [B,K,K]
    cost = (Tplan * C.unsqueeze(0)).sum(dim=(1,2))
    return cost.mean()

# ------------------------ Losses ------------------------
class ScoreHead(nn.Module):
    def __init__(self, model: QAEModel):
        super().__init__()
        self.model = model
    def logits(self, P_eff):
        return self.model.beta * (2.0 * P_eff - 1.0) + self.model.logit_bias

def head_probs(P_eff, head: 'ScoreHead'):
    return F.softmax(head.logits(P_eff), dim=1)

# Probability-based losses (NO head)
def loss_overlap(P_eff, y):
    Py = P_eff.gather(1, y[:,None]).squeeze(1).clamp_min(EPS)
    return (-Py.log()).mean()

def loss_dirichlet_multinomial(P_eff, y, shots=LABEL_SHOTS, alpha=0.5):
    """
    Dirichlet-multinomial inspired loss with gradients.
    We draw label-shot counts from P_eff (no grad), build a posterior-smoothed
    target (counts + alpha) / (shots + Kalpha), then compute CE vs. P_eff.
    """
    B, K = P_eff.shape
    Pn = renorm(P_eff)  # ensure proper distribution
    with torch.no_grad():
        counts = torch.distributions.Multinomial(total_count=shots, probs=Pn).sample()  # [B,K]
    target = (counts + alpha) / (shots + K * alpha)  # [B,K]
    return -(target * (Pn + EPS).log()).sum(dim=1).mean()

def loss_brier(P_eff, y):
    target = torch.zeros_like(P_eff).scatter_(1, y[:,None], 1.0)
    return ((P_eff - target).pow(2)).sum(1).mean()

def loss_hellinger(P_eff, y):
    Py = P_eff.gather(1, y[:,None]).squeeze(1).clamp_min(EPS)
    return (1.0 - torch.sqrt(Py)).mean()

def loss_tsallis(P_eff, y, alpha: float):
    Py = P_eff.gather(1, y[:,None]).squeeze(1).clamp_min(EPS)
    return ((1 - Py.pow(alpha)) / (alpha - 1)).mean()

def loss_ot_sinkhorn(P_eff, y, C):
    target = F.one_hot(y, num_classes=P_eff.size(1)).float()
    return sinkhorn_loss(P_eff, target, C)

# Score-based losses (USE head)
def loss_softmax(P_eff, y, head: ScoreHead):
    logits = head.logits(P_eff)
    return F.cross_entropy(logits, y)

def loss_hinge(P_eff, y, head: ScoreHead, margin=HINGE_MARGIN):
    logits = head.logits(P_eff)
    true = logits.gather(1, y[:,None]).squeeze(1)
    imp = logits.clone(); imp.scatter_(1, y[:,None], -1e9)
    max_imp = imp.max(1).values
    return torch.relu(margin - true + max_imp).mean()

def loss_softmargin(P_eff, y, head: ScoreHead, tau=TAU_SOFTMARGIN):
    logits = head.logits(P_eff) / tau
    true = logits.gather(1, y[:,None]).squeeze(1)
    lse = torch.logsumexp(logits - true[:,None], dim=1)
    return F.softplus(lse).mean()

def loss_infonce(P_eff, y, head: ScoreHead, tau=TAU_INFONCE):
    logits = head.logits(P_eff) / tau
    return F.cross_entropy(logits, y)

# Build registry with "uses_head" flag
def build_losses(C_cost):
    reg = {
        # prob-based
        "overlap":      {"uses_head": False, "fn": lambda P,y,head=None: loss_overlap(P,y)},
        "lbb":          {"uses_head": False, "fn": lambda P,y,head=None: loss_dirichlet_multinomial(P,y)},
        "brier":        {"uses_head": False, "fn": lambda P,y,head=None: loss_brier(P,y)},
        "hellinger":    {"uses_head": False, "fn": lambda P,y,head=None: loss_hellinger(P,y)},
        "ot":           {"uses_head": False, "fn": lambda P,y,head=None: loss_ot_sinkhorn(P,y,C_cost)},
        # score-based
        "softmax":      {"uses_head": True,  "fn": lambda P,y,head: loss_softmax(P,y,head)},
        "hinge":        {"uses_head": True,  "fn": lambda P,y,head: loss_hinge(P,y,head)},
        "softmargin":   {"uses_head": True,  "fn": lambda P,y,head: loss_softmargin(P,y,head)},
        "infonce":      {"uses_head": True,  "fn": lambda P,y,head: loss_infonce(P,y,head)},
    }
    for a in TSALLIS_ALPHAS:
        reg[f"tsallis_{a}"] = {"uses_head": False, "fn": (lambda alpha: (lambda P,y,head=None: loss_tsallis(P,y,alpha)))(a)}
    return reg

SCORE_LOSSES = {"softmax", "hinge", "softmargin", "infonce"}
PROB_LOSSES  = {"overlap", "lbb", "brier", "hellinger", "ot"} | {f"tsallis_{a}" for a in TSALLIS_ALPHAS}
BINOMIAL_VECTOR_LOSSES = {"overlap", "hellinger"} | {f"tsallis_{a}" for a in TSALLIS_ALPHAS}

# ------------------------ Penalties ------------------------
@dataclass
class PenaltyState:
    cov_ema: torch.Tensor
    fac_ema: torch.Tensor
    grad_ema_groups: torch.Tensor
    grad2_ema_groups: torch.Tensor

def make_penalty_state(model: QAEModel) -> PenaltyState:
    G = model.M + 1
    return PenaltyState(
        cov_ema=torch.zeros(2**L, device=DEVICE),
        fac_ema=torch.zeros(1, device=DEVICE),
        grad_ema_groups=torch.zeros(G, device=DEVICE),
        grad2_ema_groups=torch.zeros(G, device=DEVICE),
    )

def entropy_penalty(P_any):
    Pn = renorm(P_any)
    return -(Pn * (Pn+EPS).log()).sum(1).mean()

def com_penalty(model: QAEModel, state: PenaltyState, variant: str, beta_mix=COM_BETA):
    groups = model.grouped_param_views()
    mags = torch.stack([g.norm() for g in groups])  # [G]
    if variant == "com_maggrad":
        a = (1 - beta_mix) * mags + beta_mix * state.grad_ema_groups
    elif variant == "com_fisher":
        a = state.grad2_ema_groups.sqrt() + 1e-8
    else:
        raise ValueError("Unknown COM variant")
    return torch.log1p(a.sum())

def facility_penalty(model: QAEModel, state: PenaltyState, P_layers_batchmean: List[torch.Tensor], prob_scale: float):
    reps = P_layers_batchmean
    S = []
    for i in range(len(reps)):
        for j in range(i+1, len(reps)):
            S.append(F.cosine_similarity(reps[i], reps[j], dim=0))
    cur = -torch.stack(S).mean() if S else torch.tensor(0.0, device=DEVICE)
    # EMA update must NOT capture graph
    with torch.no_grad():
        state.fac_ema = (1-EMA_DECAY)*state.fac_ema + EMA_DECAY*(prob_scale * cur.detach())
    # Return smoothed scalar (no grad through time)
    return state.fac_ema.squeeze(0)

def coverage_penalty(model: QAEModel, state: PenaltyState, deltas_batchmean: List[torch.Tensor], prob_scale: float):
    if len(deltas_batchmean)==0:
        return torch.tensor(0.0, device=DEVICE)
    delta_sum = torch.stack(deltas_batchmean, dim=0).mean(0)  # [K]
    # EMA update must NOT capture graph
    with torch.no_grad():
        state.cov_ema = (1-EMA_DECAY)*state.cov_ema + EMA_DECAY*(prob_scale * delta_sum.detach())
    # Use a stabilized, positive, concave map
    return torch.log1p(state.cov_ema.clamp_min(0)).sum()

# ------------------------ Eval ------------------------
def evaluate(model: QAEModel, head: ScoreHead, loader: DataLoader, use_head_pred: bool, loss_name_for_shots: str):
    model.eval()
    corr_clean = corr_noisy = tot = 0
    entropies = []; briers = []; margins = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            P_clean = renorm(model(xb))
            P_noisy = noisy_probs_for_loss(loss_name_for_shots, P_clean, LABEL_SHOTS)
            P_noisy_normed = renorm(P_noisy)

            if use_head_pred:
                logits_clean = head.logits(P_clean)
                logits_noisy = head.logits(P_noisy)
                pred_clean = logits_clean.argmax(1)
                pred_noisy = logits_noisy.argmax(1)
                true = logits_noisy.gather(1, yb[:,None]).squeeze(1)
                imp  = logits_noisy.clone(); imp.scatter_(1, yb[:,None], -1e9)
                max_imp = imp.max(1).values
            else:
                pred_clean = P_clean.argmax(1)
                pred_noisy = P_noisy.argmax(1)
                true = P_noisy.gather(1, yb[:,None]).squeeze(1)
                imp  = P_noisy.clone(); imp.scatter_(1, yb[:,None], -1e9)
                max_imp = imp.max(1).values

            corr_clean += (pred_clean == yb).sum().item()
            corr_noisy += (pred_noisy == yb).sum().item()
            tot += yb.size(0)

            briers.append(((P_noisy_normed - F.one_hot(yb, P_noisy_normed.size(1)).float())**2).sum(1))
            entropies.append(-(P_noisy_normed * (P_noisy_normed+EPS).log()).sum(1))
            margins.append(true - max_imp)

    clean = 100.0 * corr_clean / max(1, tot)
    noisy = 100.0 * corr_noisy / max(1, tot)
    final_entropy = torch.cat(entropies).mean().item()
    final_brier   = torch.cat(briers).mean().item()
    final_margin  = torch.cat(margins).mean().item()
    return clean, noisy, final_margin, final_entropy, final_brier

# ------------------------ Train One Run ------------------------
@dataclass
class RunConfig:
    loss_name: str
    penalty: str           # "none","entropy","com_maggrad","com_fisher","facility","coverage"
    lam: float             # ignored for "none"
    epochs: int
    seed: int
    save_tag: str
    entropy_mode: Optional[str] = None  # None/"born"/"head"

@dataclass
class RunResult:
    loss_name: str; penalty: str; lam: float; seed: int; epochs: int
    best_noisy_acc: float; best_clean_acc: float; epoch_best: int
    final_margin: float; final_entropy: float; final_brier: float
    notes: Dict[str, Any]

def run_once(init_state: Dict[str, torch.Tensor], config: RunConfig,
             train_loader: DataLoader, test_loader: DataLoader,
             C_cost):
    # Model & init
    model = QAEModel().to(DEVICE)
    model.load_state_dict(init_state, strict=True)
    head = ScoreHead(model)

    # Opt & sched
    opt = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=T0, T_mult=T_MULT, eta_min=ETA_MIN)

    # Penalty state
    pen_state = make_penalty_state(model)

    # Loss
    LOSS_REG = build_losses(C_cost)
    loss_entry = LOSS_REG[config.loss_name]
    uses_head = loss_entry["uses_head"]
    loss_fn = loss_entry["fn"]

    # Layer sampling
    G_layers = M + 1
    p_g = PROBED_LAYERS_PER_MINIBATCH / G_layers
    inv_p = 1.0 / p_g

    best_noisy = 0.0; best_clean = 0.0; epoch_best = 0

    for epoch in range(1, config.epochs+1):
        model.train()
        t0 = time.time()
        running = []

        for xb, yb in train_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)

            # Clean final probs
            P_clean = renorm(model(xb))

            # Noisy final probs + STE routed by loss type (for primary loss)
            with torch.no_grad():
                P_noisy_primary = noisy_probs_for_loss(config.loss_name, P_clean, LABEL_SHOTS)
            P_eff = ste(P_noisy_primary, P_clean)

            # Primary loss
            if uses_head:
                primary = loss_fn(P_eff, yb, head)
            else:
                primary = loss_fn(P_eff, yb)

            # Penalty terms
            penalty_val = torch.tensor(0.0, device=DEVICE)
            if config.penalty != "none":
                if config.penalty == "entropy":
                    # Always compute entropy on a multinomial-shot version of the distribution
                    mode = (config.entropy_mode or "born")
                    P_ent_clean = renorm(P_clean)
                    with torch.no_grad():
                        P_ent_noisy = apply_shot_noise_multinomial(P_ent_clean, shots=LABEL_SHOTS)
                    P_ent = ste(P_ent_noisy, P_ent_clean) if PENALTY_STE else P_ent_clean

                    if mode == "head" and uses_head:
                        Pb = head_probs(P_ent, head)
                        penalty_val = entropy_penalty(Pb)
                    else:
                        penalty_val = entropy_penalty(P_ent)

                    total_loss = primary - config.lam * penalty_val

                elif config.penalty in ("com_maggrad", "com_fisher"):
                    penalty_val = com_penalty(model, pen_state, variant=config.penalty)
                    total_loss = primary + config.lam * penalty_val

                elif config.penalty in ("facility","coverage"):
                    sampled = random.sample(range(0, G_layers), k=PROBED_LAYERS_PER_MINIBATCH)
                    reps = []
                    deltas = []
                    for g in sampled:
                        Pg_clean = renorm(model.probs_after_layer(xb, g))
                        if PENALTY_STE:
                            with torch.no_grad():
                                Pg_noisy = apply_shot_noise_multinomial(Pg_clean, shots=LAYER_PROBE_SHOTS)
                            Pg_eff = ste(Pg_noisy, Pg_clean)
                        else:
                            Pg_eff = Pg_clean
                        reps.append(Pg_eff.mean(0))
                        if config.penalty == "coverage":
                            if g == 0:
                                Pprev_eff = torch.full_like(Pg_eff, 1.0 / Pg_eff.size(1))
                            else:
                                Pprev_clean = renorm(model.probs_after_layer(xb, g-1))
                                if PENALTY_STE:
                                    with torch.no_grad():
                                        Pprev_noisy = apply_shot_noise_multinomial(Pprev_clean, shots=LAYER_PROBE_SHOTS)
                                    Pprev_eff = ste(Pprev_noisy, Pprev_clean)
                                else:
                                    Pprev_eff = Pprev_clean
                            deltas.append((Pg_eff - Pprev_eff).clamp_min(0.0).mean(0))
                    if config.penalty == "facility":
                        pen_raw = facility_penalty(model, pen_state, reps, prob_scale=inv_p)
                        total_loss = primary + config.lam * pen_raw
                        penalty_val = pen_raw
                    else:
                        pen_raw = coverage_penalty(model, pen_state, deltas, prob_scale=inv_p)
                        total_loss = primary - config.lam * pen_raw
                        penalty_val = pen_raw
                else:
                    raise ValueError("Unknown penalty")
            else:
                total_loss = primary

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            # Update group-wise EMA(|grad|) and EMA(grad^2) for COM (no graph)
            with torch.no_grad():
                gparam = model.theta.grad
                if gparam is not None:
                    groups = model.grouped_param_views()
                    g_abs = []; g_sq = []
                    pidx = 0
                    for g in groups:
                        n = g.numel()
                        gi = gparam[pidx:pidx+n]
                        g_abs.append(gi.abs().mean())
                        g_sq.append((gi**2).mean())
                        pidx += n
                    pen_state.grad_ema_groups = (1-EMA_DECAY)*pen_state.grad_ema_groups + EMA_DECAY*torch.stack(g_abs)
                    pen_state.grad2_ema_groups = (1-EMA_DECAY)*pen_state.grad2_ema_groups + EMA_DECAY*torch.stack(g_sq)

            running.append(float(primary.item()))

        sch.step(epoch+1)

        use_head_pred = config.loss_name in SCORE_LOSSES
        clean, noisy, margin, entr, brier = evaluate(model, head, test_loader, use_head_pred, config.loss_name)
        if noisy > best_noisy or (abs(noisy-best_noisy)<1e-6 and clean>best_clean):
            best_noisy = noisy; best_clean = clean; epoch_best = epoch

        print(f"[{config.save_tag}] Ep {epoch:02d}/{config.epochs} "
              f"| loss={np.mean(running):.4f} | clean={clean:.2f} | noisy={noisy:.2f} "
              f"| margin={margin:.3f} | H={entr:.3f} | Brier={brier:.4f} "
              f"| beta={model.beta.item():.3f} | time={time.time()-t0:.1f}s")

        gc.collect()
        if DEVICE.type=="cuda":
            torch.cuda.empty_cache()

    return RunResult(
        loss_name=config.loss_name, penalty=config.penalty, lam=(0.0 if config.penalty=="none" else config.lam),
        seed=config.seed, epochs=config.epochs, best_noisy_acc=best_noisy, best_clean_acc=best_clean,
        epoch_best=epoch_best, final_margin=margin, final_entropy=entr, final_brier=brier,
        notes={
            "beta": float(model.beta.item()),
            "logit_bias_norm": float(model.logit_bias.norm().item()),
            "entropy_mode": config.entropy_mode or ""
        }
    )

# ------------------------ Driver ------------------------
def main():
    print("Device:", DEVICE)
    print("Seeding:", SEED)

    train_loader, test_loader = make_kmnist_loaders()
    C_cost = hamming3_cost_matrix(Kclasses=2**L, device=DEVICE)

    LOSS_REG = build_losses(C_cost)

    # Select losses based on TYPES_OF_LOSSES
    multinom_only = ["brier", "ot", "softmax", "hinge", "softmargin", "infonce"]
    binom_only    = ["overlap", "hellinger"] + [f"tsallis_{a}" for a in TSALLIS_ALPHAS]
    if TYPES_OF_LOSSES == 1:
        LOSSES = multinom_only
    elif TYPES_OF_LOSSES == 2:
        LOSSES = binom_only
    else:
        raise ValueError("TYPES_OF_LOSSES must be 1 or 2")

    PENALTIES = ["none", "entropy", "com_maggrad", "com_fisher", "facility", "coverage"]

    # Same init per loss bucket
    init_states: Dict[str, Dict[str, torch.Tensor]] = {}
    for loss_name in LOSSES:
        base_model = QAEModel().to(DEVICE)
        init_states[loss_name] = {k: v.clone().detach() for k, v in base_model.state_dict().items()}

    # -------------- PILOT RUNS --------------
    pilot_log = OUTDIR / "pilot_runs.jsonl"
    if pilot_log.exists(): pilot_log.unlink()
    finalists: Dict[str, List[Dict[str, Any]]] = {ln: [] for ln in LOSSES}

    for loss_name in LOSSES:
        uses_head = LOSS_REG[loss_name]["uses_head"]
        configs: List[RunConfig] = []

        # always include "none"
        configs.append(RunConfig(loss_name, "none", 0.0, EPOCHS_PILOT, SEED,
                                 f"{loss_name}|none"))

        for pen in PENALTIES:
            if pen == "none":
                continue
            if pen == "entropy":
                # allow both born/head if the loss uses head
                entropy_modes = ["born", "head"] if uses_head else ["born"]
                for emode in entropy_modes:
                    for lam in LAMBDA_GRID["entropy"]:
                        tag = f"{loss_name}|entropy[{emode}]|lam={lam}"
                        configs.append(RunConfig(loss_name, "entropy", lam, EPOCHS_PILOT, SEED, tag, entropy_mode=emode))
            else:
                for lam in LAMBDA_GRID[pen]:
                    tag = f"{loss_name}|{pen}|lam={lam}"
                    configs.append(RunConfig(loss_name, pen, lam, EPOCHS_PILOT, SEED, tag))

        results_this_loss: List[RunResult] = []
        for cfg in configs:
            res = run_once(init_states[loss_name], cfg, train_loader, test_loader, C_cost)
            results_this_loss.append(res)
            with open(pilot_log, "a") as f:
                f.write(json.dumps(asdict(res))+"\n")

        # Rank: noisy desc, then margin desc, then clean desc
        results_this_loss.sort(key=lambda r: (r.best_clean_acc, r.final_margin, r.best_noisy_acc), reverse=True)
        finalists[loss_name] = [asdict(results_this_loss[0]), asdict(results_this_loss[1])]

    with open(OUTDIR/"selection.json", "w") as f:
        json.dump(finalists, f, indent=2)

    # -------------- FINAL RUNS --------------
    final_log = OUTDIR / "final_runs.jsonl"
    if final_log.exists():
        final_log.unlink()

    for loss_name, picked in finalists.items():
        for pick in picked:
            pen = pick["penalty"]
            lam = pick["lam"]
            entropy_mode = (pick.get("notes", {}) or {}).get("entropy_mode") or pick.get("entropy_mode") or None

            mode_suffix = ""
            if pen == "entropy":
                mode_suffix = f"[{entropy_mode or 'born'}]"

            tag = f"FINAL|{loss_name}|{pen}{mode_suffix}|lam={lam}"

            cfg = RunConfig(
                loss_name=loss_name,
                penalty=pen,
                lam=lam,
                epochs=EPOCHS_FINAL,
                seed=SEED,
                save_tag=tag,
                entropy_mode=(entropy_mode if pen == "entropy" else None),
            )

            res = run_once(init_states[loss_name], cfg, train_loader, test_loader, C_cost)
            with open(final_log, "a") as f:
                f.write(json.dumps(asdict(res)) + "\n")

    print("Done. See:", OUTDIR)

if __name__ == "__main__":
    main()
