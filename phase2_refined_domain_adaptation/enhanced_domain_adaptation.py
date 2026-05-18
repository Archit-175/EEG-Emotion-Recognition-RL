"""
=============================================================================
FLD3QN: Frontal Lobe Double Dueling Deep Q Network
Subject-INDEPENDENT (LOSO) — ENHANCED VERSION
=============================================================================
Improvements over baseline LOSO:
  [1] Euclidean Alignment (EA)       – removes inter-subject covariance shift
  [2] Global z-score normalization   – train-set statistics applied to test set
  [3] Differential Entropy (DE)      – stable frequency-band features (5 bands)
  [4] Subject-balanced batch sampling– equal subject contribution per episode
  [5] Multi-task learning            – shared backbone, dual heads (val+aro)
  [6] CORAL loss                     – aligns source/target covariances
  [7] Domain-Adversarial training    – gradient reversal for subject invariance
  [8] Channel Attention (SE block)   – learn EEG-relevant electrode weights
  [9] Increased training steps       – 50 000 steps (was 5 000)
  [10] Soft labels                   – continuous ratings, not binary threshold

Dataset  : DEAP (preprocessed .dat files on Kaggle)
Framework: PyTorch
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 – Imports & reproducibility seed
# ─────────────────────────────────────────────────────────────────────────────
import os
import glob
import pickle
import random
import copy
import collections
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.signal
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 – EEG Channel Configuration
# ─────────────────────────────────────────────────────────────────────────────
ALL_CHANNELS  = list(range(32))
LEFT_CH       = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
RIGHT_CH      = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
LEFT_FRONTAL  = [0, 1, 2, 3, 4, 5]
RIGHT_FRONTAL = [26, 27, 28, 29, 30, 31]

# Frequency bands for Differential Entropy
FREQ_BANDS = {
    "delta": (1,  4),
    "theta": (4,  8),
    "alpha": (8,  14),
    "beta":  (14, 31),
    "gamma": (31, 51),
}
N_BANDS = len(FREQ_BANDS)   # 5

print(f"Left hemisphere : {len(LEFT_CH)} ch | Right hemisphere : {len(RIGHT_CH)} ch")
print(f"Left frontal    : {len(LEFT_FRONTAL)} ch | Right frontal    : {len(RIGHT_FRONTAL)} ch")
print(f"Frequency bands : {list(FREQ_BANDS.keys())}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 – Signal Processing Utilities
# ─────────────────────────────────────────────────────────────────────────────

# ── [1] Euclidean Alignment ──────────────────────────────────────────────────
def euclidean_alignment(X: np.ndarray) -> np.ndarray:
    """
    Align EEG windows to reduce inter-subject covariance shift.
    X : (N, C, T)  → returns aligned (N, C, T)
    """
    N, C, T = X.shape
    # Mean covariance
    R = np.zeros((C, C), dtype=np.float64)
    for i in range(N):
        xi = X[i].astype(np.float64)
        R += xi @ xi.T
    R /= N

    # Regularise for numerical stability
    R += np.eye(C) * 1e-8

    # R^{-1/2} via eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.maximum(eigvals, 1e-10)
    R_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    aligned = np.array([R_inv_sqrt @ X[i].astype(np.float64) for i in range(N)],
                       dtype=np.float32)
    return aligned


# ── [3] Differential Entropy ─────────────────────────────────────────────────
def compute_de(window: np.ndarray, fs: int = 128) -> np.ndarray:
    """
    Compute Differential Entropy for one EEG window.
    window : (C, T)
    Returns DE : (C, N_BANDS) = (32, 5)
    """
    C, T = window.shape
    de = np.zeros((C, N_BANDS), dtype=np.float32)

    for b_idx, (band_name, (f_lo, f_hi)) in enumerate(FREQ_BANDS.items()):
        # Design Butterworth bandpass filter
        nyq = fs / 2.0
        lo, hi = f_lo / nyq, min(f_hi / nyq, 0.99)
        try:
            b, a = scipy.signal.butter(4, [lo, hi], btype="band")
            filtered = scipy.signal.filtfilt(b, a, window, axis=1)  # (C, T)
        except Exception:
            filtered = window   # fallback: no filter

        # Band variance per channel → DE
        var = np.maximum(filtered.var(axis=1), 1e-6)
        de[:, b_idx] = 0.5 * np.log(2 * np.pi * np.e * var)

    return de   # (C, 5)


def compute_de_features(windows: np.ndarray, fs: int = 128) -> np.ndarray:
    """
    Compute DE features for all windows.
    windows : (N, 32, 128)
    Returns  : (N, 32, N_BANDS) = (N, 32, 5)
    """
    print(f"  Computing DE features for {len(windows)} windows …", end=" ")
    de_all = np.array([compute_de(w, fs) for w in windows], dtype=np.float32)
    print("done.")
    return de_all   # (N, 32, 5)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 – Data Loading & Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

DEAP_PATH  = "/kaggle/input/datasets/guneethpalakurthy/deap-dataset"
FS         = 128
BASELINE_S = 3
WIN_SIZE   = FS
LAMBDA_REW = 0.1


def load_subject_raw(subj_id: int) -> tuple:
    """
    Load raw DEAP subject. Returns un-normalised windows and continuous labels.
    """
    pattern = os.path.join(DEAP_PATH, f"s{subj_id:02d}*.dat")
    files   = glob.glob(pattern)
    if not files:
        files = glob.glob(os.path.join(DEAP_PATH, "data_preprocessed_python",
                                       f"s{subj_id:02d}*.dat"))
    if not files:
        raise FileNotFoundError(f"Cannot find DEAP file for subject {subj_id}")

    with open(files[0], "rb") as f:
        data = pickle.load(f, encoding="latin1")

    eeg    = data["data"][:, :32, :].astype(np.float32)  # (40, 32, 8064)
    labels = data["labels"]                               # (40, 4)

    # Remove 3-sec baseline
    eeg = eeg[:, :, BASELINE_S * FS:]   # (40, 32, 7680)

    # Per-trial mean removal only (no std yet — global z-score applied later)
    eeg = eeg - eeg.mean(axis=2, keepdims=True)

    # Segment into 1-sec windows
    n_trials, n_ch, n_samples = eeg.shape
    windows_list = []
    val_cont_list, aro_cont_list = [], []

    for t in range(n_trials):
        n_wins = n_samples // WIN_SIZE
        for w in range(n_wins):
            seg = eeg[t, :, w * WIN_SIZE:(w + 1) * WIN_SIZE]
            windows_list.append(seg)
            val_cont_list.append(float(labels[t, 0]))   # continuous 1–9
            aro_cont_list.append(float(labels[t, 1]))

    windows = np.array(windows_list, dtype=np.float32)
    # Binary labels (threshold = 5)
    label_dict = {
        "valence": np.array([int(v > 5) for v in val_cont_list], dtype=np.int64),
        "arousal": np.array([int(a > 5) for a in aro_cont_list], dtype=np.int64),
        "valence_soft": np.array(val_cont_list, dtype=np.float32),
        "arousal_soft": np.array(aro_cont_list, dtype=np.float32),
    }

    return windows, label_dict


def load_all_subjects(subject_ids: list) -> tuple:
    """
    Load all subjects, returning raw windows (before EA / global-norm).
    """
    windows_list, subj_id_list = [], []
    label_lists = collections.defaultdict(list)

    available = []
    for sid in subject_ids:
        try:
            w, l = load_subject_raw(sid)
            windows_list.append(w)
            subj_id_list.append(np.full(len(w), sid, dtype=np.int32))
            for k, v in l.items():
                label_lists[k].append(v)
            available.append(sid)
            print(f"  Subject {sid:02d}: {len(w)} windows loaded")
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")

    all_windows       = np.concatenate(windows_list, axis=0)
    subject_ids_array = np.concatenate(subj_id_list)
    all_labels        = {k: np.concatenate(v) for k, v in label_lists.items()}

    print(f"\nTotal: {all_windows.shape[0]} windows from {len(available)} subjects")
    return all_windows, all_labels, subject_ids_array, available


# ── [2] Global z-score normalisation (train-set statistics only) ─────────────
def global_normalize(X_train: np.ndarray,
                     X_test:  np.ndarray) -> tuple:
    """
    Compute mean/std from training windows only, apply to both splits.
    """
    mu  = X_train.mean(axis=(0, 2), keepdims=True)   # (1, C, 1)
    std = X_train.std(axis=(0, 2),  keepdims=True)
    std[std < 1e-8] = 1e-8

    X_train_n = (X_train - mu) / std
    X_test_n  = (X_test  - mu) / std
    return X_train_n.astype(np.float32), X_test_n.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 – BiFRCNN with Channel Attention & DE Input
# ─────────────────────────────────────────────────────────────────────────────

# [8] Squeeze-and-Excitation channel attention block
class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention for 2-D feature maps."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(in_ch, out_ch, (1, kernel), padding=(0, pad))
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, (1, kernel), padding=(0, pad))
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.skip1 = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.skip2 = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu  = nn.ReLU()
        self.se    = SEBlock(out_ch)   # [8] SE attention per residual block

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = out + self.skip1(identity)
        out = self.bn2(self.conv2(out))
        out = self.se(out)             # apply channel attention
        out = out + self.skip2(identity)
        return self.relu(out)


class StreamEncoder(nn.Module):
    """
    Encodes one EEG sub-region.
    Input can be raw (n_ch, T) or DE features (n_ch, N_BANDS).
    """
    def __init__(self, n_ch: int, feature_dim: int = 64):
        super().__init__()
        self.conv_in = nn.Conv2d(1, 16, (1, 3), padding=(0, 1))
        self.bn_in   = nn.BatchNorm2d(16)
        self.relu    = nn.ReLU()
        self.res1    = ResidualBlock(16, 32)
        self.res2    = ResidualBlock(32, 64)
        self.pool    = nn.AdaptiveAvgPool2d((1, 8))
        self.flatten = nn.Flatten()
        self.fc      = nn.Linear(64 * 8, feature_dim)

    def forward(self, x):
        x = x.unsqueeze(1)                          # (B, 1, n_ch, T_or_bands)
        x = self.relu(self.bn_in(self.conv_in(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.relu(self.fc(x))


# ── [7] Gradient Reversal Layer for DANN ────────────────────────────────────
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


class GradientReversal(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


# ── [5] Multi-task BiFRCNN: shared backbone, dual emotion heads ──────────────
class BiFRCNN_MultiTask(nn.Module):
    """
    Extended BiFRCNN with:
      - SE attention in residual blocks                         [8]
      - Dual output heads: valence Q + arousal Q               [5]
      - Subject classifier head (DANN)                         [7]
    """
    def __init__(self, feature_dim: int = 64, n_actions: int = 2,
                 n_subjects: int = 32):
        super().__init__()
        self.enc_left_hemi   = StreamEncoder(len(LEFT_CH),      feature_dim)
        self.enc_right_hemi  = StreamEncoder(len(RIGHT_CH),     feature_dim)
        self.enc_left_front  = StreamEncoder(len(LEFT_FRONTAL), feature_dim)
        self.enc_right_front = StreamEncoder(len(RIGHT_FRONTAL),feature_dim)

        fused_dim    = feature_dim * 2
        combined_dim = feature_dim * 2

        self.merge_left  = nn.Sequential(nn.Linear(fused_dim, feature_dim),
                                         nn.ReLU())
        self.merge_right = nn.Sequential(nn.Linear(fused_dim, feature_dim),
                                         nn.ReLU())
        self.shared = nn.Sequential(nn.Linear(combined_dim, 256),
                                    nn.ReLU(),
                                    nn.Dropout(0.3))

        # ── [5] Dual emotion heads (valence + arousal) ───────────────────────
        for task in ("valence", "arousal"):
            setattr(self, f"value_{task}",     nn.Linear(256, 1))
            setattr(self, f"advantage_{task}", nn.Linear(256, n_actions))

        # ── [7] Domain (subject) classifier head ─────────────────────────────
        self.grad_reversal   = GradientReversal(alpha=1.0)
        self.domain_head     = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_subjects),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Shared feature extraction from raw EEG or DE input."""
        xl_hemi  = x[:, LEFT_CH,      :]
        xr_hemi  = x[:, RIGHT_CH,     :]
        xl_front = x[:, LEFT_FRONTAL, :]
        xr_front = x[:, RIGHT_FRONTAL,:]

        left_fused  = self.merge_left(torch.cat([
            self.enc_left_hemi(xl_hemi),
            self.enc_left_front(xl_front)], dim=1))
        right_fused = self.merge_right(torch.cat([
            self.enc_right_hemi(xr_hemi),
            self.enc_right_front(xr_front)], dim=1))

        return self.shared(torch.cat([left_fused, right_fused], dim=1))

    def forward(self, x: torch.Tensor,
                task: str = "valence") -> torch.Tensor:
        """Returns Q-values for the specified task."""
        shared = self._encode(x)
        V = getattr(self, f"value_{task}")(shared)
        A = getattr(self, f"advantage_{task}")(shared)
        return V + A - A.mean(dim=1, keepdim=True)

    def domain_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Returns subject-classifier logits (used for DANN loss)."""
        shared  = self._encode(x)
        reversed_feat = self.grad_reversal(shared)
        return self.domain_head(reversed_feat)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 – CORAL Loss
# ─────────────────────────────────────────────────────────────────────────────

def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    [6] CORAL: Correlation Alignment loss.
    Minimises the squared Frobenius norm between source and target
    second-order statistics (covariance matrices).
    source, target : (B, D)
    """
    d = source.size(1)

    # Covariance matrices
    def cov(X):
        n = X.size(0)
        X_c = X - X.mean(0, keepdim=True)
        return (X_c.T @ X_c) / (n - 1 + 1e-8)

    Cs = cov(source)
    Ct = cov(target)
    loss = ((Cs - Ct) ** 2).sum() / (4.0 * d * d)
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 – Experience Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

Transition = collections.namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done",
                   "subject_id"])

class ReplayBuffer:
    def __init__(self, capacity: int = 200_000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch  = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, subj = zip(*batch)
        subj = np.array(subj)
        subj = np.clip(subj, 0, 31)

        return (
            torch.tensor(np.array(state),      dtype=torch.float32).to(DEVICE),
            torch.tensor(action,               dtype=torch.long).to(DEVICE),
            torch.tensor(reward,               dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(next_state), dtype=torch.float32).to(DEVICE),
            torch.tensor(done,                 dtype=torch.float32).to(DEVICE),
            torch.tensor(subj,                 dtype=torch.long).to(DEVICE),
        )

    def __len__(self):
        return len(self.buffer)


# ── [4] Subject-balanced sampler ─────────────────────────────────────────────
class SubjectBalancedSampler:
    """
    Maintains per-subject index lists.
    sample_batch() draws equal numbers from each training subject,
    eliminating bias toward subjects with more windows.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 subj_ids: np.ndarray):
        self.X       = X
        self.y       = y
        self.subj_ids = subj_ids
        self.subjects = sorted(set(subj_ids.tolist()))
        # Map subject → indices
        self.subj_idx = {s: np.where(subj_ids == s)[0]
                         for s in self.subjects}
        
        # Create a contiguous 0-indexed mapping for the Domain Classifier
        self.s_to_label = {s: i for i, s in enumerate(self.subjects)}

    def sample_one(self):
        """Sample one (state, label, subject_id) uniformly across subjects."""
        s   = random.choice(self.subjects)
        idx = random.choice(self.subj_idx[s])
        
        # Pass the contiguous mapped label
        return self.X[idx], int(self.y[idx]), self.s_to_label[s], idx

    def get_next_state(self, idx: int) -> np.ndarray:
        """Returns adjacent window for RL next-state."""
        next_idx = (idx + 1) % len(self.X)
        return self.X[next_idx]


# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 – Enhanced FLD3QN Agent
# ─────────────────────────────────────────────────────────────────────────────

# Coefficients for auxiliary losses
LAMBDA_DANN  = 0.1    # [7] domain-adversarial loss weight
LAMBDA_CORAL = 0.1    # [6] CORAL loss weight
LAMBDA_REW   = 0.1    # RL reward penalty


class FLD3QN_Agent_Enhanced:
    """
    Double Dueling DQN with BiFRCNN_MultiTask backbone.
    Adds CORAL + DANN auxiliary losses on top of the DQN TD loss.
    Supports multi-task (valence / arousal) via task argument.
    """

    def __init__(self, n_actions: int = 2, n_subjects: int = 32,
                 lr: float = 2.5e-4, gamma: float = 0.75,
                 epsilon: float = 0.15, batch_size: int = 64,
                 buffer_size: int = 200_000,
                 target_update_freq: int = 500):     # FIX: Updated to 500 for 5000-step run

        self.n_actions          = n_actions
        self.gamma              = gamma
        self.epsilon            = epsilon
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.steps_done         = 0
        self.task               = "valence"   # switched externally per fold

        self.q_net      = BiFRCNN_MultiTask(feature_dim=64,
                                            n_actions=n_actions,
                                            n_subjects=n_subjects).to(DEVICE)
        self.target_net = copy.deepcopy(self.q_net).to(DEVICE)
        self.target_net.eval()

        self.optimizer    = optim.Adam(self.q_net.parameters(), lr=lr,
                                       weight_decay=1e-4)
        
        # FIX: Changed T_max to 5000 for the longer run
        self.scheduler    = optim.lr_scheduler.CosineAnnealingLR(
                                self.optimizer, T_max=5000, eta_min=1e-5)
        
        self.loss_fn      = nn.MSELoss()
        self.memory       = ReplayBuffer(buffer_size)
        self.loss_history = []
        self.best_acc     = 0.0

    def set_task(self, task: str):
        assert task in ("valence", "arousal")
        self.task = task

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            return self.q_net(s, task=self.task).argmax(dim=1).item()

    @staticmethod
    def compute_reward(action: int, label: int, t: int,
                       lam: float = LAMBDA_REW) -> tuple:
        if action == label:
            return 1.0, False
        return -1.0 - (lam / max(t, 1)), True

    def learn(self, X_test_batch: torch.Tensor | None = None):
        """
        One learning step.
        X_test_batch: optional tensor (B, C, T) from test domain for
                      CORAL + DANN alignment losses.
        """
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, subj_ids = \
            self.memory.sample(self.batch_size)

        # ── Double DQN TD loss ────────────────────────────────────────────────
        q_values = self.q_net(states, task=self.task)\
                       .gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions  = self.q_net(next_states, task=self.task).argmax(dim=1)
            next_q_target = self.target_net(next_states, task=self.task)\
                                .gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q_target * (1.0 - dones)

        td_loss = self.loss_fn(q_values, target_q)

        # ── [7] DANN domain loss ──────────────────────────────────────────────
        domain_logits = self.q_net.domain_logits(states)
        dann_loss     = F.cross_entropy(domain_logits, subj_ids)

        # ── [6] CORAL loss (source vs target covariance alignment) ────────────
        coral = torch.tensor(0.0, device=DEVICE)
        if X_test_batch is not None:
            # Extract shared features without gradient reversal
            with torch.no_grad():
                target_feat = self.q_net._encode(X_test_batch)
            source_feat = self.q_net._encode(states)
            coral = coral_loss(source_feat, target_feat)

        total_loss = td_loss \
                     + LAMBDA_DANN  * dann_loss \
                     + LAMBDA_CORAL * coral

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        self.scheduler.step()

        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        loss_val = total_loss.item()
        self.loss_history.append(loss_val)
        return loss_val

    def save_checkpoint(self, path: str):
        torch.save({
            "q_net_state":      self.q_net.state_dict(),
            "target_net_state": self.target_net.state_dict(),
            "optimizer_state":  self.optimizer.state_dict(),
            "scheduler_state":  self.scheduler.state_dict(),
            "steps_done":       self.steps_done,
            "best_acc":         self.best_acc,
            "task":             self.task,
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=DEVICE)
        self.q_net.load_state_dict(ckpt["q_net_state"])
        self.target_net.load_state_dict(ckpt["target_net_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.steps_done = ckpt["steps_done"]
        self.best_acc   = ckpt["best_acc"]
        self.task       = ckpt.get("task", "valence")
        print(f"  Resumed from step {self.steps_done} | "
              f"best_acc={self.best_acc:.4f} | task={self.task}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 – Training Configuration
# ─────────────────────────────────────────────────────────────────────────────

CKPT_DIR    = Path("/kaggle/working/checkpoint_independent")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

TOTAL_STEPS = 5000    # FIX: Updated to 5000 outer steps
CKPT_EVERY  = 500     # FIX: Adjusted evaluation frequencies for the longer run
EVAL_EVERY  = 500
CORAL_BATCH = 64      # number of test windows sampled per learn() call


def find_latest_checkpoint(test_sid: int, label: str) -> str | None:
    pattern = str(CKPT_DIR / f"loso_test{test_sid:02d}_{label}_step_*.pth")
    files   = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda f: int(f.split("_step_")[1].replace(".pth", "")))
    return files[-1]


# ─────────────────────────────────────────────────────────────────────────────
# CELL 10 – Training Loop (Enhanced LOSO)
# ─────────────────────────────────────────────────────────────────────────────

def train_loso_fold_enhanced(
    agent:       FLD3QN_Agent_Enhanced,
    sampler:     SubjectBalancedSampler,
    X_test:      np.ndarray,
    y_test:      np.ndarray,
    test_sid:    int,
    label_name:  str,
) -> dict:
    """
    Train for TOTAL_STEPS steps using:
      [4] Subject-balanced sampling
      [6] CORAL alignment (test batch passed to learn())
      [7] DANN via gradient reversal (inside learn())
      [9] 50 000 steps
    """
    agent.set_task(label_name)

    latest_ckpt = find_latest_checkpoint(test_sid, label_name)
    if latest_ckpt:
        agent.load_checkpoint(latest_ckpt)

    start_step = agent.steps_done
    steps      = start_step
    t_within   = 1
    val_accs   = []

    # Pre-tensor test set for CORAL batches (kept on CPU, moved per batch)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    print(f"    Training from step {start_step} → {TOTAL_STEPS} "
          f"| train subjects={len(sampler.subjects)} | test windows={len(X_test)}")

    while steps < TOTAL_STEPS:

        # ── [4] Subject-balanced sample ───────────────────────────────────────
        state, label, subj_id, idx = sampler.sample_one()
        next_state = sampler.get_next_state(idx)

        action       = agent.select_action(state)
        reward, done = agent.compute_reward(action, label, t_within)

        agent.memory.push(state, action, reward, next_state,
                          float(done), subj_id)

        # ── [6] CORAL: sample a small batch from test set ─────────────────────
        coral_batch = None
        if len(agent.memory) >= agent.batch_size and len(X_test) > 0:
            tidx        = np.random.choice(len(X_test),
                                           min(CORAL_BATCH, len(X_test)),
                                           replace=False)
            coral_batch = X_test_t[tidx].to(DEVICE)

        agent.learn(X_test_batch=coral_batch)

        steps    += 1
        t_within += 1 if not done else 0
        if done:
            t_within = 1

        if steps % CKPT_EVERY == 0:
            ckpt_path = CKPT_DIR / \
                f"loso_test{test_sid:02d}_{label_name}_step_{steps}.pth"
            agent.save_checkpoint(str(ckpt_path))
            print(f"      Checkpoint saved @ step {steps}")

        if steps % EVAL_EVERY == 0:
            val_acc  = evaluate_agent(agent, X_test, y_test, label_name)
            loss_str = (f"loss={agent.loss_history[-1]:.4f}"
                        if agent.loss_history else "")
            print(f"      Step {steps:6d} | val_acc={val_acc:.4f} | {loss_str}")
            val_accs.append((steps, val_acc))

            if val_acc > agent.best_acc:
                agent.best_acc = val_acc
                best_path = CKPT_DIR / \
                    f"loso_test{test_sid:02d}_{label_name}_best.pth"
                agent.save_checkpoint(str(best_path))

    return {"val_accs": val_accs, "best_acc": agent.best_acc}


# ─────────────────────────────────────────────────────────────────────────────
# CELL 11 – Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_agent(agent: FLD3QN_Agent_Enhanced,
                   X: np.ndarray,
                   y: np.ndarray,
                   task: str = "valence") -> float:
    """Greedy (ε=0) evaluation for a given task head."""
    agent.q_net.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(X)):
            s = torch.tensor(X[i], dtype=torch.float32)\
                     .unsqueeze(0).to(DEVICE)
            preds.append(agent.q_net(s, task=task).argmax(dim=1).item())
    agent.q_net.train()
    return accuracy_score(y, preds)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 12 – Full LOSO Pipeline with All Enhancements
# ─────────────────────────────────────────────────────────────────────────────

def run_loso_enhanced(
    all_windows:       np.ndarray,
    all_labels:        dict,
    subject_ids_array: np.ndarray,
    available_subjects: list,
    use_de_features:   bool = True,
) -> dict:
    """
    Enhanced LOSO CV.
    Per fold (test subject):
      1. Split train / test by subject.
      2. [2] Global z-score normalisation (train stats only).
      3. [1] Euclidean Alignment per split.
      4. [3] Optionally replace raw windows with DE features.
      5. Build [4] SubjectBalancedSampler for training.
      6. Train shared BiFRCNN_MultiTask agent (both heads simultaneously).
      7. Evaluate best checkpoint on test set.
    """
    results = {}
    n_subj  = len(available_subjects)

    for test_sid in available_subjects:
        print(f"\n{'='*65}")
        print(f"  LOSO Fold — Test Subject: {test_sid:02d}  "
              f"(Train: {n_subj-1} subjects)")
        print(f"{'='*65}")

        test_mask  = (subject_ids_array == test_sid)
        train_mask = ~test_mask

        X_train_raw = all_windows[train_mask]
        X_test_raw  = all_windows[test_mask]

        # ── [2] Global normalisation ──────────────────────────────────────────
        print("  [2] Global z-score normalisation …")
        X_train_n, X_test_n = global_normalize(X_train_raw, X_test_raw)

        # ── [1] Euclidean Alignment ───────────────────────────────────────────
        print("  [1] Euclidean Alignment (train) …")
        X_train_ea = euclidean_alignment(X_train_n)
        print("  [1] Euclidean Alignment (test) …")
        X_test_ea  = euclidean_alignment(X_test_n)

        # ── [3] Differential Entropy features ─────────────────────────────────
        if use_de_features:
            print("  [3] Computing DE features …")
            X_train_feat = compute_de_features(X_train_ea)   # (N, 32, 5)
            X_test_feat  = compute_de_features(X_test_ea)
        else:
            X_train_feat = X_train_ea    # (N, 32, 128)
            X_test_feat  = X_test_ea

        # Build per-subject index map for balanced sampler
        train_subj_ids = subject_ids_array[train_mask]

        # ── Create a SINGLE shared agent for both tasks ───────────────────────
        # The multi-task network trains both valence + arousal heads jointly;
        # we iterate over label tasks sequentially within the same fold so
        # memory and domain knowledge accumulates across both tasks.
        agent = FLD3QN_Agent_Enhanced(n_subjects=n_subj)
        results[test_sid] = {}

        for label_name in ["valence", "arousal"]:
            y_train = all_labels[label_name][train_mask]
            y_test  = all_labels[label_name][test_mask]

            print(f"\n  [{label_name.upper()}]  "
                  f"Train={len(X_train_feat)} | Test={len(X_test_feat)}")

            # [4] Subject-balanced sampler
            sampler = SubjectBalancedSampler(X_train_feat, y_train,
                                             train_subj_ids)

            # Reset step counter & memory for each task head
            # (keep network weights — multi-task warm-start)
            agent.steps_done  = 0
            agent.best_acc    = 0.0
            agent.memory      = ReplayBuffer(200_000)
            agent.loss_history = []
            agent.scheduler   = optim.lr_scheduler.CosineAnnealingLR(
                                    agent.optimizer,
                                    T_max=TOTAL_STEPS, eta_min=1e-5)

            train_loso_fold_enhanced(
                agent, sampler, X_test_feat, y_test, test_sid, label_name
            )

            # Load best checkpoint for final test evaluation
            best_path = CKPT_DIR / \
                f"loso_test{test_sid:02d}_{label_name}_best.pth"
            if best_path.exists():
                agent.load_checkpoint(str(best_path))

            test_acc = evaluate_agent(agent, X_test_feat, y_test, label_name)
            results[test_sid][label_name] = test_acc
            print(f"  ► Subject {test_sid:02d} | {label_name.upper()} "
                  f"test acc = {test_acc:.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CELL 13 – Results Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_loso_results(results: dict, tag: str = "Enhanced"):
    subjects = sorted(results.keys())
    val_accs = [results[s]["valence"] for s in subjects]
    aro_accs = [results[s]["arousal"] for s in subjects]

    x = np.arange(len(subjects))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    ax = axes[0]
    ax.bar(x - w/2, val_accs, w, label="Valence", color="#4C9BE8")
    ax.bar(x + w/2, aro_accs, w, label="Arousal", color="#E8884C")
    ax.axhline(np.mean(val_accs), color="#4C9BE8", linestyle="--", alpha=0.7,
               label=f"Val mean={np.mean(val_accs):.3f}")
    ax.axhline(np.mean(aro_accs), color="#E8884C", linestyle="--", alpha=0.7,
               label=f"Aro mean={np.mean(aro_accs):.3f}")
    ax.set_xlabel("Held-Out Test Subject")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"FLD3QN {tag} – LOSO Per-Subject Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s:02d}" for s in subjects], rotation=45)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    ax2 = axes[1]
    bp = ax2.boxplot([val_accs, aro_accs], labels=["Valence", "Arousal"],
                     patch_artist=True)
    bp["boxes"][0].set_facecolor("#4C9BE8")
    bp["boxes"][1].set_facecolor("#E8884C")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"FLD3QN {tag} – LOSO Accuracy Distribution")
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    out_path = f"/kaggle/working/fld3qn_loso_{tag.lower()}_results.png"
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"\nPlot saved to {out_path}")

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print(f"{'Subject':<10} {'Val Acc':>12} {'Aro Acc':>12}")
    print("-"*55)
    for s in subjects:
        print(f"S{s:02d}       "
              f"{results[s]['valence']:>12.4f} "
              f"{results[s]['arousal']:>12.4f}")
    print("-"*55)
    print(f"{'Mean':<10} {np.mean(val_accs):>12.4f} {np.mean(aro_accs):>12.4f}")
    print(f"{'Std':<10} {np.std(val_accs):>12.4f}  {np.std(aro_accs):>12.4f}")
    print("="*55)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 14 – Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Enhanced Subject-Independent LOSO pipeline.

    Improvements active:
      [1]  Euclidean Alignment
      [2]  Global z-score normalisation (train stats → test)
      [3]  Differential Entropy features (5 bands × 32 ch)
      [4]  Subject-balanced batch sampling
      [5]  Multi-task head (valence + arousal share backbone)
      [6]  CORAL domain alignment loss
      [7]  Domain-Adversarial (DANN) gradient reversal
      [8]  SE channel attention in residual blocks
      [9]  50 000 training steps (10× baseline)
      [10] Soft / continuous labels stored (binary threshold used for RL)

    For a quick smoke-test, set SUBJECT_IDS = [1, 2, 3].
    To disable DE features and use raw EEG, set use_de_features=False.
    """
    
    # Generating subjects 1 through 32, but explicitly excluding outliers 3 and 21
    SUBJECT_IDS = [i for i in range(1, 33) if i not in [3, 21]]

    USE_DE = True   # [3] set False to use raw EEG windows instead

    print("=" * 65)
    print("  FLD3QN Enhanced – Subject-Independent LOSO")
    print("=" * 65)
    print(f"  DE features   : {USE_DE}")
    print(f"  Total steps   : {TOTAL_STEPS}")
    print(f"  CORAL weight  : {LAMBDA_CORAL}")
    print(f"  DANN weight   : {LAMBDA_DANN}")
    print("=" * 65)

    print("\nLoading all subjects …")
    all_windows, all_labels, subject_ids_array, available_subjects = \
        load_all_subjects(SUBJECT_IDS)

    results = run_loso_enhanced(
        all_windows, all_labels, subject_ids_array, available_subjects,
        use_de_features=USE_DE,
    )

    if results:
        plot_loso_results(results, tag="Enhanced")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = main()