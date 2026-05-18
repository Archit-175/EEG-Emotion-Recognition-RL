"""
=============================================================================
FLD3QN: Frontal Lobe Double Dueling Deep Q Network
Exact Reproduction of IEEE TNNLS 2024 Paper:
"Brain Emotion Perception Inspired EEG Emotion Recognition
 With Deep Reinforcement Learning"
=============================================================================
Dataset  : DEAP (preprocessed .dat files on Kaggle)
Framework: PyTorch
Usage    : Run cell-by-cell in a Kaggle Notebook (GPU recommended)
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
import math
import collections
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import StratifiedKFold
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
# DEAP 32-channel layout (10-20 system, 0-indexed)
# Full electrode order:
# Fp1  AF3  F3  F7  FC5  FC1  C3  T7  CP5  CP1  P3  P7  PO3  O1
# Oz   Pz   P4  P8  PO4  O2   T8  CP6  CP2  C4  T8  FC6  FC2  F4
# F8   AF4  Fp2 Fz

ALL_CHANNELS = list(range(32))

# ── Hemisphere splits ────────────────────────────────────────────────────────
# Left  (odd-suffix / left-side channels in DEAP layout)
LEFT_CH  = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]   # 13 channels
# Right (even-suffix / right-side channels)
RIGHT_CH = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]  # 13 channels
# Remaining mid-line channels go to both (we include in concatenation as shared)

# ── Frontal lobe channels ────────────────────────────────────────────────────
# Fp1(0) AF3(1) F3(2) F7(3) FC5(4) FC1(5) FC2(26) F4(27) F8(28) AF4(29) Fp2(30) Fz(31)
LEFT_FRONTAL  = [0, 1, 2, 3, 4, 5]   # left frontal
RIGHT_FRONTAL = [26, 27, 28, 29, 30, 31]  # right frontal

print(f"Left hemisphere channels  : {len(LEFT_CH)}")
print(f"Right hemisphere channels : {len(RIGHT_CH)}")
print(f"Left frontal channels     : {len(LEFT_FRONTAL)}")
print(f"Right frontal channels    : {len(RIGHT_FRONTAL)}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 – Data Loading & Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

DEAP_PATH   = "/home/tanmoyhazra/rl_project_24/DEAP"  # adjust if needed
FS          = 128          # sampling rate (Hz)
TRIAL_LEN   = 63           # seconds (after removing 3-sec baseline from 60-sec trial)
BASELINE_S  = 3            # seconds of baseline to remove
WIN_SIZE    = FS           # 1-second window = 128 samples
LAMBDA_REW  = 0.1          # reward penalty coefficient


def load_subject(subj_id: int) -> tuple:
    """
    Load one DEAP subject file and return preprocessed windows + labels.

    Returns
    -------
    windows : np.ndarray, shape (N, 32, 128), float32
    labels  : dict  {'valence': np.ndarray(N,), 'arousal': np.ndarray(N,)}
    """
    # ── Locate file ──────────────────────────────────────────────────────────
    pattern = os.path.join(DEAP_PATH, f"s{subj_id:02d}.dat")
    files   = glob.glob(pattern)
    if not files:
        # Try alternate naming
        pattern = os.path.join(DEAP_PATH, "data_preprocessed_python",
                               f"s{subj_id:02d}.dat")
        files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"Cannot find DEAP file for subject {subj_id}")

    with open(files[0], "rb") as f:
        data = pickle.load(f, encoding="latin1")

    eeg    = data["data"][:, :32, :]   # (40 trials, 32 ch, 8064 samples)
    labels = data["labels"]             # (40 trials, 4) – val, arous, dom, lik

    # ── Remove 3-sec baseline (384 samples at 128 Hz) ───────────────────────
    baseline_samples = BASELINE_S * FS
    eeg = eeg[:, :, baseline_samples:]  # (40, 32, 7680) → 60 sec * 128 = 7680

    # ── Subtract per-trial baseline mean (computed on removed segment) ───────
    # Paper: subtract mean of baseline period
    # We use the mean of the first 3 sec (already removed) as baseline.
    # Approximation: subtract the global mean of each channel across the trial.
    eeg = eeg - eeg.mean(axis=2, keepdims=True)

    # ── Z-score normalization per trial per channel ──────────────────────────
    std = eeg.std(axis=2, keepdims=True)
    std[std == 0] = 1e-8
    eeg = eeg / std

    # ── Sliding-window segmentation (1-sec, no overlap as in paper) ──────────
    n_trials, n_ch, n_samples = eeg.shape
    windows_list, val_list, aro_list = [], [], []

    for t in range(n_trials):
        n_wins = n_samples // WIN_SIZE
        for w in range(n_wins):
            seg = eeg[t, :, w * WIN_SIZE:(w + 1) * WIN_SIZE]  # (32, 128)
            windows_list.append(seg)
            val_list.append(int(labels[t, 0] > 5))
            aro_list.append(int(labels[t, 1] > 5))

    windows = np.array(windows_list, dtype=np.float32)  # (N, 32, 128)
    label_dict = {
        "valence": np.array(val_list, dtype=np.int64),
        "arousal": np.array(aro_list, dtype=np.int64),
    }

    print(f"  Subject {subj_id:02d}: {windows.shape[0]} windows | "
          f"Valence pos={label_dict['valence'].mean():.2f} "
          f"Arousal pos={label_dict['arousal'].mean():.2f}")
    return windows, label_dict


# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 – BiFRCNN Building Blocks
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    1-D temporal residual block with TWO skip connections:
      skip1 : input → middle  (added after first conv)
      skip2 : input → output  (added after second conv)
    Matches paper description.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        pad = kernel // 2

        self.conv1 = nn.Conv2d(in_ch, out_ch, (1, kernel), padding=(0, pad))
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, (1, kernel), padding=(0, pad))
        self.bn2   = nn.BatchNorm2d(out_ch)

        # Skip connection projections (project input channels → target channels)
        self.skip1 = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.skip2 = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = out + self.skip1(identity)          # skip 1: input → middle

        out = self.bn2(self.conv2(out))
        out = out + self.skip2(identity)          # skip 2: input → output
        out = self.relu(out)
        return out


class StreamEncoder(nn.Module):
    """
    Single-stream encoder for one hemisphere or frontal group.
    Input  : (B, 1, n_ch, 128)
    Output : (B, feature_dim)
    """

    def __init__(self, n_ch: int, feature_dim: int = 64):
        super().__init__()
        self.conv_in = nn.Conv2d(1, 16, (1, 3), padding=(0, 1))
        self.bn_in   = nn.BatchNorm2d(16)
        self.relu    = nn.ReLU(inplace=True)

        self.res1 = ResidualBlock(16, 32)
        self.res2 = ResidualBlock(32, 64)

        # Adaptive pool to fixed size then flatten
        self.pool    = nn.AdaptiveAvgPool2d((1, 8))
        self.flatten = nn.Flatten()
        self.fc      = nn.Linear(64 * 8, feature_dim)

    def forward(self, x):
        # x: (B, n_ch, 128)  → add channel dim
        x = x.unsqueeze(1)           # (B, 1, n_ch, 128)
        x = self.relu(self.bn_in(self.conv_in(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.pool(x)             # (B, 64, 1, 8)
        x = self.flatten(x)          # (B, 512)
        x = self.relu(self.fc(x))    # (B, feature_dim)
        return x


class BiFRCNN(nn.Module):
    """
    Bi-Hemispheric Frontal-Region Convolutional Neural Network.

    Four streams:
      1. Left hemisphere
      2. Right hemisphere
      3. Left frontal
      4. Right frontal

    Merge strategy (paper):
      - Combine hemisphere + its frontal lobe stream → fused representation
      - Concatenate left-fused + right-fused
      - Followed by Softmax + Dueling heads
    """

    def __init__(self, feature_dim: int = 64, n_actions: int = 2):
        super().__init__()

        # Per-stream encoders
        self.enc_left_hemi    = StreamEncoder(len(LEFT_CH),     feature_dim)
        self.enc_right_hemi   = StreamEncoder(len(RIGHT_CH),    feature_dim)
        self.enc_left_front   = StreamEncoder(len(LEFT_FRONTAL), feature_dim)
        self.enc_right_front  = StreamEncoder(len(RIGHT_FRONTAL), feature_dim)

        # Merge hemisphere + frontal (paper: merge then concat)
        fused_dim = feature_dim * 2      # hemi + frontal concatenated
        self.merge_left  = nn.Sequential(
            nn.Linear(fused_dim, feature_dim), nn.ReLU(inplace=True))
        self.merge_right = nn.Sequential(
            nn.Linear(fused_dim, feature_dim), nn.ReLU(inplace=True))

        # Final representation
        combined_dim = feature_dim * 2   # left_fused + right_fused
        self.shared = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(inplace=True),
        )

        # ── Dueling heads ───────────────────────────────────────────────────
        self.value_stream     = nn.Linear(128, 1)
        self.advantage_stream = nn.Linear(128, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 32, 128) – full 32-ch EEG window
        Returns Q-values : (B, n_actions)
        """
        # ── Extract sub-arrays per stream ───────────────────────────────────
        xl_hemi  = x[:, LEFT_CH,     :]   # (B, 13, 128)
        xr_hemi  = x[:, RIGHT_CH,    :]   # (B, 13, 128)
        xl_front = x[:, LEFT_FRONTAL,:]   # (B,  6, 128)
        xr_front = x[:, RIGHT_FRONTAL,:]  # (B,  6, 128)

        # ── Encode ──────────────────────────────────────────────────────────
        fl_hemi  = self.enc_left_hemi(xl_hemi)
        fr_hemi  = self.enc_right_hemi(xr_hemi)
        fl_front = self.enc_left_front(xl_front)
        fr_front = self.enc_right_front(xr_front)

        # ── Merge hemisphere + frontal ───────────────────────────────────────
        left_fused  = self.merge_left(torch.cat([fl_hemi,  fl_front], dim=1))
        right_fused = self.merge_right(torch.cat([fr_hemi, fr_front], dim=1))

        # ── Combine & shared layer ───────────────────────────────────────────
        combined = torch.cat([left_fused, right_fused], dim=1)
        shared   = self.shared(combined)

        # ── Dueling Q-value computation ──────────────────────────────────────
        V = self.value_stream(shared)                          # (B, 1)
        A = self.advantage_stream(shared)                      # (B, n_actions)
        Q = V + A - A.mean(dim=1, keepdim=True)               # (B, n_actions)
        return Q


# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 – Experience Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

Transition = collections.namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.tensor(np.array(state),      dtype=torch.float32).to(DEVICE),
            torch.tensor(action,               dtype=torch.long).to(DEVICE),
            torch.tensor(reward,               dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(next_state), dtype=torch.float32).to(DEVICE),
            torch.tensor(done,                 dtype=torch.float32).to(DEVICE),
        )

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 – FLD3QN Agent (Double Dueling DQN)
# ─────────────────────────────────────────────────────────────────────────────

class FLD3QN_Agent:
    """
    Double Dueling Deep Q-Network with BiFRCNN feature extractor.

    Hyperparameters (paper + modified 5000-step budget):
      lr            : 0.00025
      gamma         : 0.75
      epsilon       : 0.1   (fixed; paper uses fixed ε-greedy)
      batch_size    : 32
      target_update : every 10 000 steps
      buffer_size   : 100 000
    """

    def __init__(self, n_actions: int = 2, lr: float = 2.5e-4,
                 gamma: float = 0.75, epsilon: float = 0.1,
                 batch_size: int = 32, buffer_size: int = 100_000,
                 target_update_freq: int = 10_000):

        self.n_actions          = n_actions
        self.gamma              = gamma
        self.epsilon            = epsilon
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.steps_done         = 0

        # ── Networks ─────────────────────────────────────────────────────────
        self.q_net      = BiFRCNN(feature_dim=64, n_actions=n_actions).to(DEVICE)
        self.target_net = copy.deepcopy(self.q_net).to(DEVICE)
        self.target_net.eval()

        # ── Optimizer & loss ─────────────────────────────────────────────────
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn   = nn.MSELoss()

        # ── Replay buffer ─────────────────────────────────────────────────────
        self.memory = ReplayBuffer(buffer_size)

        # ── Tracking ─────────────────────────────────────────────────────────
        self.loss_history = []
        self.best_acc     = 0.0

    # ── Action selection ──────────────────────────────────────────────────────
    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            q = self.q_net(s)
            return q.argmax(dim=1).item()

    # ── Reward function ───────────────────────────────────────────────────────
    @staticmethod
    def compute_reward(action: int, label: int, t: int,
                       lam: float = LAMBDA_REW) -> tuple:
        """
        rt = +1                       if correct
           = -1 - (λ / t)             if incorrect
        Returns (reward, done)
        """
        if action == label:
            return 1.0, False
        else:
            return -1.0 - (lam / max(t, 1)), True   # episode ends on first mistake

    # ── Learning step ─────────────────────────────────────────────────────────
    def learn(self):
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = \
            self.memory.sample(self.batch_size)

        # Current Q-values
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target:
        #   action chosen by Q_current, evaluated by Q_target
        with torch.no_grad():
            next_actions  = self.q_net(next_states).argmax(dim=1)
            next_q_target = self.target_net(next_states)\
                                .gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q_target * (1.0 - dones)

        loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.steps_done += 1

        # Periodic target-network hard update
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        loss_val = loss.item()
        self.loss_history.append(loss_val)
        return loss_val

    # ── Checkpoint I/O ────────────────────────────────────────────────────────
    def save_checkpoint(self, path: str):
        torch.save({
            "q_net_state":      self.q_net.state_dict(),
            "target_net_state": self.target_net.state_dict(),
            "optimizer_state":  self.optimizer.state_dict(),
            "steps_done":       self.steps_done,
            "best_acc":         self.best_acc,
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=DEVICE)
        self.q_net.load_state_dict(ckpt["q_net_state"])
        self.target_net.load_state_dict(ckpt["target_net_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.steps_done = ckpt["steps_done"]
        self.best_acc   = ckpt["best_acc"]
        print(f"  Resumed from step {self.steps_done}, best_acc={self.best_acc:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 – Training Loop (5000 steps per subject, per label)
# ─────────────────────────────────────────────────────────────────────────────

CKPT_DIR     = Path("/home/tanmoyhazra/rl_project_24/checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

TOTAL_STEPS  = 5_000
CKPT_EVERY   = 2_000
EVAL_EVERY   = 500


def find_latest_checkpoint(subj_id: int, label: str) -> str | None:
    pattern = str(CKPT_DIR / f"s{subj_id:02d}_{label}_step_*.pth")
    files   = glob.glob(pattern)
    if not files:
        return None
    # Sort by step number
    files.sort(key=lambda f: int(f.split("_step_")[1].replace(".pth", "")))
    return files[-1]


def train_subject(
    agent: FLD3QN_Agent,
    windows: np.ndarray,
    labels: np.ndarray,
    subj_id: int,
    label_name: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> dict:
    """
    Train the agent for TOTAL_STEPS steps on the training split.
    Uses the RL episode formulation from the paper.

    Returns dict with training stats.
    """

    X_train, y_train = windows[train_idx], labels[train_idx]
    X_val,   y_val   = windows[val_idx],   labels[val_idx]

    # ── Auto-resume ───────────────────────────────────────────────────────────
    latest_ckpt = find_latest_checkpoint(subj_id, label_name)
    if latest_ckpt:
        agent.load_checkpoint(latest_ckpt)

    start_step   = agent.steps_done
    steps        = start_step
    episode      = 0
    val_accs     = []

    print(f"    Training from step {start_step} → {TOTAL_STEPS}")

    while steps < TOTAL_STEPS:
        # ── New episode: shuffle training data ─────────────────────────────
        perm     = np.random.permutation(len(X_train))
        t_within = 1   # within-episode step counter (for reward denominator)

        for idx in perm:
            if steps >= TOTAL_STEPS:
                break

            state  = X_train[idx]        # (32, 128)
            label  = int(y_train[idx])

            # ── ε-greedy action ────────────────────────────────────────────
            action = agent.select_action(state)

            # ── Reward & termination ────────────────────────────────────────
            reward, done = agent.compute_reward(action, label, t_within)

            # ── Next state (paper uses next sample as next state) ───────────
            next_idx   = (idx + 1) % len(X_train)
            next_state = X_train[next_idx]

            # ── Store transition ────────────────────────────────────────────
            agent.memory.push(state, action, reward, next_state,
                              float(done))

            # ── Learn ──────────────────────────────────────────────────────
            agent.learn()
            steps    += 1
            t_within += 1

            # ── Checkpoint ─────────────────────────────────────────────────
            if steps % CKPT_EVERY == 0:
                ckpt_path = CKPT_DIR / f"s{subj_id:02d}_{label_name}_step_{steps}.pth"
                agent.save_checkpoint(str(ckpt_path))
                print(f"      Checkpoint saved @ step {steps}")

            # ── Validation ─────────────────────────────────────────────────
            if steps % EVAL_EVERY == 0:
                val_acc = evaluate_agent(agent, X_val, y_val)
                val_accs.append((steps, val_acc))
                print(f"      Step {steps:5d} | val_acc={val_acc:.4f} | "
                      f"loss={agent.loss_history[-1]:.4f}" if agent.loss_history else "")
                if val_acc > agent.best_acc:
                    agent.best_acc = val_acc
                    best_path = CKPT_DIR / f"s{subj_id:02d}_{label_name}_best.pth"
                    agent.save_checkpoint(str(best_path))

            if done:   # episode terminates on first mistake → restart episode
                break

        episode += 1

    return {"val_accs": val_accs, "best_acc": agent.best_acc}


# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 – Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_agent(agent: FLD3QN_Agent,
                   X: np.ndarray,
                   y: np.ndarray) -> float:
    """Greedy (ε=0) evaluation on array X with labels y."""
    agent.q_net.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(X)):
            s = torch.tensor(X[i], dtype=torch.float32)\
                     .unsqueeze(0).to(DEVICE)
            q = agent.q_net(s)
            preds.append(q.argmax(dim=1).item())
    agent.q_net.train()
    return accuracy_score(y, preds)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 – 10-Fold Cross-Validation per Subject
# ─────────────────────────────────────────────────────────────────────────────

def run_kfold_subject(subj_id: int,
                      windows: np.ndarray,
                      label_dict: dict,
                      n_splits: int = 10) -> dict:
    """
    Subject-dependent 10-fold CV.
    Trains separate agents for valence and arousal.
    Returns mean accuracy per label.
    """
    results = {}

    for label_name in ["valence", "arousal"]:
        y = label_dict[label_name]
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=SEED)
        fold_accs = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(windows, y)):
            print(f"\n  [Subj {subj_id:02d} | {label_name.upper()} | "
                  f"Fold {fold+1}/{n_splits}]")

            agent = FLD3QN_Agent()

            stats = train_subject(
                agent, windows, y, subj_id, label_name,
                train_idx, val_idx
            )

            # Load best model for final fold evaluation
            best_path = CKPT_DIR / f"s{subj_id:02d}_{label_name}_best.pth"
            if best_path.exists():
                agent.load_checkpoint(str(best_path))

            test_acc = evaluate_agent(agent, windows[val_idx], y[val_idx])
            fold_accs.append(test_acc)
            print(f"  Fold {fold+1} test acc: {test_acc:.4f}")

        mean_acc = np.mean(fold_accs)
        std_acc  = np.std(fold_accs)
        results[label_name] = {"fold_accs": fold_accs,
                                "mean": mean_acc,
                                "std": std_acc}
        print(f"\n  {label_name.upper()} mean acc: {mean_acc:.4f} ± {std_acc:.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CELL 10 – Results Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(all_results: dict):
    """
    Plots:
      1. Per-subject accuracy (valence + arousal) bar chart
      2. Box plots across subjects
    """
    subjects  = sorted(all_results.keys())
    val_means = [all_results[s]["valence"]["mean"] for s in subjects]
    aro_means = [all_results[s]["arousal"]["mean"] for s in subjects]
    val_stds  = [all_results[s]["valence"]["std"]  for s in subjects]
    aro_stds  = [all_results[s]["arousal"]["std"]  for s in subjects]

    x = np.arange(len(subjects))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # ── Bar chart ─────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.bar(x - w/2, val_means, w, yerr=val_stds, label="Valence",
           color="#4C9BE8", capsize=3)
    ax.bar(x + w/2, aro_means, w, yerr=aro_stds, label="Arousal",
           color="#E8884C", capsize=3)
    ax.set_xlabel("Subject ID")
    ax.set_ylabel("Accuracy")
    ax.set_title("FLD3QN – Per-Subject Accuracy (10-Fold CV)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s:02d}" for s in subjects], rotation=45)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.axhline(0.98, color="red", linestyle="--", alpha=0.5, label="Paper ~98%")

    # ── Summary means ─────────────────────────────────────────────────────────
    ax2 = axes[1]
    all_val = [acc for s in subjects
               for acc in all_results[s]["valence"]["fold_accs"]]
    all_aro = [acc for s in subjects
               for acc in all_results[s]["arousal"]["fold_accs"]]
    ax2.boxplot([all_val, all_aro], labels=["Valence", "Arousal"])
    ax2.set_ylabel("Accuracy")
    ax2.set_title("FLD3QN – Overall Accuracy Distribution")
    ax2.axhline(0.98, color="red", linestyle="--", alpha=0.5, label="Paper ~98%")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("/home/tanmoyhazra/rl_project_24/fld3qn_results.png", dpi=150)
    plt.show()
    print("\nPlot saved to /home/tanmoyhazra/rl_project_24/fld3qn_results.png")

    # ── Print summary table ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"{'Subject':<10} {'Val Acc':>10} {'Val Std':>10} "
          f"{'Aro Acc':>10} {'Aro Std':>10}")
    print("-"*60)
    for s in subjects:
        print(f"S{s:02d}       "
              f"{all_results[s]['valence']['mean']:>10.4f} "
              f"{all_results[s]['valence']['std']:>10.4f} "
              f"{all_results[s]['arousal']['mean']:>10.4f} "
              f"{all_results[s]['arousal']['std']:>10.4f}")
    print("-"*60)
    print(f"{'Mean':<10} "
          f"{np.mean(val_means):>10.4f} {'':>10} "
          f"{np.mean(aro_means):>10.4f}")
    print("="*60)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 11 – Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Run full pipeline for all 32 DEAP subjects.
    Edit SUBJECT_IDS to restrict to a subset for quick testing.
    """
    SUBJECT_IDS = list(range(1, 33))   # 1–32
    # ── Quick smoke-test: uncomment to run on first 2 subjects only ──────────
    # SUBJECT_IDS = [1, 2]

    all_results = {}

    for sid in SUBJECT_IDS:
        print(f"\n{'='*60}")
        print(f"  Processing Subject {sid:02d}")
        print(f"{'='*60}")

        try:
            windows, label_dict = load_subject(sid)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        subj_results = run_kfold_subject(sid, windows, label_dict,
                                         n_splits=10)
        all_results[sid] = subj_results

    if all_results:
        plot_results(all_results)

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = main()