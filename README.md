# 🧠 NeuroRL: Subject-Independent EEG Emotion Recognition using Deep Reinforcement Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![Reinforcement Learning](https://img.shields.io/badge/RL-D3QN-green?style=for-the-badge)
![EEG](https://img.shields.io/badge/EEG-Brain%20Signals-purple?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Research%20Project-orange?style=for-the-badge)

**Calibration-Free EEG Emotion Recognition using Deep Reinforcement Learning, Domain Adaptation, and Adaptive Learning**

</div>

---

## 📌 Overview

Emotion recognition from EEG signals is an important task in Brain-Computer Interface (BCI) systems. Most existing approaches rely on **subject-specific calibration**, making deployment difficult in real-world scenarios.

This project proposes a **subject-independent EEG emotion recognition framework** that eliminates calibration requirements and improves cross-subject generalization using:

- 🧠 CNN-based feature extraction (**BiFRCNN**)
- 🎯 Double Dueling Deep Q Network (**D3QN**)
- 🔄 Multi-level Domain Adaptation
- 🚀 Hierarchical Adaptive Reinforcement Learning (**HARL**)

The framework is evaluated on:

- **DEAP Dataset**
- **SEED-IV Dataset**

---

# 🏗 System Architecture

```text
Raw EEG Signals
        │
        ▼
Differential Entropy Features
        │
        ▼
Global Normalization
        │
        ▼
Domain Adaptation
(EA + CORAL + DANN)
        │
        ▼
BiFRCNN Feature Extractor
        │
        ▼
D3QN / HARL Agent
        │
        ▼
Reward Update
        │
        ▼
Emotion Prediction
```

---

# 📂 Datasets

## 1️⃣ DEAP Dataset

Emotion dimensions:

### Valence
- Positive
- Negative

### Arousal
- High
- Low

---

## 2️⃣ SEED-IV Dataset

Emotion classes:

- Neutral
- Happy
- Fear
- Sad

SEED-IV contains multiple sessions introducing strong domain variability.

---

# 🧩 Reinforcement Learning Formulation

The problem is modeled as a **Markov Decision Process (MDP)**.

---

## State Space (S)

State corresponds to EEG features extracted from brain activity:

\[
S = DE(EEG)
\]

where:

DE = Differential Entropy features.

---

## Action Space (A)

Actions correspond to emotion predictions.

### DEAP

\[
A=
\{Valence, Arousal\}
\]

### SEED-IV

\[
A=
\{
Neutral,
Happy,
Fear,
Sad
\}
\]

---

## Reward Function

A time-dependent reward strategy is used:

\[
r_t=
\begin{cases}
+1,& a_t=l_t\\
-1-\lambda t,& a_t\ne l_t
\end{cases}
\]

where:

- \(t\) = timestep
- \(\lambda=0.1\)

### Motivation

✔ Correct prediction → positive reward

✔ Wrong prediction → increasing penalty

This improves stability and discourages repeated mistakes.

---

## Discount Factor

\[
\gamma=0.75
\]

Future rewards are considered while learning.

---

# 🎯 Deep Reinforcement Learning

The framework uses:

# Double Dueling Deep Q Network (D3QN)

D3QN improves traditional Q-learning using:

- Double Q Learning
- Dueling Architecture

---

## Q-learning Update

\[
Q(s,a)
\leftarrow
Q(s,a)
+
\alpha
[
r+
\gamma
\max Q(s',a')
-
Q(s,a)
]
\]

---

## Double Q Learning

Target estimation:

\[
Q^*(s,a)
=
r+
\gamma
Q
(
s',
argmaxQ(s',a';\theta),
\theta'
)
\]

Two networks are used:

### Main Network

Selects action.

### Target Network

Evaluates action.

This reduces overestimation.

---

## Dueling Architecture

Q-value decomposition:

\[
Q(s,a)
=
V(s)
+
\left(
A(s,a)
-
\frac1{|A|}
\sum A(s,a')
\right)
\]

Where:

### State Value

\[
V(s)
\]

Represents importance of state.

---

### Advantage Function

\[
A(s,a)
\]

Represents action usefulness.

---

# 🧠 CNN Feature Extraction

Feature extraction is performed using:

## BiFRCNN

Components:

- Convolution Layers
- Feature Refinement
- Attention Mechanism

BiFRCNN learns important EEG representations before RL decision-making.

---

# 🔄 Domain Adaptation

Inter-subject variability is a major challenge.

To reduce domain shift, three adaptation stages are introduced.

---

## Signal-Level Adaptation

### Euclidean Alignment (EA)

Aligns EEG covariance distributions.

---

## Feature-Level Adaptation

### CORAL

Minimizes feature covariance mismatch:

\[
L_{CORAL}
=
\frac1{4d^2}
||C_s-C_t||_F^2
\]

where:

- \(C_s\) = source covariance
- \(C_t\) = target covariance

---

## Representation-Level Adaptation

### DANN

Uses adversarial learning:

\[
L=
L_{task}
+
\lambda L_{domain}
\]

Helps learn subject-invariant representations.

---

# 🚀 HARL Extension (SEED-IV)

For SEED-IV:

### Hierarchical Adaptive Reinforcement Learning (HARL)

HARL dynamically adjusts adaptation weights.

Uses PPO-based optimization:

\[
\lambda^*
=
\lambda
(1+\bar w)
\]

where:

\[
\bar w
\]

is adaptive importance weight.

---

## HARL Reward

\[
r=
-CE
+
0.2(1-g)
\]

where:

- CE = classification loss
- g = domain gap

Goal:

✔ Improve alignment

✔ Reduce session variability

✔ Better generalization

---

# 📊 Experimental Results

## DEAP Dataset

| Metric | Accuracy |
|----------|-----------|
| Valence | 56.22% |
| Arousal | 58.78% |

---

## SEED-IV Dataset

| Metric | Accuracy |
|----------|-----------|
| 4-Class Emotion | 36.03% |

Chance level:

\[
25\%
\]

The model performs above baseline without calibration.

---

# 📈 Comparison with Previous Methods

| Method | Accuracy | Limitation |
|----------|-----------|-------------|
| Subject-Dependent Models | ~98% | Requires calibration |
| SVM | ~51% | Poor generalization |
| EEGNet | ~50% | Limited adaptability |
| Proposed Model | 56–58% | Calibration-free |

---

# ⭐ Key Contributions

✅ Subject-independent EEG emotion recognition

✅ Calibration-free prediction

✅ Deep RL based decision making

✅ D3QN implementation

✅ HARL extension

✅ Multi-level domain adaptation

✅ Cross-dataset evaluation

✅ Practical BCI deployment approach

---

# 🛠 Tech Stack

### Languages

- Python

### Frameworks

- PyTorch
- NumPy
- Pandas

### Machine Learning

- Reinforcement Learning
- Deep RL
- D3QN
- PPO

### Deep Learning

- CNN
- BiFRCNN

### Signal Processing

- EEG Processing
- Differential Entropy

### Domain Adaptation

- EA
- CORAL
- DANN

---

# 📁 Repository Structure

```bash
NeuroRL/
│
├── data/
│
├── notebooks/
│
├── src/
│   ├── preprocessing/
│   │   ├── feature_extraction.py
│   │   ├── normalization.py
│   │   └── euclidean_alignment.py
│   │
│   ├── adaptation/
│   │   ├── coral.py
│   │   └── dann.py
│   │
│   ├── models/
│   │   ├── bifrcnn.py
│   │   ├── d3qn.py
│   │   └── harl.py
│   │
│   └── train.py
│
├── results/
│
├── paper/
│
├── README.md
│
└── requirements.txt
```

---

# 🔮 Future Work

- Transformer-based EEG architectures
- Real-time inference
- Multi-modal emotion recognition
- BCI deployment
- Online adaptive learning

---

# 📚 Reference

Research implementation inspired by recent subject-independent EEG emotion recognition frameworks integrating reinforcement learning and adaptive learning.

---

<div align="center">

### 👨‍💻 Author

**Archit Savaliya**

AI Undergraduate • Reinforcement Learning • Deep Learning • Brain Computer Interfaces

</div>
