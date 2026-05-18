# 🔐 Adaptive Trust Scores in Federated Learning Models

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Flower](https://img.shields.io/badge/Flower_FLWR-v1.8-499848?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Research_Complete-brightgreen?style=for-the-badge)
![Patent](https://img.shields.io/badge/Patent-Pending-orange?style=for-the-badge)
![Institution](https://img.shields.io/badge/MUJ-B.Tech_Project-blue?style=for-the-badge)

> **⚠️ PATENT PENDING — See Intellectual Property Notice below before using or referencing this work.**

*A research project submitted to Manipal University Jaipur in partial fulfilment of the B.Tech CSE (AI & ML) degree.*

[📄 Project Report](#-project-report) • [🔬 Methodology](#-methodology) • [📊 Results](#-results--observations) • [🛠️ Tech Stack](#%EF%B8%8F-tech-stack) • [👤 Authors](#-authors)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Methodology](#-methodology)
- [Experimental Setup](#-experimental-setup)
- [Results & Observations](#-results--observations)
- [Tech Stack](#%EF%B8%8F-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Research Contribution](#-research-contribution)
- [Future Work](#-future-work)
- [Intellectual Property Notice](#-intellectual-property-notice)
- [Citation](#-citation)
- [Authors](#-authors)
- [Copyright](#-copyright-notice)

---

## 🧭 Overview

**Federated Learning (FL)** enables collaborative model training across distributed clients without
exposing raw data to a central server. However, classical aggregation frameworks such as **FedAvg**
assign uniform weights to all client contributions — leaving the global model vulnerable to
**Byzantine attacks** and imposing an inefficient, one-size-fits-all privacy cost on every participant.

This project introduces an **Adaptive Trust Score mechanism** for FL that:

- Dynamically adjusts each client's **aggregation influence** based on a temporal trust formulation
- Implements **adaptive differential-privacy noise injection** — more noise for low-trust (potentially
  malicious) clients, less for high-trust ones
- Effectively **isolates malicious clients** over successive communication rounds without sacrificing
  global model accuracy

> Experimental results show the malicious client's trust decaying from **0.41 → 0.098** over 10 rounds
> while all 9 benign clients converge to trust **> 0.94**, with global accuracy sustained at **~94%**
> on non-IID MNIST.

---

## ❗ Problem Statement

Classical FL frameworks treat all client updates identically during aggregation, creating two
critical vulnerabilities:

| Vulnerability | Description |
|---|---|
| **Byzantine Susceptibility** | A single malicious client submitting poisoned gradients can corrupt the global model under uniform-weight FedAvg |
| **Uniform Privacy Overhead** | Standard DP techniques add identical noise to all clients, unnecessarily degrading accuracy for reliable participants |

The core gap: **no mechanism simultaneously identifies unreliable clients AND adjusts both their
aggregation influence and privacy noise in proportion to their assessed trustworthiness.**

---

## ✨ Key Features

- 🧠 **Temporal Trust Scoring** — Per-client trust maintained as an exponential moving average,
  providing memory of past conduct across rounds
- ⚖️ **Trust-Weighted Aggregation** — Global model computed as a trust-weighted average, replacing
  FedAvg's uniform weighting
- 🔒 **Adaptive DP Noise Injection** — Noise σ scales inversely with trust; attackers receive
  progressively amplified noise each round
- 🎯 **Attacker Isolation** — Malicious client's trust decays 76% over 10 rounds with 103.4%
  increase in injected noise
- 🧩 **Multi-Component Target Formula** — Proposed formula incorporating Contribution, Performance,
  Stability, and Data Deviation signals
- 📊 **Non-IID Robustness** — Validated under realistic label-skew data partitioning across 10 clients

---

## 🏗️ System Architecture

The system comprises five principal components operating within the Flower (FLWR) simulation
framework:
┌─────────────────────────────────────────────────────────────┐
│                      FEDERATED SERVER                       │
│                                                             │
│   ┌──────────────┐   ┌──────────────┐   ┌───────────────┐  │
│   │    TRUST     │   │   PRIVACY    │   │  AGGREGATION  │  │
│   │   MANAGER   │──▶│    MODULE    │──▶│    ENGINE     │  │
│   │             │   │              │   │               │  │
│   │  T_i update │   │  σ = √(K/T) │   │  θ = ΣTθ̃/ΣT │  │
│   └──────────────┘   └──────────────┘   └───────────────┘  │
└────────────────────────────┬────────────────────────────────┘
│ Broadcast / Collect
┌──────────────────┼──────────────────┐
▼                  ▼                  ▼
┌──────────┐       ┌──────────┐       ┌──────────┐
│ Client 0 │  ...  │ Client 8 │  ...  │ Client 9 │
│ (Normal) │       │(Attacker)│       │ (Normal) │
└──────────┘       └──────────┘       └──────────┘

**Communication Round Protocol:**
1. Server broadcasts current global parameters to all N clients
2. Each client trains locally for E epochs, returns (θᵢ, Pᵢ)
3. Trust Manager updates Tᵢ using the trust formula
4. Privacy Module injects noise scaled by σᵢ = √(K / (Tᵢ + ε))
5. Aggregation Engine computes trust-weighted global model

> 📐 *Architecture diagram placeholder — add `docs/architecture.png` to your repo*

---

## 🔬 Methodology

> **Note:** Full algorithmic details, implementation specifics, and the complete target formula
> are withheld pending patent filing. This section describes the methodology at a
> conceptual level sufficient for academic understanding.

### Intermediate Trust Update Rule *(Implemented)*

Each client's trust score is updated per round using a momentum-based formulation that
combines historical trust with current local performance. Trust is initialised at 0.5 and
clamped to [0, 1].

- **λ = 0.8** — momentum coefficient, retains historical trust
- **Pᵢ⁽ᵗ⁾** — local validation accuracy of client i at round t

### Target Trust Formula *(Proposed — Details Patent Pending)*

The full formula enriches the trust signal with four independent components:

| Component | Symbol | What it Captures |
|---|---|---|
| Contribution Score | C | Gradient alignment with consensus direction (cosine similarity) |
| Performance Score | P | Local validation accuracy on held-out split |
| Stability Score | S | Consistency of trust across recent rounds |
| Data Deviation Penalty | D | KL-divergence between local and global label distributions |

### Adaptive Privacy Noise Injection

Noise standard deviation σ is calibrated **inversely** with trust — a deliberate design
choice that simultaneously improves robustness and optimises the privacy-utility trade-off
at the per-client level.

### Trust-Weighted Aggregation

The global model is computed as the trust-weighted average of noise-augmented client
parameters, meaning a client with near-zero trust contributes negligibly regardless of
the magnitude of its corrupted updates.

---

## ⚙️ Experimental Setup

| Configuration | Value |
|---|---|
| Dataset | MNIST (60K train / 10K test) |
| Partitioning | Non-IID shard (sorted by class label) |
| Clients | 10 total (1 malicious — Client 8) |
| Rounds | 10 communication rounds |
| Attacker Model | Random noise parameters + fake 5% accuracy report |
| Framework | Flower (FLWR) v1.8 + Ray backend |
| Hardware | Dual NVIDIA Tesla T4 GPU (16 GB VRAM) — Kaggle Notebooks |
| Optimiser | Adam, LR = 1e-3 |
| Local Epochs | 2 per round |
| Batch Size | 128 |
| λ (momentum) | 0.8 |
| K_noise | 0.05 |
| ε (stability) | 1e-3 |

**Model Architecture — MnistCNN:**

Conv2d(1→32) → BatchNorm → ReLU → MaxPool
Conv2d(32→64) → BatchNorm → ReLU → MaxPool
Flatten → Linear(3136→128) → ReLU → Dropout(0.25) → Linear(128→10)
---

## 📊 Results & Observations

### Trust Score Evolution

> 📈 *Place `results/fig_trust_evolution.png` here*

The malicious client (C8) experiences a monotonically decreasing trust trajectory across
all 10 rounds. Benign clients consistently converge to trust > 0.94.

| Round | Avg. Benign Trust | Attacker Trust | Attacker σ (Noise) |
|:---:|:---:|:---:|:---:|
| 1 | 0.6000 | 0.4100 | 0.3488 |
| 2 | 0.6800 | 0.3380 | 0.3840 |
| 3 | 0.7440 | 0.2804 | 0.4215 |
| 4 | 0.7952 | 0.2343 | 0.4610 |
| 5 | 0.8362 | 0.1975 | 0.5019 |
| 6 | 0.8689 | 0.1680 | 0.5440 |
| 7 | 0.8951 | 0.1444 | 0.5865 |
| 8 | 0.9161 | 0.1255 | 0.6287 |
| 9 | 0.9329 | 0.1104 | 0.6700 |
| **10** | **0.9463** | **0.0983** | **0.7095** |

### Adaptive Noise Injection

> 📈 *Place `results/fig_noise_sigma.png` here*

As benign clients build trust, their noise σ decreases from ~0.29 → ~0.23.
The attacker's noise σ increases from 0.35 → 0.71 (+103.4%).

### System Comparison

| Feature | Baseline (FedAvg) | Intermediate | Target (Proposed) |
|---|:---:|:---:|:---:|
| Trust Formula | None | λT+(1-λ)P | Multi-component *(patent pending)* |
| Adaptive Noise | ✗ | ✓ | ✓ |
| Data Setting | IID | Non-IID | Non-IID |
| Malicious Clients | None | 1 (isolated) | Generalised |
| Accuracy | 98.52% (R1) | ~94% (R10) | ~97% (projected) |
| Attacker Trust (Final) | N/A | **0.0983** | **~0.05** |

> 📊 *Place `results/fig_comparison.png` and `results/fig_final_trusts.png` here*

### Key Takeaways

- ✅ **76% trust reduction** for the attacker over 10 rounds
- ✅ **103.4% increase** in attacker noise — neutralising corrupted updates
- ✅ **57.7% increase** in average benign client trust
- ✅ Global model accuracy maintained at **~94%** under active adversarial conditions
- ✅ No false positives — all 9 benign clients retained trust above threshold

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.12 |
| Deep Learning | PyTorch 2.x |
| Federated Learning | Flower (FLWR) v1.8 |
| Distributed Backend | Ray |
| Dataset | MNIST (torchvision) |
| Visualisation | Matplotlib, NumPy |
| Hardware | NVIDIA Tesla T4 (Kaggle Notebooks) |
| Privacy Mechanism | Adaptive Gaussian Noise (custom DP module) |

---


# Install dependencies
pip install -r requirements.txt
```

**`requirements.txt`**
