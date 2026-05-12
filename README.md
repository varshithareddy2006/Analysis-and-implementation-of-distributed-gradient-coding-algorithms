# Analysis-and-implementation-of-distributed-gradient-coding-algorithms
# Hierarchical Distributed Gradient Coding using Neural Networks

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [System Architecture](#system-architecture)
- [Mathematical Formulation](#mathematical-formulation)
- [Methodology](#methodology)
- [Neural Network: ZPredictor](#neural-network-zpredictor)
- [Iterative Convergence: r and β](#iterative-convergence-r-and-β)
- [Architecture Optimization](#architecture-optimization)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [References](#references)

---

# Overview

This project analyzes and implements distributed gradient coding algorithms for hierarchical computing systems.

The core challenge in large-scale distributed machine learning is the **straggler problem** — some worker nodes are significantly slower than others, causing bottlenecks.

This project models a **two-level hierarchical system**:

**Master → Edge Nodes → Workers**

It derives the statistical distribution of iteration time and uses a neural network to efficiently find the optimal architecture that minimizes expected iteration time.

---

# Problem Statement

Given:

- A total of `N` machines arranged in a 2-level hierarchy
- Straggler fraction `α`
- Computation rate `λ₁`
- Communication rate `λ₂`
- Shift constants `a` and `b`

Goal:

Find the optimal split:

- `n1*` → number of edge nodes
- `n2*` → workers per group

that minimizes the expected iteration time:

\[
E[T_{iter}]
\]

Subject to:

\[
N = n_1 \times (n_2 + 1)
\]

---

# System Architecture

```text
                        [ Master / Root ]
                              |
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
      [Edge Node 1]       [Edge Node 2]  ...  [Edge Node n1]
          |                   |
     ┌────┼────┐         ┌────┼────┐
     ▼    ▼    ▼         ▼    ▼    ▼
   [W1] [W2] [W3]      [W1] [W2] [W3]
Level 0: Master node — collects final aggregated gradientsLevel 1 (Edge Nodes): n1 edge nodes — each aggregates gradients from its group of workersLevel 2 (Workers): n2 workers per edge node — each computes gradients over a data partition

Straggler Tolerance:

Level 2 waits for the k2 = n2 − ⌊α·n2⌋ fastest workersLevel 1 waits for the k1 = n1 − ⌊α·n1⌋ fastest edge nodes

Mathematical FormulationComputation and Communication TimesBoth times follow shifted exponential distributions (based on the CodedReduce framework):Computation time for each worker:

Tcomp=a⋅r⋅D+Exp ⁣(λ1r⋅D),t≥a⋅r⋅DT_{comp} = a \cdot r \cdot D + \text{Exp}!\left(\frac{\lambda_1}{r \cdot D}\right), \quad t \geq a \cdot r \cdot DTcomp=a⋅r⋅D+Exp(r⋅Dλ1),t≥a⋅r⋅DCommunication time for each worker:

Tcomm=b⋅p+Exp ⁣(λ2p),t≥b⋅pT_{comm} = b \cdot p + \text{Exp}!\left(\frac{\lambda_2}{p}\right), \quad t \geq b \cdot pTcomm=b⋅p+Exp(pλ2),t≥b⋅pwhere:

D = 60,000 (dataset size, e.g., MNIST training samples)p = 784 (parameter/gradient vector dimension)r = computation load factor per workera, b = shift constants

Group Finish Time (Z)The total time per worker W = T_comp + T_comp, which follows a Hypoexponential distribution. The Group Finish Time at each edge node is the k2-th order statistic of n2 such workers:fZ(t)=n!(k−1)!(n−k)!⋅[FW(t)]k−1⋅[1−FW(t)]n−k⋅fW(t)f_Z(t) = \frac{n!}{(k-1)!(n-k)!} \cdot [F_W(t)]^{k-1} \cdot [1 - F_W(t)]^{n-k} \cdot f_W(t)fZ(t)=(k−1)!(n−k)!n!⋅[FW(t)]k−1⋅[1−FW(t)]n−k⋅fW(t)Shifted Exponential ApproximationZ is approximated as:Z∼cfit+Exp(μfit)Z \sim c_{\text{fit}} + \text{Exp}(\mu_{\text{fit}})Z∼cfit+Exp(μfit)Parameters c_fit and μ_fit are estimated using the Method of Moments (matching mean and variance of Z).Computation Load Factorr=1(ns+1)2+(ns+1)r = \frac{1}{\left(\frac{n}{s+1}\right)^2 + \left(\frac{n}{s+1}\right)}r=(s+1n)2+(s+1n)1where n = number of workers, s = number of stragglers.

MethodologyThe solution pipeline consists of four stages:Stage 1: Parameter Calibration↓  Adjust λ1, λ2, a, b to match real-world timingStage 2: Dataset Generation↓  Vary (n2, a, α, λ1, r) → compute (c_fit, μ_fit)Stage 3: ZPredictor Neural Network↓  Train MLP: (n2, a, α, λ1, r) → (c_fit, μ_fit)Stage 4: Iterative r & β Convergence↓  Use ZPredictor in a fixed-point loopStage 5: Architecture Search↓  Enumerate all valid (n1, n2) for given N → find argmin E[T_iter]

Neural Network: ZPredictorComputing c_fit and μ_fit analytically for every possible (n1, n2) configuration is computationally expensive. A fully connected feedforward neural network (MLP) is trained to predict these parameters instantly.ArchitectureInput (5) → Linear(64) → BN → ReLU → Dropout(0.2)→ Linear(128) → BN → ReLU → Dropout(0.2)→ Linear(128) → BN → ReLU → Dropout(0.2)→ Linear(64)  → BN → ReLU → Dropout(0.2)→ Output (2)ComponentDetailInput featuresn2, a, α, λ1, rOutputc_fit, μ_fitHidden layers4 layers: 64 → 128 → 128 → 64RegularizationBatch Normalization + Dropout (p=0.2)Loss functionMean Squared Error (MSE)OptimizerAdamTraining data10,000 samplesTest R² Score0.97 – 0.99 on both outputsTraining CurveThe model shows rapid initial convergence with validation loss stabilizing well below training loss, indicating good generalization without overfitting.

Iterative Convergence: r and βIn a hierarchical system, computation loads differ across levels:

Level 2 workers operate with base load rLevel 1 edge nodes have effective load r1 = r + β, where β ≥ 0 captures extra aggregation work

Since r and β are mutually dependent, a fixed-point iteration is used:AlgorithmInitialize rRepeat until convergence:1. ZPredictor(n2, a, α, λ1, r) → (c_fit, μ_fit)2. Update β via mean-matching formula:β = [c_fit + (1/μ_fit)] / [D · (a + 1/λ1)] − r3. Update r via system load balance equationUntil |Δr| < ε and |Δβ| < ε

Convergence speed: Typically 5–7 iterations (updated model; earlier implementation took 7–10)Stability: Converges to a unique fixed point (r*, β*) for all valid (n1, n2) configurations

Converged quantities:

r* → effective computation load for Level 2 workersβ* → extra computation load at the edge node levelr* + β* → total effective load at Level 1

Architecture OptimizationOnce (r*, β*) are found, the expected iteration time E[T_iter] is computed analytically for a given (n1, n2). The optimizer:

Enumerates all valid (n1, n2) pairs satisfying N = n1 × (n2 + 1)Runs the ZPredictor + convergence loop for each pairComputes E[T_iter]Returns (n1*, n2*) with minimum expected iteration time
