# Analysis-and-implementation-of-distributed-gradient-coding-algorithms

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
```

---

## System Architecture

The proposed distributed learning framework follows a **three-level hierarchical structure**:

- **Level 0 (Master Node):**  
  Collects and aggregates the final gradients.

- **Level 1 (Edge Nodes):**  
  Contains `n1` edge nodes, where each edge node aggregates gradients from a group of workers.

- **Level 2 (Worker Nodes):**  
  Each edge node contains `n2` workers that compute gradients over assigned data partitions.

---

# Straggler Tolerance

To mitigate the effect of slow or failed workers (stragglers), coded redundancy is introduced.

### Level 2 (Workers)

Each edge node waits only for the fastest:

\[
k_2 = n_2 - \lfloor \alpha n_2 \rfloor
\]

workers.

### Level 1 (Edge Nodes)

The master node waits only for the fastest:

\[
k_1 = n_1 - \lfloor \alpha n_1 \rfloor
\]

edge nodes.

where:

- \( \alpha \in [0,1) \) is the straggler fraction.
- \( \lfloor \cdot \rfloor \) denotes the floor function.

---

# Mathematical Formulation

## Computation and Communication Time Models

Both computation and communication delays follow **shifted exponential distributions**, inspired by the CodedReduce framework.

---

## Worker Computation Time

\[
T_{\text{comp}} = a \cdot r \cdot D + \text{Exp}\left(\frac{\lambda_1}{rD}\right),
\quad t \ge a r D
\]

where:

- \( D = 60000 \) : dataset size
- \( r \) : computation load factor
- \( a \) : deterministic shift constant
- \( \lambda_1 \) : computation rate parameter

---

## Worker Communication Time

\[
T_{\text{comm}} = b \cdot p + \text{Exp}\left(\frac{\lambda_2}{p}\right),
\quad t \ge bp
\]

where:

- \( p = 784 \) : gradient vector dimension
- \( b \) : communication shift constant
- \( \lambda_2 \) : communication rate parameter

---

# Total Worker Time

The total worker completion time is:

\[
W = T_{\text{comp}} + T_{\text{comm}}
\]

Since \(W\) is the sum of two shifted exponentials, it follows a **Hypoexponential distribution**.

---

# Group Finish Time

For each edge node, the completion time is determined by the
\(k_2\)-th fastest worker among \(n_2\) workers.

The corresponding order statistic distribution is:

\[
f_Z(t)=
\frac{n!}{(k-1)!(n-k)!}
[F_W(t)]^{k-1}
[1-F_W(t)]^{n-k}
f_W(t)
\]

where:

- \(F_W(t)\) : CDF of worker completion time
- \(f_W(t)\) : PDF of worker completion time

---

# Shifted Exponential Approximation

The order statistic \(Z\) is approximated using another shifted exponential:

\[
Z \sim c_{\text{fit}} + \text{Exp}(\mu_{\text{fit}})
\]

The parameters:

- \(c_{\text{fit}}\)
- \(\mu_{\text{fit}}\)

are estimated using the **Method of Moments** by matching the mean and variance of \(Z\).

---

# Computation Load Factor

The computation load factor is defined as:

\[
r =
\frac{1}{
\left(\frac{n}{s+1}\right)^2
+
\left(\frac{n}{s+1}\right)
}
\]

where:

- \(n\) : total workers
- \(s\) : number of tolerated stragglers

---

# Methodology

The complete solution pipeline consists of five stages.

---

## Stage 1: Parameter Calibration

- Calibrate:
  - \( \lambda_1 \)
  - \( \lambda_2 \)
  - \( a \)
  - \( b \)

using realistic computation and communication timings.

---

## Stage 2: Dataset Generation

Generate training samples by varying:

\[
(n_2, a, \alpha, \lambda_1, r)
\]

and computing:

\[
(c_{\text{fit}}, \mu_{\text{fit}})
\]

using analytical moment matching.

---

## Stage 3: ZPredictor Neural Network

A Multi-Layer Perceptron (MLP) is trained to instantly predict:

\[
(n_2, a, \alpha, \lambda_1, r)
\rightarrow
(c_{\text{fit}}, \mu_{\text{fit}})
\]

This eliminates expensive repeated analytical computations.

---

# Neural Network Architecture

```text
Input(5)
   ↓
Linear(64) → BatchNorm → ReLU → Dropout(0.2)
   ↓
Linear(128) → BatchNorm → ReLU → Dropout(0.2)
   ↓
Linear(128) → BatchNorm → ReLU → Dropout(0.2)
   ↓
Linear(64) → BatchNorm → ReLU → Dropout(0.2)
   ↓
Output(2)
```

| Component | Details |
|---|---|
| Input Features | \(n_2, a, \alpha, \lambda_1, r\) |
| Outputs | \(c_{\text{fit}}, \mu_{\text{fit}}\) |
| Hidden Layers | 64 → 128 → 128 → 64 |
| Activation | ReLU |
| Regularization | BatchNorm + Dropout(0.2) |
| Loss Function | Mean Squared Error (MSE) |
| Optimizer | Adam |
| Training Samples | 10,000 |
| Test \(R^2\) Score | 0.97 – 0.99 |

---

# Training Characteristics

- Rapid convergence during initial epochs
- Validation loss stabilizes below training loss
- Minimal overfitting
- Strong generalization across unseen configurations

---

# Iterative Convergence of \(r\) and \(\beta\)

The hierarchical structure introduces different loads at different levels.

- Level 2 workers operate with load:

\[
r
\]

- Level 1 edge nodes operate with effective load:

\[
r_1 = r + \beta
\]

where:

- \(\beta \ge 0\) represents additional aggregation overhead.

Since \(r\) and \(\beta\) are interdependent, a fixed-point iteration is used.

---

# Fixed-Point Iteration Algorithm

```text
Initialize r

Repeat until convergence:

1. ZPredictor(n2, a, α, λ1, r)
      → (c_fit, μ_fit)

2. Update β:

   β =
   [c_fit + (1/μ_fit)]
   -------------------
   D(a + 1/λ1)
   − r

3. Update r using
   the system load balance equation

Until:
|Δr| < ε
and
|Δβ| < ε
```

---

# Convergence Properties

- Typical convergence:
  - 5–7 iterations (updated implementation)
- Earlier implementation:
  - 7–10 iterations
- Converges to a unique fixed point for all valid configurations.

Final converged quantities:

- \(r^*\) : effective worker computation load
- \(\beta^*\) : edge-node aggregation overhead
- \(r^* + \beta^*\) : total Level-1 computation load

---

# Architecture Optimization

After obtaining converged values \((r^*, \beta^*)\), the expected iteration time:

\[
E[T_{\text{iter}}]
\]

is computed analytically.

The optimizer performs:

1. Enumeration of all valid \((n_1, n_2)\) pairs satisfying:

\[
N = n_1 (n_2 + 1)
\]

2. Fixed-point convergence using the trained ZPredictor

3. Computation of expected iteration time

4. Selection of the optimal hierarchy:

\[
(n_1^*, n_2^*)
=
\arg\min E[T_{\text{iter}}]
\]

---

# Key Contributions

- Hierarchical coded distributed learning framework
- Straggler-resilient gradient aggregation
- Shifted exponential approximation for order statistics
- Neural network based latency prediction
- Fixed-point convergence framework for load balancing
- Analytical architecture optimization for minimum iteration time
