# Analysis-and-implementation-of-distributed-gradient-coding-algorithms
Table of Contents

Overview
Problem Statement
System Architecture
Mathematical Formulation
Methodology
Neural Network: ZPredictor
Iterative Convergence: r and β
Architecture Optimization
Results
Project Structure
Setup & Installation
Usage
Dependencies
References


Overview
This project analyzes and implements distributed gradient coding algorithms for hierarchical computing systems. The core challenge in large-scale distributed machine learning is the straggler problem — some worker nodes are significantly slower than others, causing bottlenecks.
This project models a two-level hierarchical system (Master → Edge Nodes → Workers), derives the statistical distribution of iteration time, and uses a neural network to efficiently find the optimal architecture that minimizes expected iteration time.

Problem Statement
Given:

A total of N machines to be arranged in a 2-level hierarchy
Straggler fraction α (fraction of slow machines tolerated at each level)
System parameters: computation rate λ₁, communication rate λ₂, shift constants a, b

Goal: Find the optimal split (n1*, n2*) — number of edge nodes and workers per group — that minimizes the expected iteration time E[T_iter].
The total machine count satisfies: N = n1 × (n2 + 1)

System Architecture
                        [ Master / Root ]
                              |
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
      [Edge Node 1]       [Edge Node 2]  ...  [Edge Node n1]   ← Level 1
          |                   |
     ┌────┼────┐         ┌────┼────┐
     ▼    ▼    ▼         ▼    ▼    ▼
   [W1] [W2] [W3]      [W1] [W2] [W3]  ...                     ← Level 2

Level 0: Master node — collects final aggregated gradients
Level 1 (Edge Nodes): n1 edge nodes — each aggregates gradients from its group of workers
Level 2 (Workers): n2 workers per edge node — each computes gradients over a data partition

Straggler Tolerance:

Level 2 waits for the k2 = n2 − ⌊α·n2⌋ fastest workers
Level 1 waits for the k1 = n1 − ⌊α·n1⌋ fastest edge nodes


Mathematical Formulation
Computation and Communication Times
Both times follow shifted exponential distributions (based on the CodedReduce framework):
Computation time for each worker:

Tcomp=a⋅r⋅D+Exp ⁣(λ1r⋅D),t≥a⋅r⋅DT_{comp} = a \cdot r \cdot D + \text{Exp}\!\left(\frac{\lambda_1}{r \cdot D}\right), \quad t \geq a \cdot r \cdot DTcomp​=a⋅r⋅D+Exp(r⋅Dλ1​​),t≥a⋅r⋅D
Communication time for each worker:

Tcomm=b⋅p+Exp ⁣(λ2p),t≥b⋅pT_{comm} = b \cdot p + \text{Exp}\!\left(\frac{\lambda_2}{p}\right), \quad t \geq b \cdot pTcomm​=b⋅p+Exp(pλ2​​),t≥b⋅p
where:

D = 60,000 (dataset size, e.g., MNIST training samples)
p = 784 (parameter/gradient vector dimension)
r = computation load factor per worker
a, b = shift constants

Group Finish Time (Z)
The total time per worker W = T_comp + T_comp, which follows a Hypoexponential distribution. The Group Finish Time at each edge node is the k2-th order statistic of n2 such workers:
fZ(t)=n!(k−1)!(n−k)!⋅[FW(t)]k−1⋅[1−FW(t)]n−k⋅fW(t)f_Z(t) = \frac{n!}{(k-1)!(n-k)!} \cdot [F_W(t)]^{k-1} \cdot [1 - F_W(t)]^{n-k} \cdot f_W(t)fZ​(t)=(k−1)!(n−k)!n!​⋅[FW​(t)]k−1⋅[1−FW​(t)]n−k⋅fW​(t)
Shifted Exponential Approximation
Z is approximated as:
Z∼cfit+Exp(μfit)Z \sim c_{\text{fit}} + \text{Exp}(\mu_{\text{fit}})Z∼cfit​+Exp(μfit​)
Parameters c_fit and μ_fit are estimated using the Method of Moments (matching mean and variance of Z).
Computation Load Factor
r=1(ns+1)2+(ns+1)r = \frac{1}{\left(\frac{n}{s+1}\right)^2 + \left(\frac{n}{s+1}\right)}r=(s+1n​)2+(s+1n​)1​
where n = number of workers, s = number of stragglers.

Methodology
The solution pipeline consists of four stages:
Stage 1: Parameter Calibration
    ↓  Adjust λ1, λ2, a, b to match real-world timing
Stage 2: Dataset Generation
    ↓  Vary (n2, a, α, λ1, r) → compute (c_fit, μ_fit)
Stage 3: ZPredictor Neural Network
    ↓  Train MLP: (n2, a, α, λ1, r) → (c_fit, μ_fit)
Stage 4: Iterative r & β Convergence
    ↓  Use ZPredictor in a fixed-point loop
Stage 5: Architecture Search
    ↓  Enumerate all valid (n1, n2) for given N → find argmin E[T_iter]

Neural Network: ZPredictor
Computing c_fit and μ_fit analytically for every possible (n1, n2) configuration is computationally expensive. A fully connected feedforward neural network (MLP) is trained to predict these parameters instantly.
Architecture
Input (5) → Linear(64) → BN → ReLU → Dropout(0.2)
          → Linear(128) → BN → ReLU → Dropout(0.2)
          → Linear(128) → BN → ReLU → Dropout(0.2)
          → Linear(64)  → BN → ReLU → Dropout(0.2)
          → Output (2)
ComponentDetailInput featuresn2, a, α, λ1, rOutputc_fit, μ_fitHidden layers4 layers: 64 → 128 → 128 → 64RegularizationBatch Normalization + Dropout (p=0.2)Loss functionMean Squared Error (MSE)OptimizerAdamTraining data10,000 samplesTest R² Score0.97 – 0.99 on both outputs
Training Curve
The model shows rapid initial convergence with validation loss stabilizing well below training loss, indicating good generalization without overfitting.

Iterative Convergence: r and β
In a hierarchical system, computation loads differ across levels:

Level 2 workers operate with base load r
Level 1 edge nodes have effective load r1 = r + β, where β ≥ 0 captures extra aggregation work

Since r and β are mutually dependent, a fixed-point iteration is used:
Algorithm
Initialize r
Repeat until convergence:
    1. ZPredictor(n2, a, α, λ1, r) → (c_fit, μ_fit)
    2. Update β via mean-matching formula:
            β = [c_fit + (1/μ_fit)] / [D · (a + 1/λ1)] − r
    3. Update r via system load balance equation
Until |Δr| < ε and |Δβ| < ε

Convergence speed: Typically 5–7 iterations (updated model; earlier implementation took 7–10)
Stability: Converges to a unique fixed point (r*, β*) for all valid (n1, n2) configurations

Converged quantities:

r* → effective computation load for Level 2 workers
β* → extra computation load at the edge node level
r* + β* → total effective load at Level 1


Architecture Optimization
Once (r*, β*) are found, the expected iteration time E[T_iter] is computed analytically for a given (n1, n2). The optimizer:

Enumerates all valid (n1, n2) pairs satisfying N = n1 × (n2 + 1)
Runs the ZPredictor + convergence loop for each pair
Computes E[T_iter]
Returns (n1*, n2*) with minimum expected iteration time
