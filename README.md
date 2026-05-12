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
