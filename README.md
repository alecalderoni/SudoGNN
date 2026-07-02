# SudoGNN 🎨🕸️

⚠️ **WORK IN PROGRESS**: This project explores the **Graph Coloring Problem** through the lens of **Graph Neural Networks (GNNs)**, driven by an **unsupervised, physics-inspired loss function**. While the ultimate fun benchmark and application is solving **Sudoku**, the core architecture is designed around solving general graph coloring challenges.

> *Note: We sincerely apologize that the initial accompanying notebook/code references contains comments and metadata in Italian.*

---

## 💡 The Core Idea: Graph Coloring via Graph Neural Network

The **Graph Coloring Problem** is a fundamental NP-complete challenge in computer science: how do you assign a "color" (or a state label) to every node in a complex network such that no two connected nodes share the same color? 

Instead of traditional algorithmic backtracking or combinatorial search, **SudoGNN** treats graph coloring as an energy-minimization problem within a dynamical system.

---

## 📉 The Objective Function: Physics Loss & Entropy Regularization

SudoGNN operates applying gradient descent directly to a composite loss function that enforces problem constraints and guides probability convergence.

The total loss is minimized when the network settles into a stable, valid coloring configuration (a zero-energy ground state).

### 1. The Physics-Inspired Constraint Loss (Adjacency Penalty)
Each node $i$ outputs a softmax probability vector $\mathbf{p}_i \in [0, 1]^C$, where $C$ is the number of available colors (e.g., 9 for Sudoku) and $\sum_c p_{ic} = 1$.

To enforce that adjacent nodes do not share the same color, we compute a **dot-product friction/repulsion penalty** across all connected edges $(i, j) \in E$:

$$\mathcal{L}_{\text{physics}} = \sum_{(i,j) \in E} \mathbf{p}_i \cdot \mathbf{p}_j = \sum_{(i,j) \in E} \sum_{c=1}^C p_{ic} \cdot p_{jc}$$

- **Why it works**: If node $i$ and node $j$ are connected and both assign a high probability to color $3$, the product $p_{i,3} \cdot p_{j,3}$ will be large, penalizing the network.
- The loss reaches its global minimum ($0$) if and only if for every edge, the selected colors have orthogonal probability vectors.

### 2. Shannon Entropy Regularization (Sharpening Penalty)
A known failure mode of purely optimizing $\mathcal{L}_{\text{physics}}$ is that the GNN can cheat by assigning a uniform distribution ($\frac{1}{C}$) to every node. While this lowers the dot-product penalty, it represents an ambiguous, unsolved state.

To force the network to make "hard" decisions and converge to a single color per node, we introduce a negative **Shannon Entropy** regularization term:

$$\mathcal{L}_{\text{entropy}} = -\gamma \sum_{i \in V} H(\mathbf{p}_i) = \gamma \sum_{i \in V} \sum_{c=1}^C p_{ic} \log(p_{ic})$$

- **The Goal**: High entropy means uncertainty (uniform distribution). By minimizing the entropy (or maximizing negative entropy), we push the probability vectors toward a **one-hot encoding** (where one color has a probability of $1.0$ and all others are $0.0$).
- $\gamma$ acts as a thermodynamic temperature scaling hyperparameter, controlling the speed of crystallization/convergence.

---

## 🧩 The Benchmark Application: Sudoku

To put this constraint engine to a fun, highly structured test, we map the game of **Sudoku** into the framework:
- **Topology**: A 9x9 board is translated into a graph of 81 nodes.
- **Constraints**: Edges are drawn between cells sharing a row, column, or 3x3 grid block.
- **Colors**: The digits 1 through 9 serve as the 9 distinct colors.

Solving the Sudoku is simply the emergent result of the GNN successfully minimizing the physical constraint and entropy loss simultaneously.

---

## 🛠️ Foundations & Tech Stack

The project builds upon foundational graph coloring algorithms initialized in `GNN.ipynb` (originally by *Alessandro Calderoni*) and scales them toward complex constraint matrices using:
- **PyTorch** & **PyTorch Geometric (PyG)** for graph convolutions (`SAGEConv`).
- **NetworkX** & **Matplotlib** for parsing and visualizing DIMACS standard (`.col`) graphs.

---

## 🚀 Status

This repository is under active development. We are currently benchmarking the unsupervised physics loss on standard complex graphs and fine-tuning the balance between the adjacency penalty and entropy minimization to guarantee convergence without traditional backtracking.
