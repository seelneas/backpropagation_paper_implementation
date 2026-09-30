# Learning Representations by Back-Propagating Errors — Paper Implementation

A from-scratch NumPy implementation of the foundational 1986 paper **"Learning representations by back-propagating errors"** by David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams, published in *Nature*.

This project faithfully reproduces the core back-propagation algorithm and the key experiments described in the paper — without relying on any high-level deep learning frameworks.

---

## 📄 About the Paper

The paper introduces the **back-propagation learning procedure** for multilayer networks of neuron-like units. The key insight is that by propagating error signals backward through a network, hidden units can discover useful **internal representations** (features) that are not explicitly present in the input, enabling the network to solve problems that single-layer perceptrons cannot.

> **Citation:** Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533–536.

A copy of the original paper is included in the [`paper/`](paper/) directory.

---

## 🎯 Objectives

- Understand the motivation behind back-propagation and why hidden units are necessary.
- Implement **forward propagation**, **backward propagation of errors** (via the chain rule), and **gradient-descent weight updates** entirely from scratch using NumPy.
- Reproduce selected experiments from the original paper.
- Explore and visualize the **internal representations** learned by hidden units.
- Compare implementation results with those reported by the authors.

---

## 📂 Project Structure

```
.
├── README.md
├── LICENSE                              # MIT License
├── requirements.txt                     # Python dependencies
├── paper/
│   └── first paper.pdf                  # Original 1986 Nature paper
└── notebooks/
    ├── backpropagation.ipynb             # Core algorithm implementation
    ├── mirror_symmetry.ipynb             # Mirror symmetry experiment
    └── family_tree_paper_experiment.ipynb # Family tree experiment
```

---

## 🧪 Experiments

### 1. Core Back-Propagation Algorithm ([`backpropagation.ipynb`](notebooks/backpropagation.ipynb))

The foundational notebook that builds the back-propagation algorithm step by step from first principles.

**What it covers:**
- **Sigmoid activation function** and its derivative — the nonlinearity described in the paper.
- **Network initialization** — a 2-layer network (input → hidden → output) with small random weights.
- **Forward propagation** — computing activations layer by layer via matrix multiplication and the sigmoid function.
- **Error computation** — Mean Squared Error (MSE) loss between predicted and target outputs.
- **Backward propagation** — computing error gradients for each layer using the chain rule.
- **Weight updates** — applying vanilla gradient descent with a configurable learning rate.

**Demonstration task:** The classic **XOR problem** — a non-linearly separable function that requires hidden units to solve.

| Input | Target | Predicted (after training) |
|-------|--------|---------------------------|
| [0, 0] | 0 | ≈ 0.017 |
| [0, 1] | 1 | ≈ 0.982 |
| [1, 0] | 1 | ≈ 0.983 |
| [1, 1] | 0 | ≈ 0.014 |

**Hyperparameters:** 10,000 epochs, learning rate = 0.5, 3 hidden units.

The notebook also includes a **training loss curve** visualization showing convergence.

---

### 2. Mirror Symmetry Experiment ([`mirror_symmetry.ipynb`](notebooks/mirror_symmetry.ipynb))

Reproduces one of the paper's key experiments — detecting **mirror symmetry** in 6-bit binary patterns.

**Task:** Given a 6-bit binary input `[b1, b2, b3, b4, b5, b6]`, determine whether the pattern is mirror-symmetric, i.e., whether `[b1, b2, b3] == [b6, b5, b4]`.

**Dataset:**
- All 2⁶ = **64** possible 6-bit patterns.
- **8** symmetric patterns (positive class), **56** non-symmetric (negative class).

**Architecture:**
- Input: 6 units → Hidden: **2 units** → Output: 1 unit
- This is a deliberately constrained bottleneck — the paper emphasizes that with only 2 hidden units, the network must discover compact, distributed representations of the input.

**Results:**
- Final training loss: ~0.087 (MSE)
- Accuracy: **89.06%** on the full dataset
- The notebook provides a detailed per-pattern breakdown showing actual labels, predicted classes, and output probabilities.

**Key insight from the paper:** The network struggles with this task when the hidden layer is too small, demonstrating the importance of representational capacity for learning abstract features.

---

### 3. Family Tree Experiment ([`family_tree_paper_experiment.ipynb`](notebooks/family_tree_paper_experiment.ipynb))

The most complex experiment — a faithful reproduction of the paper's family tree relational learning task.

**Task:** Given a person and a relationship, predict the related person(s).  
Example: `(Colin, aunt) → Margaret, Jennifer`

**Knowledge Base:**
- **24 people** across two isomorphic family trees (12 English, 12 Italian).
- **12 relationship types:** father, mother, husband, wife, son, daughter, uncle, aunt, brother, sister, nephew, niece.
- **104 valid facts** (person₁, relationship, person₂) generated from the family structure.
- **100 training facts**, **4 held-out** for testing.

**Architecture** (five-layer network, as described in the paper):

```
Person₁ (24-unit one-hot) → 6-unit bottleneck ──┐
                                                  ├→ 12-unit central hidden → 6-unit penultimate → 24-unit output
Relationship (12-unit one-hot) → 6-unit bottleneck ┘
```

The 24-unit input and output representations are **localist (one-hot)**. The person and relationship inputs are compressed through learned **6-unit distributed representations** (bottleneck layers). The output is a 24-unit multi-label vector where multiple people can be activated simultaneously.

**Training Schedule** (following the paper):
- Sweeps 1–20: ε = 0.005, α (momentum) = 0.5
- Sweeps 21–1500: ε = 0.01, α = 0.9
- **0.2% weight decay** after each update

**Evaluation Criterion** (as described in the paper):
- Output units that should be ON must have activity > **0.8**
- Output units that should be OFF must have activity < **0.2**
- A query is correct only if **all 24 output units** satisfy the criterion.

**What the notebook includes:**
1. Full reconstruction of both family trees from the paper's Figure 2.
2. Automatic generation of all 104 valid relational facts.
3. A hand-coded five-layer neural network implemented purely in NumPy — no frameworks.
4. Full backpropagation with momentum and weight decay.
5. Training for 1,500 sweeps with the paper's learning rate schedule.
6. Training loss curve and accuracy plots.
7. Evaluation on the 4 held-out test cases.
8. Inspection of multi-answer queries (e.g., `Colin + aunt`).
9. Visualization of the **learned person representations** — the 6-dimensional bottleneck activations that the network discovers to encode latent properties such as nationality, generation, and family branch.

**Reproducibility note:** The paper does not specify the exact 4 held-out triples or the initial weight matrices. This implementation uses a fixed, documented split and a reproducible random seed.

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/seelneas/backpropagation_paper_implementation.git
cd backpropagation_paper_implementation

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Matrix operations, core algorithm implementation |
| `matplotlib` | Training loss curves and representation visualizations |
| `jupyter` | Interactive notebook environment |

### Running the Notebooks

```bash
jupyter notebook
```

Then open any notebook from the `notebooks/` directory in your browser.

---

## 🔑 Key Concepts Demonstrated

| Concept | Where |
|---------|-------|
| Sigmoid activation & derivative | All notebooks |
| Forward propagation (matrix form) | All notebooks |
| Backpropagation via chain rule | All notebooks |
| Gradient descent weight updates | All notebooks |
| XOR problem (non-linear separability) | `backpropagation.ipynb` |
| Bottleneck / distributed representations | `mirror_symmetry.ipynb`, `family_tree_paper_experiment.ipynb` |
| Multi-label classification | `family_tree_paper_experiment.ipynb` |
| Momentum-based optimization | `family_tree_paper_experiment.ipynb` |
| Weight decay regularization | `family_tree_paper_experiment.ipynb` |
| Learning rate scheduling | `family_tree_paper_experiment.ipynb` |
| Representation visualization | `family_tree_paper_experiment.ipynb` |

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

**Copyright © 2026 Selamawit Elias**