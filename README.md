# Recurrent 1.5-Bit Concept Bottleneck Architecture

This repository contains the implementation of the **Ternary Tiny Recursive Model (T-TRM)** integrated with a **Hybrid Concept Bottleneck Model (HybridCBM)** and a **Concept-based Memory Decider (CMR)**. 

The framework establishes a unified neuro-symbolic core optimized to maximize benchmarks in conversational evaluation, multi-step logical deduction, and structured code generation within the strict computational bounds of consumer-grade hardware (such as a single 16 GB VRAM GPU on Kaggle).

---

## 🗺️ Architectural Paradigm & Decoupled Execution

Traditional autoregressive recursive models often attempt to run nested, token-internal recurrence directly inside a causal decoder. This leads to **gradient interference** (exploding/vanishing gradients through dual-sequence backpropagation) and a **compute allocation disadvantage** (due to a tied representation space).

To resolve this **Autoregressive Compute Paradox**, this architecture deploys a decoupled, two-pathway system:
* **System 1**: A standard causal language model (e.g., `DeepSeek-R1-Distill-Qwen-1.5B` or `Qwen2.5-Coder-1.5B`) to generate tokens, preserving causal masking and key-value cache efficiency.
* **System 2 (T-TRM Loop)**: An external, non-disruptive feedback loop. At each decoding step, hidden activations $x$ are extracted from Layer 14, projected to the concept space, and recurrently updated inside a ternary logic monitor to guide subsequent token probabilities via a continuous steering vector $y$.

```
              +-------------------------------------------+
              |   System 1: Causal Autogeneration Engine  |
              |       (DeepSeek-R1-Distill-Qwen-1.5B      |
              |         or Qwen2.5-Coder-1.5B/3B)         |
              +-------------------------------------------+
                                    |
                          (Extract Layer 14)
                                    |
                                    v
                      +---------------------------+
                      |    System 2: T-TRM Loop   |
                      |   (PST Trained Recurrence) |
                      +---------------------------+
                                    |
                                    v
                     +-----------------------------+
                     |  Hybrid Concept Bottleneck  |
                     | (Static KG + Dynamic Bank)  |
                     +-----------------------------+
                                    |
                                    v
                   +---------------------------------+
                   | Concept-based Memory Decider    |
                   |     (Rule Selection & Eval)     |
                   +---------------------------------+
                                    |
                         (Project steering vector)
                                    |
                                    v
```

---

## 🧬 Mathematical Formulations

### 1. Differentiable Ternary Logic Gate Network (DTLGN)
To make Kleene's three-valued logic domain $\mathbb{T} = \{-1, 0, 1\}$ (where $0$ is an unknown state) differentiable during training, we implement **Polynomial Surrogate Training (PST)**. 

Each ternary logic gate is represented during training as a degree-(2,2) bivariate polynomial. Rather than running a softmax over $3^{3^2} = 19,683$ possible discrete ternary logic gates, we learn continuous outputs $o_{i,j} \in [-1, 1]$ at the 9 grid points of $\{-1, 0, 1\}^2$. For any continuous inputs $x_1, x_2$, the output is computed via **Lagrange basis interpolation**:

$$P(x_1, x_2) = \sum_{i \in \{-1, 0, 1\}} \sum_{j \in \{-1, 0, 1\}} o_{i, j} L_i(x_1) L_j(x_2)$$

where the Lagrange basis polynomials are:
* $L_{-1}(x) = \frac{x(x-1)}{2}$
* $L_0(x) = 1 - x^2$
* $L_1(x) = \frac{x(x+1)}{2}$

### 2. Recurrent Stability & Formal Guarantees
To prevent chaotic oscillations in the latent recurrence and support principled uncertainty propagation under degraded sensor/context inputs, the DTLGN gates are regularized into two structured vocabularies:

* **Numerically Monotone (NM) Gates**: Preserves numerical ordering ($x \le x' \implies f(x) \le f(x')$). Enforces stability and guarantees fixed points via Tarski's theorem.
* **Information-Monotone (IM) Gates**: Preserves certainty ordering ($x \sqsubseteq x' \implies f(x) \sqsubseteq f(x')$). Guarantees **principled abstention**—unknown inputs ($0$) propagate safely without flipping to false active states ($1$ or $-1$).

These properties are optimized using differentiable ReLU penalty functions:
* **NM Loss**: Penalizes numerical decreases along rows/columns of the gate outputs.
* **IM Loss**: For all $u \sqsubseteq v$, penalizes if the output certainty decreases.

### 3. Rule Selection & Hardening
At inference, the continuous gate outputs are hardened into exact discrete ternary logic circuits. The optimization is guided by a **Commitment Loss** that pushes softmax probabilities $p_{k,v}$ at each grid point $k$ to become one-hot:

$$\mathcal{L}_{commit} = \sum_{k=0}^{8} \sum_{v=0}^{2} (p_{k, v}^2 - p_{k, v})^2$$

The task prediction is handled by the **Concept-based Memory Decider (CMR)** which evaluates selected rules symbolically on the concept bottleneck:

$$p(y, r, c \mid x) = p(y \mid c, r) p(r \mid x) p(c \mid x)$$

To map the Kleene states $\{-1, 0, 1\}$ from the DTLGN loop to standard CMR concept probability targets in $[0, 1]$, we apply the transformation:

$$z_{CMR} = \frac{z_{T-TRM} + 1}{2}$$

---

## 📂 Codebase Layout

```
.
├── notebooks/
│   └── kaggle_master_run.ipynb   # SFT, Activation Caching, HybridCBM training & Concept Translation
├── src/
│   ├── system1/
│   │   ├── hook_extractor.py     # Detached activation forward hooks
│   │   ├── hybrid_cbm.py         # Hybrid CBM with Tanh bounding & CLIP translation
│   │   ├── train_sft.py          # Supervised Fine-Tuning CLI script
│   │   └── extract_activations.py # Chunked activation extraction script
│   ├── t_trm/
│   │   ├── __init__.py
│   │   ├── loop.py               # DTLGN Gate, Neuron, Layer, and T-TRM loop
│   │   └── train_trm.py          # Standalone T-TRM loop training
│   └── cmr/
│       ├── logic.py              # Product & Gödel T-Norms, Concept Embeddings
│       ├── rule_module.py        # Rule categorical decoders
│       └── model.py              # CMR Model and PyTorch Lightning wrappers
├── plan.md                       # Technical viability and math blueprint
└── requirements.txt              # Dependency specifications
```

---

## 📊 Performance Profiles & Baselines

| Optimization Layer | Target Domain | Model Architecture | Parameters | VRAM (4-bit SFT) | Performance Baseline |
|--------------------|---------------|--------------------|------------|------------------|----------------------|
| System 1: General  | Logical deduction & conversational Elo. | DeepSeek-R1-Distill-Qwen-1.5B. | 1.5 Billion | 8 - 10 GB (Unsloth). | AIME 2024: 28.9% pass@1; MATH-500: 83.9%. |
| System 1: Coding   | Syntactic code generation & logic. | Qwen2.5-Coder-1.5B. | 1.5 Billion | 8 - 10 GB (Unsloth). | HumanEval: 43.3% pass@1; supports 92 languages. |
| System 2: Monitor  | Real-time latent state steering. | Decoupled T-TRM Loop. | 7 Million | Negligible (<1 GB). | Inherent stability and formal safety guarantees. |
| Standalone Solver  | Spatial grid mapping (ARC-AGI). | Standalone T-TRM. | 7 Million | 3 - 4 GB (System RAM). | Compete-ready performance at 0.0001x scale. |

---

## 🚀 Execution Pipeline

The execution is partitioned into four modular stages:

### Stage 1: Base Model Fine-Tuning
Execute LoRA SFT on the base model (quantized in 4-bit) with instructions formatted using `<think>...</think>` tags to preserve distilled reasoning traces:
```bash
python src/system1/train_sft.py \
    --model_name "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit" \
    --dataset "tatsu-lab/alpaca" \
    --output_dir "./outputs" \
    --adapter_dir "./adapters" \
    --epochs 1
```

### Stage 2: Batched Layer Caching
Hook Layer 14 of the fine-tuned model and save intermediate activations. To prevent CPU memory overflow on large datasets, the extractor automatically flushes batched tensors to chunked `.safetensors` files:
```bash
python src/system1/extract_activations.py \
    --model_name "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit" \
    --adapter_dir "./adapters" \
    --dataset "tatsu-lab/alpaca" \
    --output_dir "./cached_activations" \
    --chunk_size 100
```

### Stage 3: Incremental Concept Translation
Train the `HybridCBM` model (with `tanh` bounding to avoid representation leakage) on cached activations. Post-training, project the learnable dynamic concept vectors into the Candidate Concept Bank via CLIP space cosine similarity:
```python
from safetensors.torch import load_file
from src.system1.hybrid_cbm import HybridCBM

# Train HybridCBM layers
hybrid_cbm = HybridCBM(n_dynamic=5, clip_dim=512, emb_dim=1536).to(device)
# [run training loop on activations...]

# Translate dynamic concept vectors
translated_labels, similarities = hybrid_cbm.translate_dynamic_concepts(candidate_embeddings, candidate_labels)
```

### Stage 4: T-TRM Loop & Rule Hardening
Instantiate the `TTRMLoop` to coordinate DTLGN latent steps and CMR rule updates, optimizing with NM, IM, and routing commitment regularizers.

---

## 🛠️ Installation & Tests

### Installation
1. Clone the repository and install the standard dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. For optimized, high-speed SFT on NVIDIA GPUs, install Unsloth:
   ```bash
   pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
   ```

