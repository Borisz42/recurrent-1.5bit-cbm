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
├── docker/
│   └── Dockerfile.eval           # Sandboxed HumanEval benchmarking configuration
├── notebooks/
│   ├── kaggle_master_run.ipynb   # Master notebook for general models (DeepSeek-R1-Distill-Qwen)
│   └── coder_master_run.ipynb    # Master notebook for coder models (Qwen2.5-Coder)
├── src/
│   ├── system1/
│   │   ├── hook_extractor.py     # Detached activation forward hooks
│   │   ├── hybrid_cbm.py         # Hybrid CBM with Tanh bounding, CLIP, and custom static concepts (DEFAULT/CODER)
│   │   ├── train_sft.py          # Supervised Fine-Tuning CLI script
│   │   ├── extract_activations.py # Chunked activation extraction script
│   │   └── conceptnet_harvester.py # ConceptNet relation scraper & domain classifier
│   ├── t_trm/
│   │   ├── __init__.py
│   │   ├── loop.py               # DTLGN Gate, Neuron, Layer, and T-TRM loop
│   │   └── train_trm.py          # Standalone T-TRM loop training supporting --concepts_type
│   ├── cmr/
│   │   ├── logic.py              # Product & Gödel T-Norms, Concept Embeddings
│   │   ├── rule_module.py        # Rule categorical decoders
│   │   └── model.py              # CMR Model and PyTorch Lightning wrappers
│   └── eval/
│       ├── baseline.py           # Black-box task predictor baseline
│       ├── run_tests.py          # Post-training neuro-symbolic testing suite (CUE, Adversarial, etc.)
│       ├── run_coder_eval.py     # Safe, dockerized generative coding benchmark evaluator (HumanEval)
│       └── chatbot_app.py        # Interactive Gradio/CLI steered chatbot application
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
To generate the candidate concept bank for CLIP alignment, you can use the automated ConceptNet harvester script to gather clean domain-relevant concepts grouped by domain:
```bash
python src/system1/conceptnet_harvester.py --limit_per_seed 50
```
This queries the ConceptNet API (with automatic fallback to a local 55-concept taxonomy if the API is offline) and prints a copy-pasteable Python list.

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

For the general text model (default concepts):
```bash
python src/t_trm/train_trm.py --cache_dir "./cached_activations" --hybrid_cbm_path "./hybrid_cbm.pt" --output_dir "./t_trm_outputs"
```

For the coder model (30 static coder concepts + 20 dynamic concepts):
```bash
python src/t_trm/train_trm.py --cache_dir "./cached_coder_activations" --hybrid_cbm_path "./hybrid_cbm_coder.pt" --concepts_type coder --n_dynamic 20 --output_dir "./t_trm_coder_outputs"
```

---

## 🤖 Using the Model (Local Chatbot Inference)

Once training on Kaggle is complete, you can run the full steered chatbot locally on your RTX 3070 (which has 8GB VRAM—more than enough for this 1.5B parameters pipeline).

### 1. What to Export/Download from Kaggle
From your completed Kaggle workspace, download the following model checkpoint files:
*   **System 1 Adapters:** The `./adapters` directory (which contains `adapter_model.safetensors` and `adapter_config.json`).
*   **System 2 Checkpoints:** `./t_trm_outputs/t_trm_loop.pt` and `./t_trm_outputs/cmr_model.pt`.
*   **HybridCBM weights:** `./hybrid_cbm.pt`.

Place these in your local repository root folder.

### 2. Can I use this in LM Studio?
**No, not for the steered neuro-symbolic behavior.** 
*   **Why:** LM Studio executes standard GGUF models via `llama.cpp`. However, our framework relies on **System 2 steering** which requires executing a Python runtime to hook Layer 14 activations mid-generation, passing them through custom PyTorch tensor operations (`T-TRM` and `CMR`), and injecting a steering vector back. 
*   **Alternative:** You *can* merge your trained System 1 LoRA adapters into the base model, convert it to GGUF, and run it in LM Studio. However, this will only run the unsteered base model; it will lose the real-time safety, logical rules, and interpretability guarantees provided by System 2.
*   **To run the full steered chatbot:** Use the provided chatbot script `src/eval/chatbot_app.py` which will launch an interactive Gradio Web UI (with a live Neuro-Symbolic Debugger Panel showing active concepts and logic rules) or fall back to a terminal-based CLI.

    Run the script locally:
    ```bash
    # Run the interactive Gradio Web UI (default)
    python src/eval/chatbot_app.py --model_name "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit" --adapter_dir "./adapters"

    # Or run in terminal-only CLI mode
    python src/eval/chatbot_app.py --cli
    ```

### 3. How to Publish the Model
To publish the model so others can use it:
1.  **Merge the LoRA Adapters:** Merge the System 1 adapters back into the base model to create a single standalone model:
    ```python
    # Run locally or on Kaggle
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained("./adapters", load_in_4bit=False)
    model.save_pretrained_merged("your-hf-username/steered-system1-1.5b", tokenizer, save_method="merged_16bit")
    model.push_to_hub_merged("your-hf-username/steered-system1-1.5b", tokenizer, save_method="merged_16bit", token="YOUR_HF_TOKEN")
    ```
2.  **Publish System 2 Weights:** Upload `t_trm_loop.pt`, `cmr_model.pt`, and `hybrid_cbm.pt` to either:
    *   The "Files" tab of your Hugging Face model repository.
    *   A GitHub Release asset on your repository.
3.  **Provide the Wrapper:** Provide the code in `src/` so others can clone the repository, download your weights, and execute the steering loop.

---

## 🧪 Running the Evaluation Suite

To empirically validate the faithfulness and stability of the trained model, we provide a dedicated testing script that operates on the cached evaluation data:

For the general text model:
```bash
python src/eval/run_tests.py --cache_dir "./cached_activations" --model_dir "./t_trm_outputs"
```

For the coder model (30 static concepts, 20 dynamic concepts):
```bash
python src/eval/run_tests.py --cache_dir "./cached_coder_activations" --concepts_type coder --n_dynamic 20 --model_dir "./t_trm_coder_outputs"
```

The evaluation suite performs three major checks:
1. **Faithfulness & CUE Metric:** Trains a standard neural network (`BlackBoxBaseline`) directly on Layer 14 activations and compares its accuracy against our Concept Bottleneck. A CUE score > 0.90 proves the bottleneck is fully utilized and not leaking statistical variance.
2. **Adversarial Steering:** Verifies the temporal stability of the T-TRM loop by injecting Gaussian noise into activations and testing predicate dropout (masking active concepts). This ensures the model degrades gracefully (abstains) rather than confidently hallucinating.
3. **Rule Extraction:** Physically extracts and prints the highest confidence logic rules learned by the CMR module, proving that the decision-making process is human-auditable.

### 💻 Coder Model & HumanEval Benchmark (Safe Dockerized Evaluation)

For structured code generation, the model is evaluated on the standard **HumanEval** benchmark. To ensure complete safety from executing untrusted LLM-generated code locally on the host machine, the evaluation executes within an isolated, sandboxed Docker container using the official OpenAI HumanEval harness.

First, ensure Docker is running, then execute the evaluation runner:
```bash
# Evaluate the base coder model
python src/eval/run_coder_eval.py --model_name "unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit" --output_dir "./eval_outputs_base"

# Evaluate the SFT fine-tuned coder model
python src/eval/run_coder_eval.py --model_name "unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit" --adapter_dir "./adapters_coder" --output_dir "./eval_outputs_sft"
```
The script will automatically:
1. Build the sandboxed Docker image `humaneval-evaluator` from `docker/Dockerfile.eval`.
2. Generate completions for the HumanEval prompts using standard Hugging Face/Unsloth token generation.
3. Run the Docker container, mounting the completions directory, to safely calculate and print the `pass@1` score.

---

## 🛠️ Installation & Setup

### Installation
1. Clone the repository and install the standard dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. For optimized, high-speed SFT on NVIDIA GPUs, install Unsloth:
   ```bash
   pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
   ```

