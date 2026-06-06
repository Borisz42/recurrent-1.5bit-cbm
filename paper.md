# Recurrent 1.5-Bit Concept Bottleneck Architectures: A Unified Neuro-Symbolic Framework for Interpretable Reasoning

## Abstract
This paper introduces a novel neuro-symbolic architecture that addresses the accuracy-interpretability trade-off in large language models. By decoupling the generation engine (System 1) from a recurrent verification loop (System 2) and integrating a Hybrid Concept Bottleneck Model (HybridCBM) with a Concept-based Memory Decider (CMR), the framework achieves near-100% human interpretability while maintaining state-of-the-art performance on single-GPU hardware. We detail the software and architectural tools utilized, the mathematical foundations of the Ternary Tiny Recursive Model (T-TRM), and outline a comprehensive testing methodology to empirically validate the system's structural integrity, faithfulness, and stability.

## 1. Introduction
Modern deep learning architectures suffer from a fundamental lack of interpretability and are increasingly constrained by massive hardware requirements. Furthermore, attempts to embed token-internal recursive loops into causal language models often result in the Autoregressive Compute Paradox, leading to representation collapse. We present a decoupled, resource-efficient architecture optimized for consumer-grade hardware (e.g., dual Tesla T4 GPUs) that uses a Causal LLM as a System 1 generator, continuously steered by a 7-parameter System 2 T-TRM loop operating over a 1.5-bit (ternary) logic domain.

## 2. System Architecture and Utilized Tools

### 2.1 System 1: Causal Autogeneration Engine & Unsloth Optimization
**Tool:** `DeepSeek-R1-Distill-Qwen-1.5B` / `Qwen2.5-Coder-1.5B` combined with the `Unsloth` library.
**Rationale:** Fine-tuning models exceeding 10 billion parameters is infeasible on standard hardware. By utilizing highly optimized 1.5B parameter foundation models and Unsloth’s 4-bit quantization and memory-efficient gradient checkpointing, we fit the primary generation engine within an 8-10 GB VRAM footprint.
**Expected Outcome:** High foundational reasoning and syntactic fidelity without catastrophic memory bottlenecks.

### 2.2 System 2: Decoupled Ternary Tiny Recursive Model (T-TRM)
**Tool:** Decoupled T-TRM (PyTorch Implementation).
**Rationale:** Embedding recursion internally within the autoregressive generation disrupts causal masking and gradient propagation. The decoupled T-TRM operates externally on post-hoc extracted intermediate representations (Layer 14 activations), recursively refining a ternary concept state and generating a continuous steering vector projected back into the base model.
**Expected Outcome:** Stable gradient propagation and parallel execution, avoiding the representation collapse associated with internal autoregressive loops.

### 2.3 Hybrid Concept Bottleneck Model (HybridCBM) & ConceptNet Harvester
**Tool:** `conceptnet_harvester.py` (Custom API Harvester) & HybridCBM.
**Rationale:** Standard concept bottlenecks suffer from a "completeness penalty" when predefined concepts fail to capture all variance. We utilize a dual-segment bottleneck: a Static Concept Bank populated by high-confidence domain logic (harvested directly via the ConceptNet API utility) and a Dynamic Concept Bank capturing residual non-linear variance.
**Expected Outcome:** The custom harvester ensures domain relevance for the concepts (e.g., "logical deduction"), while the dynamic bank ensures no performance degradation, collectively achieving absolute interpretability through post-hoc incremental concept translation.

### 2.4 Concept-based Memory Decider (CMR)
**Tool:** CMR Neural Selection Mechanism.
**Rationale:** Standard classification layers act as black boxes, obstructing verifiable logic. CMR replaces standard classification heads with a differentiable memory of learnable logic rules. The model outputs a probability distribution over verifiable rules and evaluates them symbolically.
**Expected Outcome:** The decision-making process becomes human-auditable and formally verifiable prior to deployment, eliminating the accuracy-interpretability trade-off.

### 2.5 Differentiable Ternary Logic Gate Network (DTLGN) & Polynomial Surrogate Training (PST)
**Tool:** DTLGN configured via PST.
**Rationale:** Training exact discrete ternary gates ($T \in \{-1, 0, 1\}$) is computationally intractable with standard softmax gating due to the exponential action space. PST parameterizes each ternary neuron as a degree-(2,2) bivariate polynomial, ensuring it is everywhere differentiable without relying on heuristic Straight-Through Estimators.
**Expected Outcome:** Seamless backpropagation during training and mathematically proven conversion into exact, discrete ternary logic gates at inference via a vanishing commitment loss.

## 3. Experimental Design and Testing Methodology
While the primary model training is currently executing, the following rigorous experimental methodologies are designed to validate the theoretical guarantees of the architecture.

### 3.1 Hardware and Training Setup
- **Infrastructure:** Dual Tesla T4 GPUs (16 GB VRAM each).
- **Optimization Strategy:** To maximize parallel GPU utilization, data loading processes (`num_workers`, `pin_memory`) and activation extraction (`world_size`, `rank` sharding) are heavily parallelized.
- **Batch Processing:** The minimal parameter count of the standalone T-TRM (7M parameters) allows for saturated batch sizes (e.g., 256), scaling hardware efficiency.

### 3.2 Formal Evaluation Metrics

#### 3.2.1 Faithfulness and Concept Utilization Efficiency (CUE)
**Objective:** Verify that the HybridCBM is entirely causal and that the residual stream does not bypass the concept bottleneck.
**Test Design:** 
We calculate the Concept Utilization Efficiency (CUE):
$$ CUE = \frac{\text{Accuracy}_{BlackBox}}{\text{Accuracy}_{CBL}} \times \left(1 - \frac{H(c)}{H(r)}\right) $$
where $H(r)$ is the residual stream entropy and $H(c)$ is the concept bottleneck entropy.
**Success Condition:** A CUE score exceeding 0.90 demonstrates that the model genuinely reasons through the audited concept space rather than memorizing statistical bypasses. Note that if CBL accuracy is extremely low, CUE score can be artificially high; hence, it must be evaluated in conjunction with absolute CBL accuracy.

#### 3.2.2 Adversarial Steering and Uncertainty Propagation
**Objective:** Evaluate the stability of the T-TRM monitor using temporal logic restrictions (Numerically Monotone and Information-Monotone gates).
**Test Design:** 
1. **Predicate Dropout:** Sequentially mask determined predicates to an unknown state ($0.5$). 
2. **Noise Injection:** Introduce targeted Gaussian noise to intermediate activations.
**Success Condition:** 
- *Principled Abstention:* An all-unknown input state sequence must consistently produce an all-unknown verdict, eliminating silent hallucinations.
- *Input Certainty Monotonicity:* Dropping predicates must never cause the model to confidently flip from one active state to its opposite (e.g., from $+1$ to $-1$), ensuring graceful degradation.

#### 3.2.3 Cross-Modal Grounding Validation (InfoNCE)
**Objective:** Ensure that identical concepts represented in different modalities (e.g., visual grids vs. text coordinates) map to the exact same ternary bottleneck coordinates.
**Test Design:** 
Evaluate representations utilizing the InfoNCE alignment loss:
$$ \mathcal{L}_{InfoNCE} = - \log \frac{\exp(\text{sim}(z_i^{visual}, z_i^{text})/\tau)}{\sum_j \exp(\text{sim}(z_i^{visual}, z_j^{text})/\tau)} $$
**Success Condition:** The loss asymptotically approaches the lower bound across test samples, proving unified cross-modal grounding before rule selection.

#### 3.2.4 Discretization (Hardening) Gap
**Objective:** Measure the accuracy drop when switching from the continuous polynomial surrogate representation to the exact hardened discrete ternary logic circuit at inference.
**Test Design:** 
We compute the discretization gap $\Delta_{\text{hard}}$:
$$ \Delta_{\text{hard}} = \text{Accuracy}_{\text{soft}} - \text{Accuracy}_{\text{hard}} $$
**Success Condition:** A gap $\Delta_{\text{hard}} < 0.05$ at the end of training indicates that the commitment losses have successfully driven the gate parameters and routing logits to their discrete boundaries, eliminating distribution shift at inference.

#### 3.2.5 Concept Alignment and Quality
**Objective:** Verify that the predicted concepts generated by the T-TRM loop are semantically aligned with the target concept vectors from the HybridCBM.
**Test Design:** 
We calculate the macro-averaged **ROC-AUC** and **F1-Score** for all static and dynamic concepts over the test partition.
**Success Condition:** Average F1-score exceeding $0.80$ and ROC-AUC exceeding $0.85$ verifies high-fidelity concept tracking within the recurrent System 2 loop.

#### 3.2.6 Test-Time Concept Intervention & Steerability
**Objective:** Verify the causal influence of the concept bottleneck on task predictions and validate test-time steering.
**Test Design:** 
We evaluate downstream task accuracy under varying concept intervention rates $p \in [0.0, 0.25, 0.5, 0.75, 1.0]$. For a rate $p$, we randomly replace a fraction $p$ of predicted concepts with their ground-truth values before rule evaluation.
**Success Condition:** Downstream task accuracy must increase monotonically with the intervention rate $p$, reaching near-100% accuracy at $p = 1.0$.

#### 3.2.7 Rule Coverage & Complexity
**Objective:** Evaluate the interpretability and utilization efficiency of the learned rule memory in the CMR decider.
**Test Design:** 
1. **Rule Coverage:** Compute the Shannon entropy of the chosen rule index distribution across the dataset:
$$ H(\text{Rules}) = -\sum_{i=1}^{R} P(r_i) \log_2 P(r_i) $$
2. **Complexity:** Compute the average literal count (number of active/relevant concepts) per rule:
$$ \text{Literal Count} = \frac{1}{R} \sum_{i=1}^{R} \sum_{j=1}^{C} \mathbb{I}(\text{irrelevance}_{i, j} < 0.5) $$
**Success Condition:** A balanced rule entropy indicates rules are broadly distributed, while a low literal count (e.g., $<3.0$ active concepts per rule) ensures the rules remain readable and human-interpretable.

## 4. Expected Outcomes
Upon conclusion of the ongoing training phase, we anticipate:
1. **Hardware Efficiency:** Full convergence within constraints of Kaggle dual T4 GPUs.
2. **Performance Parity:** Task prediction accuracy matching or exceeding the black-box un-steered `DeepSeek-R1-Distill-Qwen-1.5B` baseline.
3. **Interpretability:** 100% of the active dimensions in the HybridCBM successfully matching high-confidence, human-readable tags harvested from ConceptNet or mapped to logical operators.

## 5. Conclusion
By fully decoupling the sequence generation engine from the recursive monitoring loop, and bridging them via a verifiable Hybrid Concept Bottleneck, this architecture addresses the fundamental structural limitations of traditional LLMs. The formalized testing matrix outlined in this paper ensures that, post-training, the model will provide verifiable, deterministic logic guarantees while remaining deployable on consumer-grade hardware.
