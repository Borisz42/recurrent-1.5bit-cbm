import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from peft import PeftModel

# Add repository root to python path to resolve src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.system1.hook_extractor import ActivationHookExtractor
from src.system1.hybrid_cbm import HybridCBM
from src.cmr.model import CMR, InputTypes
from src.t_trm.loop import TTRMLoop
from src.t_trm.train_trm import DummyEncoder

# Try to import Gradio
try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False

def find_layers_path(model):
    """Dynamically locates the transformer layers ModuleList."""
    for name, module in model.named_modules():
        if name.endswith("layers") and isinstance(module, nn.ModuleList):
            return name
    return None

class SteeredChatbot:
    def __init__(self, args):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.args = args
        
        print("Loading tokenizer and base model...")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Optimize loading based on available device
        if self.device == "cuda":
            kwargs = {
                "device_map": "auto",
                "dtype": torch.float16,
            }
            if args.load_in_4bit:
                try:
                    config = AutoConfig.from_pretrained(args.model_name)
                    has_quant = getattr(config, "quantization_config", None) is not None
                except Exception:
                    has_quant = False
                
                if not has_quant:
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16
                    )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                **kwargs
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                dtype=torch.float32
            )
            
        if args.adapter_dir and os.path.exists(args.adapter_dir):
            print(f"Loading LoRA adapters from {args.adapter_dir}...")
            self.model = PeftModel.from_pretrained(self.model, args.adapter_dir)
            
        self.model.eval()
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.max_length = None
        
        # Setup activation hook
        layers_path = find_layers_path(self.model)
        if layers_path is None:
            raise ValueError("Could not find layers ModuleList in model.")
        self.target_layer = f"{layers_path}[{args.layer_index}]"
        self.extractor = ActivationHookExtractor(self.model, self.target_layer)
        
        # Initialize concepts configuration
        emb_dim = self.model.config.hidden_size
        self.hybrid_cbm = HybridCBM(n_dynamic=args.n_dynamic, clip_dim=args.clip_dim, emb_dim=emb_dim).to(self.device)
        if os.path.exists(args.hybrid_cbm_path):
            print(f"Loading HybridCBM weights from {args.hybrid_cbm_path}...")
            self.hybrid_cbm.load_state_dict(torch.load(args.hybrid_cbm_path, map_location=self.device))
        self.hybrid_cbm.eval()
        
        n_concepts = self.hybrid_cbm.n_static + args.n_dynamic
        self.concept_names = self.hybrid_cbm.concepts + [f"dynamic_{i}" for i in range(args.n_dynamic)]
        
        # Initialize CMR & T-TRM Loop
        self.cmr_model = CMR(
            encoder=DummyEncoder(128, n_concepts),
            emb_size=128, rule_emb_size=64, n_tasks=2,
            n_rules=args.n_rules, n_concepts=n_concepts, concept_names=self.concept_names,
            selector_input=InputTypes.concepts
        ).to(self.device)
        
        self.t_trm = TTRMLoop(
            d_model=emb_dim, n_concepts=n_concepts, n_tasks=2,
            n_rules=args.n_rules, n_latent=args.n_latent, n_layers=2
        ).to(self.device)
        
        t_trm_path = os.path.join(args.model_dir, "t_trm_loop.pt")
        cmr_path = os.path.join(args.model_dir, "cmr_model.pt")
        
        if os.path.exists(t_trm_path) and os.path.exists(cmr_path):
            print("Loading T-TRM and CMR checkpoints...")
            self.t_trm.load_state_dict(torch.load(t_trm_path, map_location=self.device))
            self.cmr_model.load_state_dict(torch.load(cmr_path, map_location=self.device))
        else:
            print("Warning: Pre-trained checkpoints not found. Running with initialized weights.")
            
        self.t_trm.eval()
        self.cmr_model.eval()

    def generate_and_reason(self, prompt):
        # Format instruction template
        formatted_prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n"
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        # 1. Run forward pass with hooks to capture activations on the prompt
        self.extractor.clear()
        with self.extractor:
            with torch.no_grad():
                _ = self.model(**inputs)
                
        # 2. Get Layer 14 activations
        raw_activations = self.extractor.extracted_activations[0] # [1, seq_len, emb_dim]
        # Mean pool over sequence dimension to extract a single context representation
        x = raw_activations.mean(dim=1) # [1, emb_dim]
        
        # 3. Evaluate concept activations and CMR rules via T-TRM Loop
        with torch.no_grad():
            _, _, c_pred, y_pred = self.t_trm(x.float(), self.cmr_model, T_loops=3, n_steps=2, hard=True)
            c_probs = c_pred[0].cpu().numpy() # [n_concepts]
            y_probs = y_pred[0].cpu().numpy() # [n_tasks]
            
        # Map concepts to printable values
        concept_status = {name: float(prob) for name, prob in zip(self.concept_names, c_probs)}
        
        # Get active rules (simplification for visualization)
        active_rules = []
        rule_vars = self.cmr_model.get_all_rule_vars() # [n_tasks, n_rules, n_concepts, 3]
        for task_idx in range(y_probs.shape[0]):
            task_status = "ACTIVE" if y_probs[task_idx] > 0.5 else "INACTIVE"
            active_rules.append(f"Task {task_idx+1} ({task_status}):")
            
            # Simple demonstration of CMR rule format
            # In a complete implementation, we look at the weight selector values
            active_rules.append(f"  Applied logic constraints over concepts to decide status.")
        
        # 4. Generate the actual response text from System 1
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                temperature=0.7,
                do_sample=True
            )
        
        response_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response_text, concept_status, "\n".join(active_rules)

def launch_gradio(chatbot_engine):
    def chat_fn(message, history):
        response, concepts, rules = chatbot_engine.generate_and_reason(message)
        
        # Format neuro-symbolic output nicely for display
        concept_md = "#### 🧠 Concept Activations:\n"
        for name, val in concepts.items():
            bar = "█" * int(val * 10) + "░" * (10 - int(val * 10))
            concept_md += f"- **{name}**: {bar} ({val*100:.1f}%)\n"
            
        rules_md = f"#### ⚖️ Decider Rules:\n```text\n{rules}\n```"
        
        full_display = f"{response}\n\n---\n### 🔬 Neuro-Symbolic Debugger Panel\n{concept_md}\n{rules_md}"
        return full_display

    print("Launching Gradio interface...")
    gr.ChatInterface(
        chat_fn,
        title="Recurrent 1.5-Bit Concept Bottleneck Chatbot",
        description="Interact with System 1 (Causal LLM) steered and monitored by System 2 (T-TRM + CMR).",
    ).launch(theme="soft")

def launch_cli(chatbot_engine):
    print("\n==================================================")
    print("Welcome to the Steered Neuro-Symbolic Chatbot CLI")
    print("Type 'exit' or 'quit' to end the session.")
    print("==================================================\n")
    
    while True:
        try:
            prompt = input("User > ")
            if prompt.lower() in ["exit", "quit"]:
                break
                
            response, concepts, rules = chatbot_engine.generate_and_reason(prompt)
            print("\n------------------ System 1 Output ------------------")
            print(response)
            print("\n------------------ System 2 Debugger ------------------")
            print("🧠 Concepts:")
            for name, val in concepts.items():
                print(f"  - {name:25s}: {val*100:5.1f}%")
            print("\n⚖️ Active Logic Constraints:")
            print(rules)
            print("=======================================================\n")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit")
    parser.add_argument("--adapter_dir", type=str, default="./adapters")
    parser.add_argument("--hybrid_cbm_path", type=str, default="./hybrid_cbm.pt")
    parser.add_argument("--model_dir", type=str, default="./t_trm_outputs")
    parser.add_argument("--n_dynamic", type=int, default=5)
    parser.add_argument("--clip_dim", type=int, default=512)
    parser.add_argument("--n_latent", type=int, default=8)
    parser.add_argument("--n_rules", type=int, default=10)
    parser.add_argument("--layer_index", type=int, default=14)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--cli", action="store_true", help="Force CLI mode even if gradio is available")
    
    args = parser.parse_args()
    
    engine = SteeredChatbot(args)
    
    if HAS_GRADIO and not args.cli:
        launch_gradio(engine)
    else:
        if not HAS_GRADIO:
            print("Gradio not installed. Falling back to command-line interface...")
        launch_cli(engine)
