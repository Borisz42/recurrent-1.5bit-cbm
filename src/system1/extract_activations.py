import os
import argparse
import torch

# Import unsloth first to ensure all optimizations are applied correctly
try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False

import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from safetensors.torch import save_file
from src.system1.hook_extractor import ActivationHookExtractor
if not HAS_UNSLOTH:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

def find_layers_path(model):
    """
    Dynamically finds the dot-separated path to the transformer layers ModuleList.
    This resolves issues where PEFT or other wrappers nest the layer attributes.
    """
    for name, module in model.named_modules():
        # Look for a ModuleList named 'layers'
        if name.endswith("layers") and isinstance(module, nn.ModuleList):
            return name
    return None

def extract(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading tokenizer and model: {args.model_name}")
    
    if HAS_UNSLOTH and not args.no_unsloth:
        # Load using Unsloth
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = args.model_name,
            max_seq_length = args.max_seq_length,
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
            load_in_4bit = args.load_in_4bit,
        )
        if args.adapter_dir and os.path.exists(args.adapter_dir):
            print(f"Loading PEFT adapters from: {args.adapter_dir}")
            # Unsloth supports loading adapters directly:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = args.adapter_dir,
                max_seq_length = args.max_seq_length,
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
                load_in_4bit = args.load_in_4bit,
            )
    else:
        # Standard HuggingFace fallback
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            load_in_4bit=args.load_in_4bit,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        # Configure padding if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        if args.adapter_dir and os.path.exists(args.adapter_dir):
            print(f"Loading PEFT adapters from: {args.adapter_dir}")
            model = PeftModel.from_pretrained(model, args.adapter_dir)

    model.eval()
    
    # Auto-resolve layer path to avoid hardcoded attribute errors
    layers_base_path = find_layers_path(model)
    if layers_base_path is None:
        raise ValueError("Could not automatically locate the transformer layers ModuleList in the model.")
    
    target_layer_path = f"{layers_base_path}[{args.layer_index}]"
    print(f"Resolved target layer path: {target_layer_path}")
    
    # Initialize hook extractor
    extractor = ActivationHookExtractor(model, layer_path=target_layer_path)
    extractor.register()
    
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="train")
    if args.debug:
        dataset = dataset.select(range(min(20, len(dataset))))
        print(f"Debug mode: subsetting dataset to {len(dataset)} examples.")

    os.makedirs(args.output_dir, exist_ok=True)
    
    batch_size = args.batch_size
    chunk_size = args.chunk_size
    
    chunk_idx = 0
    chunk_samples = 0
    
    print("Extracting layer activations...")
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size)):
            # Load batch slice
            batch_slice = dataset.select(range(i, min(i + batch_size, len(dataset))))
            
            # Format inputs
            texts = []
            for item in batch_slice:
                # Alpaca format template without response (to extract contextual reasoning stream)
                text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n"
                texts.append(text)
                
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=args.max_seq_length, return_tensors="pt").to(device)
            
            # Run forward pass (hook captures target layer outputs)
            _ = model(**inputs)
            
            chunk_samples += len(batch_slice)
            
            # Save chunk and flush RAM if capacity reached
            if chunk_samples >= chunk_size or (i + batch_size) >= len(dataset):
                chunk_filepath = os.path.join(args.output_dir, f"chunk_{chunk_idx}.safetensors")
                
                # Concatenate along batch dimension (dim=0)
                activations = extractor.get_concatenated_activations(dim=0)
                
                # Save using safetensors
                save_file({"activations": activations}, chunk_filepath)
                print(f"Saved chunk {chunk_idx} (shape {activations.shape}) to: {chunk_filepath}")
                
                # Clear memory buffer
                extractor.clear()
                chunk_samples = 0
                chunk_idx += 1

    extractor.remove()
    print("Activation extraction completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 Activation Extraction")
    parser.add_argument("--model_name", type=str, default="unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit")
    parser.add_argument("--adapter_dir", type=str, default="./adapters", help="Path to SFT LoRA adapters")
    parser.add_argument("--dataset", type=str, default="tatsu-lab/alpaca")
    parser.add_argument("--output_dir", type=str, default="./cached_activations")
    parser.add_argument("--layer_index", type=int, default=14)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--load_in_4bit", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--chunk_size", type=int, default=100, help="Number of samples per saved safetensors file")
    parser.add_argument("--no_unsloth", action="store_true", help="Force standard transformers instead of unsloth")
    parser.add_argument("--debug", action="store_true", help="Debug mode with subset data")
    args = parser.parse_args()
    extract(args)
