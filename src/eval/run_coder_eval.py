import os
import sys
import json
import argparse
import subprocess
import torch
from tqdm import tqdm

# Add project root to python path to resolve src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False

from datasets import load_dataset

def clean_completion(generation):
    """
    Cleans model generation by removing text after the function body.
    For HumanEval, we stop generation when encountering class/function definitions
    at indent level 0, or standard script execution entrypoints.
    """
    lines = generation.split('\n')
    cleaned_lines = []
    for line in lines:
        if line.startswith('def ') or line.startswith('class ') or line.startswith('if __name__'):
            break
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def run_evaluation(args):
    print("==================================================")
    print("Coder Model Generative Evaluation (HumanEval)")
    print("==================================================")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Dataset
    print("Loading HumanEval dataset from Hugging Face...")
    try:
        dataset = load_dataset("openai_humaneval", split="test")
    except Exception as e:
        print(f"Error loading openai_humaneval from HF: {e}")
        print("Attempting to load from backup GitHub repository...")
        # Fallback URL
        import urllib.request
        import gzip
        url = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
        local_zip = "HumanEval.jsonl.gz"
        try:
            urllib.request.urlretrieve(url, local_zip)
            dataset = []
            with gzip.open(local_zip, "rb") as f_in:
                for line in f_in:
                    dataset.append(json.loads(line.decode('utf-8')))
            os.remove(local_zip)
            print("Successfully loaded HumanEval from backup source.")
        except Exception as ex:
            raise RuntimeError(f"Could not load HumanEval dataset: {ex}")
            
    if args.debug:
        print("Debug mode: limiting to first 3 problems.")
        if hasattr(dataset, "select"):
            dataset = dataset.select(range(min(3, len(dataset))))
        else:
            dataset = dataset[:3]
    elif args.limit > 0:
        print(f"Limiting to first {args.limit} problems.")
        if hasattr(dataset, "select"):
            dataset = dataset.select(range(min(args.limit, len(dataset))))
        else:
            dataset = dataset[:args.limit]
        
    # 2. Load Model
    print(f"Loading model: {args.model_name}")
    if HAS_UNSLOTH and not args.no_unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name,
            max_seq_length=args.max_seq_length,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            load_in_4bit=args.load_in_4bit,
        )
        if args.adapter_dir and os.path.exists(args.adapter_dir):
            print(f"Loading adapters from {args.adapter_dir}...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.adapter_dir,
                max_seq_length=args.max_seq_length,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                load_in_4bit=args.load_in_4bit,
            )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        kwargs = {}
        if args.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
        if device == "cuda":
            kwargs["device_map"] = "auto"
            kwargs["torch_dtype"] = torch.float16
            
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **kwargs)
        if args.adapter_dir and os.path.exists(args.adapter_dir):
            print(f"Loading adapters from {args.adapter_dir}...")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter_dir)
            
    model.eval()
    
    # 3. Generate Completions
    os.makedirs(args.output_dir, exist_ok=True)
    completions_file = os.path.join(args.output_dir, "humaneval_completions.jsonl")
    
    print(f"Generating completions for {len(dataset)} problems...")
    completions = []
    
    for item in tqdm(dataset):
        task_id = item["task_id"]
        prompt = item["prompt"]
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.temperature > 0.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        raw_gen = tokenizer.decode(generated_ids, skip_special_tokens=True)
        cleaned_gen = clean_completion(raw_gen)
        
        for _ in range(args.n_samples):
            completions.append({
                "task_id": task_id,
                "completion": cleaned_gen
            })
            
    with open(completions_file, "w", encoding="utf-8") as f:
        for entry in completions:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Saved {len(completions)} completions to: {completions_file}")
    
    # 4. Docker-based Safe Evaluation
    if args.no_docker:
        print("Skipping Docker evaluation because --no_docker was passed.")
        return
        
    print("\n--- Running Safe Dockerized HumanEval Harness ---")
    
    # Build the docker image
    dockerfile_path = os.path.join(args.project_root, "docker", "Dockerfile.eval")
    print(f"Building Docker image 'humaneval-evaluator' from {dockerfile_path}...")
    
    build_cmd = [
        "docker", "build", 
        "-t", "humaneval-evaluator", 
        "-f", dockerfile_path, 
        args.project_root
    ]
    
    try:
        subprocess.run(build_cmd, check=True)
        print("Docker image built successfully.")
    except Exception as e:
        print(f"Failed to build Docker image: {e}")
        return
        
    # Convert local absolute path to container-safe path
    abs_output_dir = os.path.abspath(args.output_dir).replace("\\", "/")
    
    run_cmd = [
        "docker", "run", "--rm",
        "-v", f"{abs_output_dir}:/app/eval_outputs",
        "humaneval-evaluator",
        "/app/eval_outputs/humaneval_completions.jsonl"
    ]
    
    print(f"Running evaluation command: {' '.join(run_cmd)}")
    try:
        result = subprocess.run(run_cmd, capture_output=True, text=True, check=True)
        print("\n--- HumanEval Output ---")
        print(result.stdout)
        if result.stderr:
            print("Stderr:")
            print(result.stderr)
            
        # Check results file
        results_file = completions_file + "_results.jsonl"
        if os.path.exists(results_file):
            print(f"Detailed results saved locally to: {results_file}")
            passed = 0
            failed = 0
            errors = {}
            with open(results_file, "r", encoding="utf-8") as rf:
                for line in rf:
                    try:
                        data = json.loads(line.strip())
                        res = data.get("result", "")
                        if res == "passed":
                            passed += 1
                        else:
                            failed += 1
                            if "AssertionError" in res:
                                err_type = "AssertionError"
                            elif "SyntaxError" in res:
                                err_type = "SyntaxError"
                            elif "NameError" in res:
                                err_type = "NameError"
                            elif "TypeError" in res:
                                err_type = "TypeError"
                            elif "IndexError" in res:
                                err_type = "IndexError"
                            elif "KeyError" in res:
                                err_type = "KeyError"
                            elif "Timeout" in res or "timed-out" in res.lower() or "time out" in res.lower():
                                err_type = "Timeout"
                            else:
                                err_type = res.split("\n")[0][:40] if "\n" in res else res[:40]
                            errors[err_type] = errors.get(err_type, 0) + 1
                    except Exception:
                        pass
            total_eval = passed + failed
            if total_eval > 0:
                print("\n==================================================")
                print("           HumanEval Execution Summary            ")
                print("==================================================")
                print(f"Total Evaluated Tasks: {total_eval}")
                print(f"Passed:                {passed} ({passed/total_eval*100:.2f}%)")
                print(f"Failed:                {failed} ({failed/total_eval*100:.2f}%)")
                if failed > 0:
                    print("\nFailure Breakdown by Error Type:")
                    for err_type, count in sorted(errors.items(), key=lambda x: x[1], reverse=True):
                        print(f"  - {err_type}: {count} ({count/failed*100:.1f}% of failures)")
                print("==================================================")
            
    except subprocess.CalledProcessError as e:
        print(f"Docker execution failed with exit code {e.returncode}")
        print("Stdout:")
        print(e.stdout)
        print("Stderr:")
        print(e.stderr)
    except Exception as e:
        print(f"Failed to run Docker container: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit")
    parser.add_argument("--adapter_dir", type=str, default="", help="Path to SFT adapter weights")
    parser.add_argument("--output_dir", type=str, default="./eval_outputs")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0, help="0.0 for deterministic pass@1")
    parser.add_argument("--n_samples", type=int, default=1, help="Samples per task")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--load_in_4bit", type=bool, default=True)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of evaluation problems")
    parser.add_argument("--no_unsloth", action="store_true", help="Force standard transformers")
    parser.add_argument("--no_docker", action="store_true", help="Skip Docker execution")
    parser.add_argument("--debug", action="store_true", help="Quick run with 3 problems")
    parser.add_argument("--project_root", type=str, default=".")
    
    args = parser.parse_args()
    args.project_root = os.path.abspath(args.project_root)
    run_evaluation(args)
