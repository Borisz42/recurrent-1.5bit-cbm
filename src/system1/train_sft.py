import os
import argparse
import torch

# Import unsloth first to ensure all optimizations are applied correctly
try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False

from datasets import load_dataset
from transformers import TrainingArguments
if not HAS_UNSLOTH:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

def train(args):
    print(f"Loading dataset: {args.dataset}")
    # Load dataset
    dataset = load_dataset(args.dataset, split="train")
    
    # Optional: Take a small subset if debug mode
    if args.debug:
        dataset = dataset.select(range(min(10, len(dataset))))
        print(f"Debug mode: subsetting dataset to {len(dataset)} examples.")
        
    print(f"Loading model: {args.model_name}")
    if HAS_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = args.model_name,
            max_seq_length = args.max_seq_length,
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
            load_in_4bit = args.load_in_4bit,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r = args.lora_r,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha = args.lora_alpha,
            lora_dropout = 0,
            bias = "none",
            use_gradient_checkpointing = "unsloth"
        )
    else:
        print("Warning: Unsloth is not installed. Falling back to standard HuggingFace transformers and PEFT.")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        if args.load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        else:
            quantization_config = None

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=quantization_config,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)

    # Standard Chat Template / Prompt Formatter
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    EOS_TOKEN = tokenizer.eos_token
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            # Must add EOS_TOKEN, otherwise generation will go on forever
            text = alpaca_prompt.format(instruction, input_text, output) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts }

    formatted_dataset = dataset.map(formatting_prompts_func, batched=True)

    print("Starting SFTTrainer setup...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = formatted_dataset,
        dataset_text_field = "text",
        max_seq_length = args.max_seq_length,
        dataset_num_proc = 1, # Set to 1 to prevent multiprocessing overhead
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = args.batch_size,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            warmup_steps = 5,
            max_steps = args.max_steps if args.debug else -1,
            num_train_epochs = args.epochs if not args.debug else 1,
            learning_rate = args.lr,
            fp16 = not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = args.output_dir,
        ),
    )

    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning completed successfully!")
    
    # Save adapter model
    print(f"Saving PEFT adapters to: {args.adapter_dir}")
    model.save_pretrained(args.adapter_dir)
    tokenizer.save_pretrained(args.adapter_dir)
    print("Model saved successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 SFT Training with Unsloth")
    parser.add_argument("--model_name", type=str, default="unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit")
    parser.add_argument("--dataset", type=str, default="tatsu-lab/alpaca")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--adapter_dir", type=str, default="./adapters")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--load_in_4bit", type=bool, default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=10, help="Used only for debug or max step capping")
    parser.add_argument("--debug", action="store_true", help="Debug mode with subset data and limited steps")
    args = parser.parse_args()
    train(args)
