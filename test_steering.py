import argparse
from src.eval.chatbot_app import SteeredChatbot

args = argparse.Namespace(
    model_name="unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit",
    adapter_dir="",
    hybrid_cbm_path="./hybrid_cbm.pt",
    model_dir="./t_trm_outputs",
    n_dynamic=5,
    clip_dim=512,
    n_latent=8,
    n_rules=10,
    layer_index=14,
    max_tokens=50,
    load_in_4bit=True,
    concepts_type="general",
    steering_alpha=1.0
)

print("Initializing engine...")
engine = SteeredChatbot(args)
print("Engine initialized.")

prompt = "Write a python function to add two numbers."
print("Prompting:", prompt)

response, concepts, rules = engine.generate_and_reason(prompt)

print("--- Response ---")
print(response)
print("--- Concepts ---")
for k, v in concepts.items():
    if v > 0.1:
        print(f"{k}: {v:.2f}")
print("--- Rules ---")
print(rules)
print("Test completed successfully.")
