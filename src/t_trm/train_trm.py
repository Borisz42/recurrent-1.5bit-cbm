import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from safetensors.torch import load_file

# Add repository root to python path to resolve src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.system1.hybrid_cbm import HybridCBM
from src.cmr.model import CMR, InputTypes
from src.t_trm.loop import TTRMLoop

class DummyEncoder(nn.Module):
    def __init__(self, emb_size, n_concepts):
        super().__init__()
        self.emb_size = emb_size
        self.n_concepts = n_concepts
    def forward(self, x):
        # We bypass this in T-TRM loop but CMRModel demands it for constructor
        batch_size = x.shape[0]
        return torch.sigmoid(torch.zeros(batch_size, self.n_concepts)), torch.zeros(batch_size, self.emb_size)

def train_trm_pipeline(args):
    print("--------------------------------------------------")
    print("Stage 4: T-TRM Loop and Rule-Memory Optimization")
    print("--------------------------------------------------")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load cached activations
    if not os.path.exists(args.cache_dir):
        raise FileNotFoundError(f"Activation cache directory '{args.cache_dir}' does not exist.")
        
    chunk_files = sorted([os.path.join(args.cache_dir, f) for f in os.listdir(args.cache_dir) if f.endswith(".safetensors")])
    if len(chunk_files) == 0:
        raise FileNotFoundError(f"No cached activation chunk files found in '{args.cache_dir}'.")
        
    print(f"Loading activations from {len(chunk_files)} chunk files...")
    activations_list = []
    for f in chunk_files:
        chunk_data = load_file(f)
        activations_list.append(chunk_data["activations"])
        
    activations = torch.cat(activations_list, dim=0)
    print(f"Loaded activations shape: {activations.shape}")
    
    # 2. Get target concepts via HybridCBM
    emb_dim = activations.shape[-1]
    
    # Instantiate HybridCBM to compute concept targets
    # If a checkpoint is provided, load it. Otherwise, use random projections.
    hybrid_cbm = HybridCBM(n_dynamic=args.n_dynamic, clip_dim=args.clip_dim, emb_dim=emb_dim).to(device)
    if args.hybrid_cbm_path and os.path.exists(args.hybrid_cbm_path):
        print(f"Loading trained HybridCBM weights from {args.hybrid_cbm_path}")
        hybrid_cbm.load_state_dict(torch.load(args.hybrid_cbm_path, map_location=device))
    else:
        print("Warning: No pre-trained HybridCBM checkpoint found or specified. Running with initialized weights.")
        
    # Project activations to obtain target concepts in mini-batches to prevent CUDA OOM
    hybrid_cbm.eval()
    c_probs_list = []
    batch_size = 4096
    
    with torch.no_grad():
        model_dtype = hybrid_cbm.proj_clip.weight.dtype
        for i in range(0, activations.shape[0], batch_size):
            batch_x = activations[i:i+batch_size].to(device=device, dtype=model_dtype)
            z, _, _ = hybrid_cbm(batch_x)
            # Map z from [-1, 1] to concept probability targets in [0, 1]
            c_probs_batch = (z + 1.0) / 2.0
            c_probs_list.append(c_probs_batch.float().cpu())
            
        c_probs = torch.cat(c_probs_list, dim=0)
        
    n_concepts = c_probs.shape[-1]
    print(f"Computed target concepts shape: {c_probs.shape} (Static + Dynamic: {n_concepts})")
    
    # 3. Create downstream task labels (y)
    # For Alpaca instruction dataset, we synthesize a binary task label based on concept activation profiles
    # Task 1: is the first concept active?
    # Task 2: is the second concept active?
    y_task1 = (c_probs[:, 0] > 0.5).float().unsqueeze(1)
    y_task2 = (c_probs[:, 1] > 0.5).float().unsqueeze(1)
    y_tasks = torch.cat([y_task1, y_task2], dim=1)
    n_tasks = y_tasks.shape[-1]
    print(f"Synthesized downstream tasks shape: {y_tasks.shape}")
    
    # Subsetting for debug mode if specified
    if args.debug:
        print("Debug mode active: subsetting to 10 samples.")
        activations = activations[:10]
        c_probs = c_probs[:10]
        y_tasks = y_tasks[:10]
        args.epochs = 5
        
    # Create Dataloader
    dataset = TensorDataset(activations, c_probs, y_tasks)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=(device == "cuda")
    )
    
    # 4. Instantiate CMR and TTRMLoop
    emb_size = 128
    rule_emb_size = 64
    concept_names = hybrid_cbm.concepts + [f"dynamic_{i}" for i in range(args.n_dynamic)]
    
    dummy_encoder = DummyEncoder(emb_size, n_concepts)
    
    print("Initializing CMR Model...")
    cmr_model = CMR(
        encoder=dummy_encoder,
        emb_size=emb_size,
        rule_emb_size=rule_emb_size,
        n_tasks=n_tasks,
        n_rules=args.n_rules,
        n_concepts=n_concepts,
        concept_names=concept_names,
        selector_input=InputTypes.concepts
    ).to(device)
    
    print("Initializing TTRMLoop...")
    t_trm = TTRMLoop(
        d_model=emb_dim,
        n_concepts=n_concepts,
        n_tasks=n_tasks,
        n_rules=args.n_rules,
        n_latent=args.n_latent,
        n_layers=2
    ).to(device)
    
    # 5. Joint optimization
    optimizer = torch.optim.Adam(
        list(t_trm.parameters()) + list(cmr_model.parameters()), 
        lr=args.lr
    )
    
    # Loss weights
    w_y = 1.0       # Task classification loss weight
    w_c = 0.5       # Concept prediction loss weight
    w_commit = 0.1  # Gate commitment weight (hardening)
    w_nm = 0.1      # Numerical Monotonicity regularizer weight
    w_im = 0.1      # Information Monotonicity regularizer weight
    w_route = 0.1   # Routing commitment weight
    
    print("Starting training loop...")
    t_trm.train()
    cmr_model.train()
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_loss_y = 0.0
        epoch_loss_c = 0.0
        
        for batch_x, batch_c, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_c = batch_c.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass through T-TRM loop (which evaluates CMR rules internally)
            y_steering, z_latent, c_pred, y_pred = t_trm(
                batch_x, 
                cmr_model, 
                T_loops=3, 
                n_steps=2, 
                hard=False
            )
            
            # 1. Downstream task loss
            loss_y = F.binary_cross_entropy(y_pred.float(), batch_y.float())
            
            # 2. Concept prediction loss
            loss_c = F.binary_cross_entropy(c_pred.float(), batch_c.float())
            
            # 3. T-TRM regularizations
            commit, nm, im, route_commit = t_trm.compute_losses()
            
            # Combined Loss
            total_loss = (
                w_y * loss_y + 
                w_c * loss_c + 
                w_commit * commit + 
                w_nm * nm + 
                w_im * im + 
                w_route * route_commit
            )
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_loss_y += loss_y.item()
            epoch_loss_c += loss_c.item()
            
        avg_loss = epoch_loss / len(dataloader)
        avg_loss_y = epoch_loss_y / len(dataloader)
        avg_loss_c = epoch_loss_c / len(dataloader)
        
        if (epoch + 1) % max(1, args.epochs // 10) == 0 or args.debug:
            print(f"Epoch {epoch+1:03d}/{args.epochs:03d} | Loss: {avg_loss:.4f} | Task Loss: {avg_loss_y:.4f} | Concept Loss: {avg_loss_c:.4f}")
            
    print("Training complete!")
    
    # 6. Save model checkpoints
    os.makedirs(args.output_dir, exist_ok=True)
    t_trm_path = os.path.join(args.output_dir, "t_trm_loop.pt")
    cmr_path = os.path.join(args.output_dir, "cmr_model.pt")
    
    torch.save(t_trm.state_dict(), t_trm_path)
    torch.save(cmr_model.state_dict(), cmr_path)
    print(f"Saved T-TRM Loop checkpoint to {t_trm_path}")
    print(f"Saved CMR Model checkpoint to {cmr_path}")
    
    # 7. Evaluate on a hardened discrete circuit
    print("Evaluating logic under hardened discrete configuration...")
    t_trm.eval()
    cmr_model.eval()
    with torch.no_grad():
        # Evaluate first batch
        eval_x, eval_c, eval_y = next(iter(dataloader))
        eval_x = eval_x.to(device)
        eval_y = eval_y.to(device)
        
        _, _, c_pred_hard, y_pred_hard = t_trm(
            eval_x, 
            cmr_model, 
            T_loops=3, 
            n_steps=2, 
            hard=True
        )
        
        hard_acc_y = ((y_pred_hard > 0.5).float() == eval_y).float().mean().item()
        print(f"Hardened Discrete Task Prediction Accuracy (Sample Batch): {hard_acc_y * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4: T-TRM Loop & Rule Hardening Joint Training")
    parser.add_index = False
    parser.add_argument("--cache_dir", type=str, default="./cached_activations", help="Directory with cached activations")
    parser.add_argument("--hybrid_cbm_path", type=str, default="./hybrid_cbm.pt", help="Path to trained HybridCBM checkpoint")
    parser.add_argument("--output_dir", type=str, default="./t_trm_outputs", help="Directory to save output checkpoints")
    parser.add_argument("--n_dynamic", type=int, default=5, help="Number of dynamic concepts in HybridCBM")
    parser.add_argument("--clip_dim", type=int, default=512, help="Dimensionality of CLIP projection space")
    parser.add_argument("--n_latent", type=int, default=8, help="Number of latent variables in T-TRM loop")
    parser.add_argument("--n_rules", type=int, default=10, help="Number of decision rules inside CMR Decider")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of worker processes for data loading")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with minimal samples")
    
    args = parser.parse_args()
    train_trm_pipeline(args)
