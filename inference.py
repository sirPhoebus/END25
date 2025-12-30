import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys

print(f"Python: {sys.executable}")
print(f"Torch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import HybridTRM
from data import get_loader

def benchmark(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check model
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint {args.checkpoint} not found.")
        return

    # Data (Test Mode = No Augmentation)
    test_loader = get_loader(args.data_path, batch_size=args.batch_size, mode='test', augment=False)
    
    # Model
    model = HybridTRM(dim=args.dim, layers=args.layers).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    
    print(f"Benchmarking on {len(test_loader.dataset)} tasks...")
    
    total_tasks = 0
    solved_tasks = 0
    total_acc = 0.0
    
    # Store results
    results = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            x = batch['query_input'].to(device) # (B, H, W)
            target = batch['target'].to(device) # (B, H, W)
            mask = batch['target_mask'].to(device) # (B, H, W)
            
            support_pairs = batch.get('support', None)
            if support_pairs is not None:
                support_pairs = support_pairs.to(device)
            
            # Forward with Context
            z_global_init = None
            if support_pairs is not None:
                z_global_init = model.encode_support(support_pairs)
                
            y_preds, critic_scores = model(x, initial_state=z_global_init) 
            
            # Smart Inference: Use Critic?
            # For now, let's look at the FINAL step prediction.
            # OR simple logic: Pick step with max critic score?
            
            B = x.shape[0]
            
            # Collect final predictions
            # Stack preds: (Steps, B, C, H, W)
            stack_preds = torch.stack(y_preds)
            stack_critic = torch.stack(critic_scores).squeeze(-1) # (Steps, B)
            
            # For each batch item, pick best step based on critic
            best_steps = stack_critic.argmax(dim=0) # (B,)
            
            final_pred_grids = []
            
            for b in range(B):
                step = best_steps[b] if args.use_critic else -1
                logits = stack_preds[step, b] # (C, H, W)
                pred_grid = logits.argmax(dim=0) # (H, W)
                final_pred_grids.append(pred_grid)
            
            final_pred_grids = torch.stack(final_pred_grids) # (B, H, W)
            
            # Calculate Accuracy
            # Exact match on masked pixels
            correct_pixels = (final_pred_grids == target.long()).float() * mask
            
            # Per sample accuracy
            # (B,)
            acc = correct_pixels.sum(dim=(1,2)) / (mask.sum(dim=(1,2)) + 1e-6)
            
            # Solved = Acc > 0.99 (Implicit "Exact Match" for ARC)
            is_solved = (acc > 0.99).float()
            
            solved_tasks += is_solved.sum().item()
            total_acc += acc.sum().item()
            total_tasks += B
            
    print("-" * 30)
    print(f"Benchmark Results:")
    print(f"Total Tasks: {total_tasks}")
    if total_tasks > 0:
        print(f"Solved: {solved_tasks} ({solved_tasks/total_tasks*100:.2f}%)")
        print(f"Avg Pixel Acc: {total_acc/total_tasks*100:.2f}%")
    else:
        print("Solved: 0 (0.00%)")
        print("Avg Pixel Acc: 0.00%")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument('--data_path', type=str, default='data/evaluation')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--use_critic', action='store_true', help="Use critic score to select best step, else use final step")
    
    # Comfy/Env args
    parser.add_argument('--cuda_device', type=int, default=None)
    
    args = parser.parse_args()
    
    if args.cuda_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
        
    benchmark(args)
