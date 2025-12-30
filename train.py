import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import numpy as np
import sys
import logging

logging.basicConfig(level=logging.INFO)

print(f"Python: {sys.executable}")
print(f"Torch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import HybridTRM
from data import get_loader

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    augment = not args.no_augment
    print(f"Data Augmentation: {augment}")
    train_loader = get_loader(args.data_path, batch_size=args.batch_size, augment=augment)
    
    # Model
    model = HybridTRM(dim=args.dim, layers=args.layers).to(device)
    
    # Optimize for H100
    # Optimize for H100
    if not args.no_compile and torch.cuda.get_device_capability()[0] >= 7: # Volta or newer
        print("Enabling torch.compile() for H100 Acceleration...")
        # Enable logs so user sees progress
        torch._logging.set_logs(inductor=logging.INFO)
        model = torch.compile(model)
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        try:
            state_dict = torch.load(args.resume, map_location=device)
            model.load_state_dict(state_dict)
            print("Successfully loaded checkpoint weights.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Losses
    ce_loss_fn = nn.CrossEntropyLoss(reduction='none') 
    bce_loss_fn = nn.BCEWithLogitsLoss()
    
    scaler = torch.cuda.amp.GradScaler()

    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        total_loss = 0
        total_acc = 0
        
        for batch in pbar:
            if batch is None: continue
            
            x = batch['query_input'].to(device) # (B, H, W)
            target = batch['target'].to(device) # (B, H, W)
            mask = batch['target_mask'].to(device) # (B, H, W)
            
            optimizer.zero_grad()
            
            # Forward
            support_pairs = batch.get('support', None) # (B, S, 2, H, W)
            
            if support_pairs is not None:
                support_pairs = support_pairs.to(device)
                
            # 1. Encode Support (Context Priming) - ONCE per batch (invariant context)
            z_global_init = None
            if support_pairs is not None:
                with torch.cuda.amp.autocast():
                    z_global_init = model.encode_support(support_pairs)
            
            # --- IMPL: Multi-View Consistency Loss (Omnidirectional) ---
            # Instead of just one forward pass, we do 4 (0, 90, 180, 270)
            # This stabilizes gradients by averaging out orientation noise.
            
            views_loss = 0
            final_acc_step = 0
            
            # We will accumulate gradients over 4 views
            # Or simpler: compute total loss then backward once (autograd handles it)
            
            for k in [0, 1, 2, 3]: # 4 Rotations
                with torch.cuda.amp.autocast():
                    # Rotate Input and Target k times (90 degrees)
                    x_rot = torch.rot90(x, k, [1, 2])
                    target_rot = torch.rot90(target, k, [1, 2])
                    mask_rot = torch.rot90(mask, k, [1, 2])
                    
                    # Context: Ideally context is invariant. 
                    # If we rotate input, does context change? 
                    # Our 'encode_support' is NOT rotation invariant unless providing rotated support.
                    # For now, we assume z_global_init is a "concept" and reuse it. 
                    # (Refinement: We could rotate z_global_init? No, it's latent).
                    
                    y_preds, critic_scores = model(x_rot, initial_state=z_global_init) 
                
                steps = len(y_preds)
                view_loss = 0
                
                for i, (logits, c_logit) in enumerate(zip(y_preds, critic_scores)):
                    # 1. Reconstruction Loss (CrossEntropy)
                    ce = ce_loss_fn(logits, target_rot.long()) # (B, H, W)
                    ce = (ce * mask_rot).sum() / (mask_rot.sum() + 1e-6)
                    
                    step_weight = (i + 1) / steps
                    view_loss += ce * step_weight
                    
                    # 2. Critic Loss
                    pred_grid = logits.argmax(dim=1) # (B, H, W)
                    correct_pixels = (pred_grid == target_rot.long()).float() * mask_rot
                    
                    item_acc = correct_pixels.sum(dim=(1, 2)) / (mask_rot.sum(dim=(1, 2)) + 1e-6) 
                    c_target = (item_acc > 0.99).float().unsqueeze(1)
                    
                    bce = bce_loss_fn(c_logit, c_target)
                    view_loss += bce * 0.1 
                    
                    if i == steps - 1 and k == 0: # Log acc for canonical view
                        final_acc_step = item_acc.mean().item()
                
                # Sum up view losses
                views_loss += view_loss

            # Average loss over 4 views
            total_view_loss = views_loss / 4.0
            
        # Backward with Scaling
            scaler.scale(total_view_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += total_view_loss.item()
            total_acc += final_acc_step
            pbar.set_postfix({'loss': total_view_loss.item(), 'acc': final_acc_step})
            
        avg_loss = total_loss/len(train_loader)
        avg_acc = total_acc/len(train_loader)
        
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f} Avg Acc: {avg_acc:.4f}")
        
        # Step Scheduler
        scheduler.step(avg_loss)
        
        # Log current LR
        current_lr = optimizer.param_groups[0]['lr']
        # print(f"Current LR: {current_lr}") # Built-in verbose=True handles print, but we can be explicit if needed.
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"checkpoint_{epoch+1}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/training')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--no_compile', action='store_true', help="Disable torch.compile")
    
    # ComfyUI-style arguments
    parser.add_argument('--windows-standalone-build', action='store_true', help="Windows standalone build flag")
    parser.add_argument('--cuda_device', type=int, default=None, help="Set cuda device")
    parser.add_argument('--dont-print-server', action='store_true', help="Don't print server log")
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument('--no_augment', action='store_true', help="Disable data augmentation")
    
    args = parser.parse_args()

    # --- ComfyUI Environment Setup Mimic ---
    if os.name == "nt":
        os.environ['MIMALLOC_PURGE_DELAY'] = '0'
        
    os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '1'
    
    if args.cuda_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
        os.environ['HIP_VISIBLE_DEVICES'] = str(args.cuda_device)
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.cuda_device)
        print(f"Set cuda device to: {args.cuda_device}")

    # Create dummy data if missing (Legacy logic)
    if not os.path.exists(args.data_path):
        print(f"Creating dummy data at {args.data_path} for testing...")
        os.makedirs(args.data_path, exist_ok=True)
        import json
        dummy = {
            "train": [{"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]}],
            "test": [{"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]}]
        }
        with open(os.path.join(args.data_path, 'dummy.json'), 'w') as f:
            json.dump(dummy, f)
            
    train(args)
