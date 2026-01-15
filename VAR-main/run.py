"""
Inference Script for VAR-Scaling
Paper: Inference-Time Scaling for Visual AutoRegressive Modeling by Searching Representative Samples
"""

import os
import os.path as osp
import torch
import random
import numpy as np
import argparse
from tqdm import tqdm
import PIL.Image as PImage

# Disable default parameter init for faster speed
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

from models import build_vae_var

def parse_args():
    parser = argparse.ArgumentParser(description="VAR-Scaling Inference Script")
    
    # === Path Arguments (Required) ===
    parser.add_argument("--vae_ckpt", type=str, required=True, help="Path to VAE checkpoint")
    parser.add_argument("--var_ckpt", type=str, required=True, help="Path to VAR checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated images")
    
    # === Model Arguments ===
    parser.add_argument("--model_depth", type=int, default=30, choices=[16, 20, 24, 30], help="Depth of the VAR model")
    
    # === Inference Arguments ===
    parser.add_argument("--num_images_per_class", type=int, default=50, help="Number of images to generate per class")
    parser.add_argument("--cfg", type=float, default=1.0, help="Classifier-free guidance scale (Paper recommends 1.0 for scaling verification)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--more_smooth", action="store_true", help="Enable smoothing for visualization (disable for FID metrics)")
    
    # === VAR-Scaling Hyperparameters (The Core Contribution) ===
    parser.add_argument("--enable_scaling", action="store_true", default=True, help="Enable VAR-Scaling strategy")
    parser.add_argument("--scaling_scale", type=int, default=1, help="The scale level to apply scaling (Paper recommends 1 for VAR)")
    parser.add_argument("--scaling_samples", type=int, default=50, help="Number of candidate samples (N) for search")
    parser.add_argument("--scaling_alpha", type=float, default=2.3, help="Density threshold coefficient (alpha). Adjust based on model depth.")
    parser.add_argument("--scaling_rep_num", type=int, default=10, help="Number of representative samples to keep (k)")
    parser.add_argument("--conflict_thresh", type=float, default=1.8, help="Threshold for conflict detection to maintain diversity")

    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup Device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Set Seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Optimization
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')

    # Ensure Output Directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build Models
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=args.model_depth, shared_aln=False,
    )

    # Load Checkpoints
    print(f"Loading VAE from: {args.vae_ckpt}")
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location='cpu'), strict=True)
    
    print(f"Loading VAR from: {args.var_ckpt}")
    var.load_state_dict(torch.load(args.var_ckpt, map_location='cpu'), strict=True)
    
    vae.eval(), var.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)
    
    print(f"Model loaded successfully. Starting inference with VAR-Scaling...")
    print(f"Config: Scale={args.scaling_scale}, N={args.scaling_samples}, Alpha={args.scaling_alpha}")

    # Generate Loop (ImageNet-50k Standard)
    # Loop over all 1000 ImageNet classes
    for class_label in tqdm(range(1000), desc="Generating Classes"):
        # Create labels for the batch
        class_labels = [class_label] * args.num_images_per_class
        B = len(class_labels)
        label_B = torch.tensor(class_labels, device=device)
        
        with torch.inference_mode():
            with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
                # Inference with VAR-Scaling
                recon_B3HW = var.autoregressive_infer_cfg(
                    B=B, 
                    label_B=label_B, 
                    cfg=args.cfg,           
                    top_k=900,         # Default VAR setting
                    top_p=0.96,        # Default VAR setting
                    g_seed=args.seed, 
                    more_smooth=args.more_smooth,
                    # --- VAR-Scaling Parameters ---
                    enable_var_scaling=args.enable_scaling,
                    scaling_scale=args.scaling_scale,      
                    scaling_samples=args.scaling_samples,    
                    scaling_alpha=args.scaling_alpha,     
                    scaling_rep_num=args.scaling_rep_num,    
                    conflict_thresh=args.conflict_thresh    
                )
        
        # Post-process and Save
        recon_B3HW = recon_B3HW.float().cpu()  
        
        for i in range(B):
            img_tensor = recon_B3HW[i].permute(1, 2, 0)  
            img_tensor = img_tensor * 255                
            img_np = img_tensor.numpy().astype(np.uint8) 
            img = PImage.fromarray(img_np)              
            
            # Save format: class_{id}_img_{index}.png
            # This naming convention is common for FID calculation tools
            save_path = osp.join(args.output_dir, f'class_{class_label}_img_{i}.png')
            img.save(save_path)

    print(f'\nAll images saved to {args.output_dir}')

if __name__ == "__main__":
    main()