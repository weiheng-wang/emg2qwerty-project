import sys
import torch
import glob

folder = sys.argv[1]
try:
    ckpt_path = glob.glob(f"{folder}/checkpoints/epoch=*.ckpt")[0]
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    for k, v in ckpt.get("callbacks", {}).items():
        if "ModelCheckpoint" in k:
            score = v.get("best_model_score").item()
            print(f"[{folder}] Best Validation CER: {score:.4f} ({score * 100:.2f}%)")
except IndexError:
    print(f"Could not find a valid checkpoint in {folder}/checkpoints/")
