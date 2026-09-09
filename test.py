import os
import shutil


files = os.listdir("dataset/testset/mask_0")
for f in files:
    idx = f.split("_")[0]
    shutil.move(f"dataset/testset/mask_0/{f}", f"dataset/testset/mask_0/{idx}.png")