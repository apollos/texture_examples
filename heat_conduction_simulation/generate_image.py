import cv2
import numpy as np
import os

dim=1024
channel = 4
files = ["linear_result_file.dmp", "texture_result_file.dmp"]
for file_name in files:
    if os.path.exists(file_name):
        with open(file_name, "rb") as f:
            content = f.read()
    
        rgba = np.zeros((dim, dim, channel), dtype=np.uint8)
        for x in range(int(len(content)/channel)):
            rgba[int(x/dim)][int(x%dim)] = np.frombuffer(content[x*channel:(x+1)*channel], dtype=np.uint8)
        # Convert RGBA to BGRA
        BGRA = rgba[..., [2,1,0,3]]
        cv2.imwrite(file_name+".png", BGRA)

