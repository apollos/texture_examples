import cv2
import numpy as np

with open("file.txt", "rb") as f:
    content = f.read()

dim=1024
channel = 4
rgba = np.zeros((dim, dim, channel), dtype=np.uint8)

for x in range(int(len(content)/channel)):
    rgba[int(x/dim)][int(x%dim)] = np.frombuffer(content[x*channel:(x+1)*channel], dtype=np.uint8)
# Convert RGBA to BGRA
BGRA = rgba[..., [2,1,0,3]]
cv2.imwrite("generate.png", BGRA)
