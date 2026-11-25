"""Make the inspection figure from a completed result directory."""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

out = Path(sys.argv[1])
root = Path(__file__).resolve().parents[3]
cams = json.loads((out / "cameras.json").read_text())
pairs = json.loads((out / "pairs.json").read_text())
fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), layout="constrained")
colors = {"train":"#236890", "val":"#bb5a1e", "test":"#308155"}
for ax, (split, color) in zip(axes[0], colors.items()):
    c = next(c for c in cams if c["path"].startswith(split+"/"))
    with Image.open(root/"data/raw/nerf_synthetic/lego"/c["path"]) as im:
        white = Image.new("RGBA", im.size, "white")
        white.alpha_composite(im)
        ax.imshow(white.convert("RGB"))
    ax.plot(399.5,399.5,marker="+",color="#ec3162",markersize=14,markeredgewidth=1.4)
    ax.set_title(c["path"], color=color)
    ax.set_xlabel("C = (" + ", ".join(f"{v:.2f}" for v in c["C"]) + ")")
    ax.set_xticks([])
    ax.set_yticks([])
for ax, (u,v), title in zip(axes[1,:2], [(0,1),(0,2)], ["Vị trí camera nhìn từ trên","Vị trí camera nhìn ngang"]):
    for split, color in colors.items():
        group = [c for c in cams if c["path"].startswith(split+"/")]
        C = np.array([c["C"] for c in group])
        ax.scatter(C[:,u],C[:,v],s=9,color=color,alpha=.69,label=split)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("XYZ"[u]); ax.set_ylabel("XYZ"[v])
    ax.grid(alpha=.18); ax.set_title(title)
axes[1,0].legend(fontsize=8)
ax = axes[1,2]
for key,label,color in [("median","Đúng pose","#236890"),("ctrl_median","Gán sai pose","#b65c2a")]:
    vals = np.sort([p[key] for p in pairs])
    ax.plot(vals,np.arange(1,len(vals)+1)/len(vals),label=label,color=color)
ax.set_xscale("log"); ax.set_ylim(0,1.05); ax.set_xlabel("Sai số trung vị từng cặp (px)")
ax.set_ylabel("Tỷ lệ tích lũy"); ax.legend(); ax.grid(alpha=.18); ax.set_title("Đối chiếu trên ảnh thật")
fig.suptitle("NeRF-Synthetic Lego · 400 frame và camera",fontsize=17)
fig.savefig(out/"camera_checks.png",dpi=155)

