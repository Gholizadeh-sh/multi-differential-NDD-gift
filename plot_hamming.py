"""Rebuild Figure 2 from held-out-test Hamming histograms, with exact row labels."""
import argparse
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("results",type=Path)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--out",type=Path,default=Path("hamming_corrected.pdf"))
    a=p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,6),sharex=True,sharey=True)
    for col,r in enumerate((6,7,8)):
        with np.load(a.results/f"r{r}_1_A"/f"seed_{a.seed}"/"hamming_histograms.npz",allow_pickle=False) as f:
            counts,ds=f["counts"],f["deltas_hex"]
        for row in range(2):
            ax=axes[row,col]
            for label,title,color in [(0,"Random","#64748b"),(1,"Structured","#087e8b")]:
                c=counts[label,row]
                ax.plot(np.arange(65),c/c.sum(),label=title,color=color,lw=1.7)
            ax.set_title(f"{r} rounds, delta = {ds[row]}",fontsize=9)
            ax.set_xlabel("Ciphertext-XOR Hamming weight")
            if col==0: ax.set_ylabel("Relative frequency")
            ax.grid(alpha=.18)
    axes[0,0].legend(frameon=False)
    fig.tight_layout()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.out,bbox_inches="tight")
    print(a.out)


if __name__=="__main__":main()
