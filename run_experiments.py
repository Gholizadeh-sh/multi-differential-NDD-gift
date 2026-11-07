"""Run every experimental configuration in Tables III--VI, with fixed settings."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import unittest
from data import generate, save_dataset, DEFAULT_KEY
from train import train_dataset

PAPER = [
    ("r4_A_1",4,[0xA,0x1],"III"),
    ("r6_1_A",6,[0x1,0xA],"IV,V"),
    ("r7_1_A",7,[0x1,0xA],"IV"),
    ("r8_1_A",8,[0x1,0xA],"IV"),
    ("r6_AAAA_F0F0",6,[0xAAAAAAAAAAAAAAAA,0xF0F0F0F0F0F0F0F0],"V"),
    ("r6_0101_FFFF",6,[0x0101010101010101,0xFFFFFFFF00000000],"V"),
    ("r6_weak4",6,[0x0101010101010101,0xFFFFFFFF00000000,0xAAAAAAAAAAAAAAAA,0xF0F0F0F0F0F0F0F0],"VI"),
    ("r6_mixed4",6,[0x0101010101010101,0xFFFFFFFF00000000,0x1,0xA],"VI"),
]
CONTROLS = [
    ("r6_single_1",6,[0x1],"additional single-row control"),
    ("r6_single_A",6,[0xA],"additional single-row control"),
    ("r28_1_A",28,[0x1,0xA],"additional full-round null diagnostic"),
]


def export_summary(out, configurations, seeds):
    rows=[]
    for name,rounds,deltas,table in configurations:
        metrics=[json.loads((out/name/f"seed_{s}"/"metrics.json").read_text()) for s in seeds]
        acc=[m["accuracy"] for m in metrics]
        auc=[m["roc_auc"] for m in metrics]
        rows.append({"configuration":name,"paper_table":table,"rounds":rounds,
                     "deltas_hex":";".join(f"{d:016x}" for d in deltas),
                     "n_total":metrics[0]["dataset"]["n_total"],"repeats":len(seeds),
                     "accuracy_mean":statistics.mean(acc),
                     "accuracy_sample_std":statistics.stdev(acc) if len(acc)>1 else None,
                     "auc_mean":statistics.mean(auc),
                     "auc_sample_std":statistics.stdev(auc) if len(auc)>1 else None})
    with (out/"summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/"summary.json").write_text(json.dumps(rows,indent=2)+"\n")
    lines=["% Generated from actual completed runs. Single runs do not estimate run variance.",
           r"\begin{tabular}{lrrr}",r"Configuration & Rounds & Accuracy (\%) & ROC-AUC \\",r"\hline"]
    for row in rows:
        label=row["configuration"].replace("_",r"\_")
        lines.append(f"{label} & {row['rounds']} & {100*row['accuracy_mean']:.2f} & {row['auc_mean']:.4f} \\\\")
    lines.append(r"\end{tabular}")
    (out/"results_table.tex").write_text("\n".join(lines)+"\n")
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True)
    p.add_argument("--n",type=int,default=2**20,help="TOTAL per dataset")
    p.add_argument("--seeds",type=int,nargs="+",default=[0])
    p.add_argument("--epochs",type=int,default=15)
    p.add_argument("--key",type=lambda s:int(s,0),default=DEFAULT_KEY)
    p.add_argument("--device",default="auto")
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--negative",choices=["encrypted-pairs","uniform"],default="encrypted-pairs")
    p.add_argument("--include-controls",action="store_true")
    p.add_argument("--keep-datasets",action="store_true")
    p.add_argument("--resume",action="store_true")
    a=p.parse_args()
    if len(set(a.seeds))!=len(a.seeds): p.error("seeds must be distinct")
    configurations=PAPER+(CONTROLS if a.include_controls else [])
    manifest={"n_total":a.n,"seeds":a.seeds,"epochs":a.epochs,"batch_size":128,
              "key_hex":f"{a.key:032x}","negative":a.negative,"split_seed":42,
              "configurations":[[n,r,ds,t] for n,r,ds,t in configurations]}
    manifest["code_sha256"]={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                             for name in ("gift64.py","data.py","train.py","run_experiments.py")}
    out=Path(a.out)
    if out.exists():
        if not a.resume: raise FileExistsError("Output exists; use --resume or a fresh directory")
        if json.loads((out/"run_manifest.json").read_text())!=manifest:
            raise ValueError("Resume settings differ from the existing manifest")
    else:
        out.mkdir(parents=True)
        (out/"run_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    # Correctness gate before any experiment: no dependency on pytest or a network.
    tests=unittest.defaultTestLoader.discover(str(Path(__file__).parent/"tests"))
    if not unittest.TextTestRunner(verbosity=1).run(tests).wasSuccessful():
        raise SystemExit("Correctness gate failed")
    for name,rounds,deltas,_ in configurations:
        for seed in a.seeds:
            dest=out/name/f"seed_{seed}"
            completed=dest/"metrics.json"
            if completed.exists() and a.resume:
                m=json.loads(completed.read_text());md=m["dataset"]
                if (md["n_total"],md["data_seed"],md["cipher_rounds"],md["key_hex"],md["negative"],md["deltas_hex"],m["epochs"],m["train_seed"]) != (
                    a.n,seed,rounds,f"{a.key:032x}",a.negative,[f"{d:016x}" for d in deltas],a.epochs,seed):
                    raise ValueError(f"Completed run does not match manifest: {dest}")
                print(f"Skipping complete run: {dest}",flush=True);continue
            if dest.exists():
                raise FileExistsError(f"Incomplete run at {dest}; move it aside and rerun --resume")
            dataset=out/name/f"dataset_seed_{seed}.npz"
            if not dataset.exists():
                print(f"Generating {name}, seed {seed}, N={a.n}",flush=True)
                save_dataset(dataset,*generate(a.n,deltas,a.key,rounds,seed,a.negative))
            # train.py verifies the archive fingerprint; check its intended settings as well.
            with __import__('numpy').load(dataset,allow_pickle=False) as f:
                md=json.loads(str(f["metadata"]))
            if (md["n_total"],md["data_seed"],md["cipher_rounds"],md["key_hex"],md["negative"],md["deltas_hex"]) != (
                a.n,seed,rounds,f"{a.key:032x}",a.negative,[f"{d:016x}" for d in deltas]):
                raise ValueError(f"Dataset settings mismatch: {dataset}")
            train_dataset(dataset,dest,seed=seed,epochs=a.epochs,device=a.device,threads=a.threads)
            if not a.keep_datasets: dataset.unlink()
    export_summary(out,configurations,a.seeds)
    print(f"Completed. See {out/'summary.csv'} and {out/'results_table.tex'}",flush=True)


if __name__=="__main__": main()
