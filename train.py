"""Paper MLP: 64k -> 128 ReLU -> 64 ReLU -> 2 logits; one final test."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from data import SCHEMA, fingerprint


def load_dataset(path):
    with np.load(path, allow_pickle=False) as f:
        if not {"X", "y", "positive_orbit_ids", "metadata"}.issubset(f.files):
            raise ValueError("Legacy/unverified dataset: regenerate it with corrected data.py")
        X, y, groups = f["X"], f["y"], f["positive_orbit_ids"]
        meta = json.loads(str(f["metadata"]))
    if meta.get("schema") != SCHEMA:
        raise ValueError("Unsupported dataset schema")
    for field, filename in [("implementation_sha256", "gift64.py"), ("generator_sha256", "data.py")]:
        current = hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest()
        if meta.get(field) != current:
            raise ValueError(f"Dataset was generated with a different {filename}; regenerate or use its exact recorded source")
    expected = meta.pop("dataset_sha256")
    if fingerprint(X, y, groups, meta) != expected:
        raise ValueError("Dataset content/metadata fingerprint mismatch")
    meta["dataset_sha256"] = expected
    if X.dtype != np.uint8 or X.ndim != 3 or X.shape[-1] != 64:
        raise ValueError("Expected uint8 X with shape (N,k,64)")
    if X.shape != tuple(meta["shape"]) or len(y) != len(X) or len(groups) != len(X):
        raise ValueError("Inconsistent dataset dimensions")
    if not np.all((X == 0) | (X == 1)) or set(np.unique(y)) != {0, 1}:
        raise ValueError("Dataset is not binary")
    return X, y, groups, meta


def split_indices(y, groups, split_seed=42):
    train, test = train_test_split(np.arange(len(y)), test_size=0.3,
                                   random_state=split_seed, stratify=y)
    left = groups[train[y[train] == 1]]
    right = groups[test[y[test] == 1]]
    if np.intersect1d(left, right).size:
        raise ValueError("Related positive plaintext orbits cross train/test; regenerate with a new data seed")
    return train, test


def train_dataset(dataset, out, seed=0, split_seed=42, epochs=15,
                  batch_size=128, device="auto", threads=2):
    if epochs < 1 or batch_size < 1 or threads < 1 or seed < 0:
        raise ValueError("Invalid training arguments")
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite run {out}")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import sklearn
    from torch import nn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    X, y, groups, metadata = load_dataset(dataset)
    train_idx, test_idx = split_indices(y, groups, split_seed)
    out.mkdir(parents=True)
    np.savez(out/"split_indices.npz", train=train_idx, test=test_idx)
    tensor_data = torch.utils.data.TensorDataset(
        torch.from_numpy(X.reshape(len(X), -1)), torch.from_numpy(y.astype(np.int64)))
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tensor_data, train_idx.tolist()),
        batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    model = nn.Sequential(nn.Linear(64*X.shape[1], 128), nn.ReLU(),
                          nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2)).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    history = []
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(dev, dtype=torch.float32), yb.to(dev)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()*len(yb)
        entry = {"epoch": epoch+1, "train_loss": loss_sum/len(train_idx)}
        history.append(entry)
        print(json.dumps(entry), flush=True)
    model.eval()
    scores, predicted = [], []
    with torch.no_grad():
        for start_idx in range(0, len(test_idx), 4096):
            indices = test_idx[start_idx:start_idx+4096]
            logits = model(torch.from_numpy(X[indices].reshape(len(indices), -1)).to(dev, dtype=torch.float32))
            scores.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            predicted.append(logits.argmax(dim=1).cpu().numpy())
    score, pred, truth = np.concatenate(scores), np.concatenate(predicted), y[test_idx]
    acc = float(accuracy_score(truth, pred))
    # Wilson interval: uncertainty conditional on this test set protocol, not run-to-run variance.
    z, ntest = 1.959963984540054, len(truth)
    center = (acc + z*z/(2*ntest))/(1+z*z/ntest)
    margin = z*np.sqrt(acc*(1-acc)/ntest+z*z/(4*ntest*ntest))/(1+z*z/ntest)
    metrics = {
        "accuracy": acc, "roc_auc": float(roc_auc_score(truth, score)),
        "confusion_matrix_labels_0_1": confusion_matrix(truth, pred, labels=[0,1]).tolist(),
        "accuracy_wilson95_conditional": [center-margin, center+margin],
        "n_train": len(train_idx), "n_test": ntest, "train_seed": seed,
        "split_seed": split_seed, "epochs": epochs, "batch_size": batch_size,
        "learning_rate": 1e-3, "hidden_units": [128,64], "device": str(dev),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "elapsed_training_and_test_seconds": time.perf_counter()-start,
        "dataset": metadata,
        "software": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "torch": torch.__version__, "scikit_learn": sklearn.__version__,
                     "platform": platform.platform()},
        "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out/"history.json").write_text(json.dumps(history, indent=2)+"\n")
    np.savez(out/"predictions.npz", test_indices=test_idx, y_true=truth,
             y_pred=pred, y_score=score)
    torch.save({"model_state_dict": model.cpu().state_dict(),
                "input_dim": X.shape[1]*64, "hidden_units": [128,64],
                "dataset_sha256": metadata["dataset_sha256"], "train_seed": seed}, out/"model.pt")
    # Histograms over the untouched TEST set, one row and one class at a time.
    weights = X[test_idx].sum(axis=-1)
    hist = np.zeros((2, X.shape[1], 65), dtype=np.int64)
    for label in (0,1):
        for row in range(X.shape[1]):
            hist[label,row] = np.bincount(weights[truth == label,row].astype(np.int64), minlength=65)
    np.savez(out/"hamming_histograms.npz", counts=hist,
             deltas_hex=np.array(metadata["deltas_hex"]), bins=np.arange(65))
    # A complete metrics file is the suite's completion marker: write it LAST.
    pending = out/"metrics.pending.json"
    pending.write_text(json.dumps(metrics, indent=2)+"\n")
    pending.replace(out/"metrics.json")
    print(json.dumps({"out": str(out), "accuracy": acc, "roc_auc": metrics["roc_auc"]}), flush=True)
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args()
    train_dataset(a.dataset, a.out, a.seed, a.split_seed, a.epochs, a.batch_size, a.device, a.threads)


if __name__ == "__main__":
    main()
