"""
train_hybrid.py
===============
Train the alpha-gated residual MLP for PMSM thermal virtual sensing.

Targets  : [delta_Tw, delta_Tt, delta_Tpm]  (winding, tooth, PM residuals in °C)
Gate input: rolling_residual_mag  (14th feature, index 13)

Architecture mirrors the electrical hybrid exactly:
    ResidualMLP : R^13 → FC64 → ReLU → FC64 → ReLU → FC3   (delta_T for 3 nodes)
    GateNetwork : R^1  → FC16 → ReLU → FC1  → sigmoid       (alpha, shared across nodes)

Final prediction:
    T_hybrid[node] = T_phys[node] + alpha * delta_T_hat[node]

FIX 1 — Profile-level train/val split  : rows from the same profile never
         appear in both train and val sets (no data leakage).
FIX 2 — Physical-unit RMSE tracking    : RMSE reported in °C per node per epoch.
FIX 3 — Output dimension               : ResidualMLP outputs 3 values (not 1).
FIX 4 — Normaliser                     : per-column Y normalisation (not scalar).
FIX 5 — Early stopping                 : patience-based, saves best checkpoint.
FIX 6 — Unified checkpoint             : single .pt file with model + normaliser.

Usage:
    python train_hybrid.py
    python train_hybrid.py --dataset results/thermal_hybrid_dataset.npz --epochs 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("PyTorch not installed.  pip install torch")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

# ── feature / target layout (must match feature_engineering.py) ──────────────
#  0  motor_speed      5  ambient         10  d(speed)/dt
#  1  i_d              6  coolant         11  d(iq)/dt
#  2  i_q              7  T_winding_phys  12  dynamic_score
#  3  u_d              8  T_tooth_phys    13  rolling_residual_mag  ← gate input
#  4  u_q              9  T_pm_phys
N_FEATURES   = 24    # total columns in X
N_MLP_FEAT   = 21    # features fed to MLP  (X[:, :13])
GATE_FEAT_IDX = 21   # column index for gate input (X[:, 13:14])
N_TARGETS    = 3     # [delta_Tw, delta_Tt, delta_Tpm]
TARGET_NAMES = ["winding", "tooth", "pm"]
SEQ_LEN      = 30
N_GATE_FEAT   = 3


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

class ResidualGRU(nn.Module):
    """Maps (batch, seq_len, N_MLP_FEAT) → (batch, 3) residual corrections."""

    def __init__(self, input_dim: int = N_MLP_FEAT, hidden_dim: int = 16,
                 output_dim: int = N_TARGETS):
        super().__init__()
        self.gru = nn.GRU(
        input_dim,
        hidden_dim,
        num_layers=2,
        dropout=0.2,
        batch_first=True
    )
        self.fc  = nn.Linear(hidden_dim, output_dim)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out[:, -1])


# NEW - one alpha per node

class GateNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(),   # 3 inputs: r_bar_winding, r_bar_tooth, r_bar_pm
            nn.Linear(16, 3),
            nn.Sigmoid(),
        )
    def forward(self, r_bar):              # r_bar shape: (batch, 3)
        return self.net(r_bar)             # output: (batch, 3) — one alpha per node


# ══════════════════════════════════════════════════════════════════════════════
# NORMALISER
# ══════════════════════════════════════════════════════════════════════════════

class HybridNormaliser:
    """
    Per-column z-score normalisation for both X (14 features) and Y (3 targets).
    Fit on training data only.  Mirrors electrical HybridNormalizer exactly.
    """

    def __init__(self):
        self.X_mean: np.ndarray | None = None
        self.X_std:  np.ndarray | None = None
        self.Y_mean: np.ndarray | None = None
        self.Y_std:  np.ndarray | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.X_mean = X.mean(axis=0).astype(np.float32)
        self.X_std  = (X.std(axis=0) + 1e-8).astype(np.float32)
        self.Y_mean = Y.mean(axis=0).astype(np.float32)
        self.Y_std  = (Y.std(axis=0) + 1e-8).astype(np.float32)

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.X_mean) / self.X_std).astype(np.float32)

    def transform_Y(self, Y: np.ndarray) -> np.ndarray:
        return ((Y - self.Y_mean) / self.Y_std).astype(np.float32)

    def inverse_Y(self, Y_norm: np.ndarray) -> np.ndarray:
        return Y_norm * self.Y_std + self.Y_mean

    def to_dict(self) -> dict:
        return {
            "X_mean": self.X_mean, "X_std": self.X_std,
            "Y_mean": self.Y_mean, "Y_std": self.Y_std,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HybridNormaliser":
        n = cls()
        n.X_mean = d["X_mean"]
        n.X_std  = d["X_std"]
        n.Y_mean = d["Y_mean"]
        n.Y_std  = d["Y_std"]
        return n


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / VAL SPLIT  (FIX 1 — profile-level, no leakage)
# ══════════════════════════════════════════════════════════════════════════════

def make_sequences(X: np.ndarray, Y: np.ndarray, seq_len: int = SEQ_LEN):
    """Build sliding-window sequences. Returns (N-seq_len, seq_len, F) and (N-seq_len, T)."""
    X_seq, Y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i - seq_len:i])
        Y_seq.append(Y[i])
    return (
        np.array(X_seq, dtype=np.float32),
        np.array(Y_seq, dtype=np.float32),
    )


def profile_level_split(
    X: np.ndarray,
    Y: np.ndarray,
    profile_ids: np.ndarray,
    val_fraction: float = 0.20,
    seed: int = 42,
):
    """
    All rows from one profile go entirely into train OR val.
    Prevents the model from seeing the same driving dynamics in both splits.
    """
    rng      = np.random.default_rng(seed)
    profiles = np.unique(profile_ids)
    rng.shuffle(profiles)

    n_val    = max(1, int(len(profiles) * val_fraction))
    val_pids = set(profiles[:n_val])

    trn_mask = np.array([p not in val_pids for p in profile_ids])
    val_mask  = ~trn_mask

    print(f"  Train profiles ({(trn_mask).sum() // max(1, len(profiles) - n_val)}): "
          f"{sorted(set(profiles) - val_pids)}")
    print(f"  Val   profiles ({n_val}): {sorted(val_pids)}")
    print(f"  Train samples: {trn_mask.sum()}   Val samples: {val_mask.sum()}")

    return (X[trn_mask], Y[trn_mask],
            X[val_mask],  Y[val_mask],
            profile_ids[val_mask])


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train thermal hybrid residual MLP")
    parser.add_argument("--dataset",    default="results/thermal_hybrid_dataset.npz")
    parser.add_argument("--out-model",  default="results/thermal_hybrid_model.pt")
    parser.add_argument("--out-plot",   default="results/thermal_hybrid_training_curve.png")
    parser.add_argument("--epochs",     type=int,   default=200)
    parser.add_argument("--batch-size", type=int,   default=512)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val-split",  type=float, default=0.20)
    parser.add_argument("--patience",   type=int,   default=25)
    args = parser.parse_args()

    # ── device ────────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    torch.set_float32_matmul_precision("high")

    # ── load dataset ──────────────────────────────────────────────────────────
    print(f"\nLoading {args.dataset} ...")
    data        = np.load(args.dataset)
    X_np        = data["X"].astype(np.float32)          # (N, 14)
    Y_np        = data["Y"].astype(np.float32)          # (N, 3)
    profile_ids = data["profile_ids"]
    print(f"X: {X_np.shape}   Y: {Y_np.shape}   "
          f"Profiles: {np.unique(profile_ids).tolist()}")

    assert X_np.shape[1] == N_FEATURES, (
        f"Expected {N_FEATURES} features, got {X_np.shape[1]}. "
        f"Re-run feature_engineering.py."
    )
    assert Y_np.shape[1] == N_TARGETS, (
        f"Expected {N_TARGETS} targets, got {Y_np.shape[1]}."
    )

    # ── profile-level split ───────────────────────────────────────────────────
    print(f"\nProfile-level train/val split (val={args.val_split:.0%}) ...")
    X_trn, Y_trn, X_val, Y_val, val_pids = profile_level_split(
        X_np, Y_np, profile_ids, val_fraction=args.val_split
    )

    # ── normalise on train only ───────────────────────────────────────────────
    norm = HybridNormaliser()
    norm.fit(X_trn, Y_trn)

    X_trn_n = norm.transform_X(X_trn)
    Y_trn_n = norm.transform_Y(Y_trn)
    X_val_n = norm.transform_X(X_val)
    Y_val_n = norm.transform_Y(Y_val)

    X_trn_seq, Y_trn_seq = make_sequences(X_trn_n, Y_trn_n)
    X_val_seq, Y_val_seq = make_sequences(X_val_n, Y_val_n)

    Y_std_t  = torch.from_numpy(norm.Y_std).to(device)   # (3,)
    Y_mean_t = torch.from_numpy(norm.Y_mean).to(device)  # (3,)

    # ── data loaders ──────────────────────────────────────────────────────────
    def make_loader(X_seq, Y_seq, shuffle):
        Xt = torch.from_numpy(X_seq[:, :, :N_MLP_FEAT])              # (N, seq_len, 21)
        rt = torch.from_numpy(X_seq[:, -1, GATE_FEAT_IDX:GATE_FEAT_IDX+N_GATE_FEAT])  # (N, 3)
        Yt = torch.from_numpy(Y_seq)                                  # (N, 3)
        return DataLoader(
            TensorDataset(Xt, rt, Yt),
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=(device.type == "cuda"),
        )

    train_loader = make_loader(X_trn_seq, Y_trn_seq, shuffle=True)
    val_loader   = make_loader(X_val_seq, Y_val_seq, shuffle=False)

    n_train = len(X_trn_seq)
    n_val   = len(X_val_seq)

    # ── model + optimiser ─────────────────────────────────────────────────────
    mlp  = ResidualGRU().to(device)
    gate = GateNetwork().to(device)
    optimizer = torch.optim.Adam(
        list(mlp.parameters()) + list(gate.parameters()),
        lr=args.lr,
    )
    criterion = nn.MSELoss()

    # ── training loop ─────────────────────────────────────────────────────────
    train_losses = []
    val_losses   = []
    val_rmse     = {n: [] for n in TARGET_NAMES}   # physical-unit RMSE per node

    best_val     = float("inf")
    patience_ctr = 0
    best_mlp_state  = None
    best_gate_state = None

    header = (f"  {'Epoch':>6}  {'TrainLoss':>12}  {'ValLoss':>12}  "
              f"{'RMSE_w(°C)':>11}  {'RMSE_t(°C)':>11}  {'RMSE_pm(°C)':>11}")
    print(f"\nTraining ({args.epochs} epochs, patience={args.patience}) ...\n")
    print(header)
    print(f"  {'-'*75}")

    for epoch in range(1, args.epochs + 1):

        # ── train ─────────────────────────────────────────────────────────────
        mlp.train(); gate.train()
        t_loss = 0.0
        for Xb, rb, Yb in train_loader:
            Xb, rb, Yb = Xb.to(device), rb.to(device), Yb.to(device)
            optimizer.zero_grad()
            delta_hat = mlp(Xb)                        # (batch, 3)
            alpha     = gate(rb)                       # (batch, 1)
            pred      = alpha * delta_hat              # (batch, 3)
            loss      = criterion(pred, Yb)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * len(Xb)
        t_loss /= n_train

        # ── validate ──────────────────────────────────────────────────────────
        mlp.eval(); gate.eval()
        v_loss    = 0.0
        all_pred  = []
        all_true  = []
        with torch.no_grad():
            for Xb, rb, Yb in val_loader:
                Xb, rb, Yb = Xb.to(device), rb.to(device), Yb.to(device)
                delta_hat = mlp(Xb)
                alpha     = gate(rb)
                pred      = alpha * delta_hat
                v_loss   += criterion(pred, Yb).item() * len(Xb)
                all_pred.append(pred)
                all_true.append(Yb)
        v_loss /= n_val

        # ── physical-unit RMSE (FIX 2) ────────────────────────────────────────
        all_pred_cat = torch.cat(all_pred)              # (N_val, 3)
        all_true_cat = torch.cat(all_true)
        pred_phys    = all_pred_cat * Y_std_t + Y_mean_t
        true_phys    = all_true_cat * Y_std_t + Y_mean_t

        epoch_rmse = {}
        for i, name in enumerate(TARGET_NAMES):
            rmse_i = float(torch.sqrt(
                torch.mean((pred_phys[:, i] - true_phys[:, i]) ** 2)
            ))
            epoch_rmse[name] = rmse_i
            val_rmse[name].append(rmse_i)

        train_losses.append(t_loss)
        val_losses.append(v_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  {epoch:6d}  {t_loss:12.6f}  {v_loss:12.6f}  "
                  f"{epoch_rmse['winding']:11.3f}  "
                  f"{epoch_rmse['tooth']:11.3f}  "
                  f"{epoch_rmse['pm']:11.3f}")

        # ── early stopping + checkpoint (FIX 5 + 6) ───────────────────────────
        if v_loss < best_val - 1e-7:
            best_val        = v_loss
            patience_ctr    = 0
            best_mlp_state  = {k: v.clone() for k, v in mlp.state_dict().items()}
            best_gate_state = {k: v.clone() for k, v in gate.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\n  Early stop at epoch {epoch}  "
                      f"(best val_loss={best_val:.6f})")
                break

    # ── restore best weights ──────────────────────────────────────────────────
    if best_mlp_state is not None:
        mlp.load_state_dict(best_mlp_state)
        gate.load_state_dict(best_gate_state)

    best_idx = int(np.argmin(val_losses))
    print(f"\nBest epoch {best_idx + 1}:"
          f"  val_loss={val_losses[best_idx]:.6f}")
    for name in TARGET_NAMES:
        print(f"  RMSE_{name} = {val_rmse[name][best_idx]:.3f} °C")

    # ── save unified checkpoint (FIX 6) ───────────────────────────────────────
    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "mlp_state":    mlp.state_dict(),
        "gate_state":   gate.state_dict(),
        "normaliser":   norm.to_dict(),
        "train_losses": train_losses,
        "val_losses":   val_losses,
        "val_rmse":     val_rmse,
        "val_profiles": val_pids.tolist(),
        "n_features":   N_FEATURES,
        "n_mlp_feat":   N_MLP_FEAT,
        "n_targets":    N_TARGETS,
    }, args.out_model)
    print(f"\nSaved model → {args.out_model}")

    # ── training curves ───────────────────────────────────────────────────────
    if HAS_PLT:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Thermal Hybrid Residual MLP — Training", fontsize=13,
                     fontweight="bold")

        ax = axes[0]
        ax.plot(train_losses, label="Train loss", color="#2196F3")
        ax.plot(val_losses,   label="Val loss",   color="#F44336")
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss (normalised)")
        ax.set_title("Loss curves"); ax.legend()
        ax.grid(True, alpha=0.3); ax.set_yscale("log")

        ax = axes[1]
        colors = {"winding": "#FF9800", "tooth": "#4CAF50", "pm": "#9C27B0"}
        for name in TARGET_NAMES:
            ax.plot(val_rmse[name], label=f"Val RMSE {name} (°C)",
                    color=colors[name])
        ax.set_xlabel("Epoch"); ax.set_ylabel("RMSE (°C)")
        ax.set_title("Validation RMSE — physical units")
        ax.legend(); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(args.out_plot, dpi=150, bbox_inches="tight")
        print(f"Saved plot → {args.out_plot}")
        plt.close(fig)


if __name__ == "__main__":
    main()