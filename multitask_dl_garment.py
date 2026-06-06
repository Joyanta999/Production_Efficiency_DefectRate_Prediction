#!/usr/bin/env python3
"""
=============================================================================
Multi-Task Deep Learning for Garment Manufacturing
Simultaneous Prediction of Production Efficiency & Quality (DHU)

Research Paper Implementation -- Model Comparison Study

Dataset  : VistaQDailyProduction.csv
Tasks    :
    Task 1 → AchievedEfficiency (%) -- Production efficiency
    Task 2 → dhu (%)               -- Defects per Hundred Units

Models Compared:
    1. MLP            -- Baseline Multi-Layer Perceptron
    2. DeepMLP        -- Deep MLP with BatchNorm, Dropout, GELU
    3. CNN1D          -- 1-D Convolutional Neural Network
    4. TabTransformer -- Transformer-based (feature tokens + self-attention)
    5. BiLSTM         -- Bidirectional LSTM (sliding-window time-series)

Outputs (saved to ./results/):
    model_comparison.csv   -- metric table
    model_comparison.png   -- comparison plots
    best_model_<name>.pth  -- best model checkpoint
=============================================================================
"""

import os, sys, warnings, time, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG = {
    'data_path'   : 'VistaQDailyProduction.csv',
    'results_dir' : 'results',
    'random_seed' : 42,
    'train_frac'  : 0.70,
    'val_frac'    : 0.15,
    # test_frac = 1 - train - val = 0.15 (time-based split, no shuffle)
    'batch_size'  : 64,
    'epochs'      : 150,
    'patience'    : 20,
    'lr'          : 1e-3,
    'weight_decay': 1e-4,
    'seq_len'     : 7,          # sliding-window length for BiLSTM
    'eff_weight'  : 0.5,        # multi-task loss weights
    'dhu_weight'  : 0.5,
    # Anomaly filters on targets
    'eff_min': 5.0,  'eff_max': 130.0,
    'dhu_min': 0.1,  'dhu_max':  50.0,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

torch.manual_seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])
os.makedirs(CONFIG['results_dir'], exist_ok=True)

print("=" * 70)
print("  Multi-Task Deep Learning -- Garment Production Analysis")
print("=" * 70)
print(f"  Device : {CONFIG['device']}")
print(f"  Results: ./{CONFIG['results_dir']}/")
print("=" * 70)


# -----------------------------------------------------------------------------
# 2. DATA LOADING & PREPROCESSING
# -----------------------------------------------------------------------------
print("\n[STEP 1] Data Loading & Preprocessing")

def load_and_preprocess(cfg):
    df = pd.read_csv(cfg['data_path'], encoding='utf-8-sig')
    print(f"  Raw rows : {len(df):,}")

    # -- Date parsing ------------------------------------------------------
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # -- Numeric conversion ------------------------------------------------
    num_raw = [
        'SMV', 'SampleSMV', 'CM', 'DayTarget', 'IETarget',
        'ManPowerPresent', 'PlannedHour', 'ActualHour', 'RunningWorkDay',
        'AchievedEfficiency', 'TargetEfficiency',
        'dhu', 'OutputPCS', 'InputPCS', 'InspectionPCS', 'DefectPCS',
        'TTLInputMin', 'ttlOutputMin', 'RejectPCS',
    ]
    for c in num_raw:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # -- Filter: keep only named factories (drop rows with numeric-only names) -
    df = df[df['WorkspaceFactoryName'].str.contains('Ltd', na=False)].copy()

    # -- Filter: anomalous target values ----------------------------------
    df = df[df['AchievedEfficiency'].between(cfg['eff_min'], cfg['eff_max'])]
    df = df[df['dhu'].between(cfg['dhu_min'], cfg['dhu_max'])]

    # -- Sort by date (required for time-based split & LSTM) ---------------
    df = df.sort_values('Date').reset_index(drop=True)
    print(f"  After filtering: {len(df):,} rows")

    # -- Feature engineering -----------------------------------------------
    df['DayOfWeek']      = df['Date'].dt.dayofweek.fillna(0).astype(float)
    df['Month']          = df['Date'].dt.month.fillna(10).astype(float)
    df['SMV_per_Worker'] = df['SMV'] / (df['ManPowerPresent'].clip(lower=1))
    df['HourUtil']       = df['ActualHour'] / (df['PlannedHour'].clip(lower=0.1))
    df['OutputPerWorker']= df['OutputPCS']  / (df['ManPowerPresent'].clip(lower=1))
    df['Eff_Gap']        = df['TargetEfficiency'] - df['AchievedEfficiency']

    # -- Categorical encoding ----------------------------------------------
    cat_cols = ['WorkspaceFactoryName', 'WorkspaceBuildingName', 'BuyerName']
    le_dict  = {}
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    # -- Numerical features ------------------------------------------------
    num_feats = [
        'SMV', 'DayTarget', 'IETarget', 'ManPowerPresent',
        'PlannedHour', 'ActualHour', 'RunningWorkDay', 'CM',
        'TargetEfficiency', 'TTLInputMin', 'ttlOutputMin',
        'DayOfWeek', 'Month', 'SMV_per_Worker',
        'HourUtil', 'OutputPerWorker', 'Eff_Gap',
    ]
    for c in num_feats:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    cat_feats = [c + '_enc' for c in cat_cols]
    all_feats = num_feats + cat_feats

    # -- Scale features (RobustScaler → handles outliers well) ------------
    feat_scaler = RobustScaler()
    X = feat_scaler.fit_transform(df[all_feats].values).astype(np.float32)

    # -- Scale targets (StandardScaler for inverse-transform metrics) ------
    targets = df[['AchievedEfficiency', 'dhu']].values.astype(np.float32)
    tgt_scaler = StandardScaler()
    y = tgt_scaler.fit_transform(targets).astype(np.float32)

    print(f"  Features : {len(all_feats)}  "
          f"(numerical={len(num_feats)}, categorical={len(cat_feats)})")
    print(f"  Samples  : {len(X):,}")

    return X, y, all_feats, tgt_scaler


X, y, feature_cols, tgt_scaler = load_and_preprocess(CONFIG)
input_dim = X.shape[1]

# -- Time-based (chronological) split -----------------------------------------
n = len(X)
train_end = int(n * CONFIG['train_frac'])
val_end   = int(n * (CONFIG['train_frac'] + CONFIG['val_frac']))

X_train, y_train = X[:train_end],         y[:train_end]
X_val,   y_val   = X[train_end:val_end],   y[train_end:val_end]
X_test,  y_test  = X[val_end:],            y[val_end:]

print(f"  Split    : Train={len(X_train):,}  Val={len(X_val):,}  Test={len(X_test):,}")


# -----------------------------------------------------------------------------
# 3. DATASET CLASSES
# -----------------------------------------------------------------------------
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class SequenceDataset(Dataset):
    """Sliding-window dataset for BiLSTM (time-ordered data required)."""
    def __init__(self, X, y, seq_len=7):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_len = seq_len
        self.indices = list(range(seq_len - 1, len(X)))

    def __len__(self):  return len(self.indices)
    def __getitem__(self, idx):
        i = self.indices[idx]
        return self.X[i - self.seq_len + 1: i + 1], self.y[i]


def make_loaders(Xtr, ytr, Xv, yv, Xte, yte, batch_size, seq_len, lstm=False):
    Cls = SequenceDataset if lstm else TabularDataset
    kw  = {'seq_len': seq_len} if lstm else {}
    tr  = DataLoader(Cls(Xtr, ytr, **kw), batch_size=batch_size, shuffle=not lstm)
    vl  = DataLoader(Cls(Xv,  yv,  **kw), batch_size=batch_size, shuffle=False)
    te  = DataLoader(Cls(Xte, yte, **kw), batch_size=batch_size, shuffle=False)
    return tr, vl, te

tab_tr, tab_vl, tab_te = make_loaders(
    X_train, y_train, X_val, y_val, X_test, y_test,
    CONFIG['batch_size'], CONFIG['seq_len'], lstm=False)

seq_tr, seq_vl, seq_te = make_loaders(
    X_train, y_train, X_val, y_val, X_test, y_test,
    CONFIG['batch_size'], CONFIG['seq_len'], lstm=True)


# -----------------------------------------------------------------------------
# 4. MODEL ARCHITECTURES
# -----------------------------------------------------------------------------
print("\n[STEP 2] Model Architectures")


class MLP(nn.Module):
    """Model 1 -- Baseline Multi-Layer Perceptron with dual regression heads."""
    def __init__(self, in_dim, hidden=(256, 128), drop=0.3):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.backbone  = nn.Sequential(*layers)
        self.eff_head  = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 1))
        self.dhu_head  = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        f = self.backbone(x)
        return self.eff_head(f).squeeze(-1), self.dhu_head(f).squeeze(-1)


class DeepMLP(nn.Module):
    """Model 2 -- Deep MLP with BatchNorm, Dropout, GELU activations."""
    def __init__(self, in_dim, hidden=(512, 256, 128, 64), drop=0.3):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden[0])
        blocks = []
        for i in range(len(hidden) - 1):
            blocks.append(nn.Sequential(
                nn.Linear(hidden[i], hidden[i + 1]),
                nn.BatchNorm1d(hidden[i + 1]),
                nn.GELU(),
                nn.Dropout(drop),
            ))
        self.blocks   = nn.ModuleList(blocks)
        d = hidden[-1]
        self.eff_head = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 1))
        self.dhu_head = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):
        x = F.gelu(self.proj(x))
        for blk in self.blocks:
            x = blk(x)
        return self.eff_head(x).squeeze(-1), self.dhu_head(x).squeeze(-1)


class CNN1D(nn.Module):
    """Model 3 -- 1-D CNN treating feature vector as a 1-D signal."""
    def __init__(self, in_dim, drop=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64,  kernel_size=3, padding=1), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.pool     = nn.AdaptiveAvgPool1d(1)
        self.dropout  = nn.Dropout(drop)
        self.eff_head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64, 1))
        self.dhu_head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64, 1))

    def forward(self, x):
        x = x.unsqueeze(1)            # (B, 1, F)
        x = self.conv(x)              # (B, 256, F)
        x = self.pool(x).squeeze(-1)  # (B, 256)
        x = self.dropout(x)
        return self.eff_head(x).squeeze(-1), self.dhu_head(x).squeeze(-1)


class _TransformerBlock(nn.Module):
    def __init__(self, d, heads, ffn_d, drop):
        super().__init__()
        self.attn   = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.norm1  = nn.LayerNorm(d)
        self.ffn    = nn.Sequential(
            nn.Linear(d, ffn_d), nn.GELU(), nn.Dropout(drop), nn.Linear(ffn_d, d))
        self.norm2  = nn.LayerNorm(d)
        self.drop   = nn.Dropout(drop)

    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x    = self.norm1(x + self.drop(a))
        x    = self.norm2(x + self.drop(self.ffn(x)))
        return x


class TabTransformer(nn.Module):
    """Model 4 -- Transformer for tabular data: each feature is a token."""
    def __init__(self, in_dim, d_model=64, heads=4, n_layers=3, drop=0.2):
        super().__init__()
        self.feat_proj    = nn.Linear(1, d_model)
        self.pos_emb      = nn.Parameter(torch.randn(1, in_dim, d_model) * 0.02)
        self.blocks       = nn.ModuleList([
            _TransformerBlock(d_model, heads, d_model * 4, drop)
            for _ in range(n_layers)
        ])
        self.norm         = nn.LayerNorm(d_model)
        self.eff_head     = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(drop), nn.Linear(64, 1))
        self.dhu_head     = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(drop), nn.Linear(64, 1))

    def forward(self, x):
        x = x.unsqueeze(-1)          # (B, F, 1)
        x = self.feat_proj(x)        # (B, F, d_model)
        x = x + self.pos_emb
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x).mean(dim=1) # (B, d_model) -- mean pooling over feature tokens
        return self.eff_head(x).squeeze(-1), self.dhu_head(x).squeeze(-1)


class BiLSTM(nn.Module):
    """Model 5 -- Bidirectional LSTM for sequential production time-series."""
    def __init__(self, in_dim, hidden=128, n_layers=2, drop=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim, hidden_size=hidden,
            num_layers=n_layers, batch_first=True,
            dropout=drop if n_layers > 1 else 0.0,
            bidirectional=True,
        )
        d = hidden * 2
        self.norm     = nn.LayerNorm(d)
        self.drop     = nn.Dropout(drop)
        self.eff_head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64, 1))
        self.dhu_head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        x = self.norm(self.drop(out[:, -1, :]))  # last time-step
        return self.eff_head(x).squeeze(-1), self.dhu_head(x).squeeze(-1)


# Print parameter counts
all_model_specs = [
    ('MLP',            MLP(input_dim),                            tab_tr, tab_vl, tab_te),
    ('DeepMLP',        DeepMLP(input_dim),                        tab_tr, tab_vl, tab_te),
    ('CNN1D',          CNN1D(input_dim),                          tab_tr, tab_vl, tab_te),
    ('TabTransformer', TabTransformer(input_dim),                 tab_tr, tab_vl, tab_te),
    ('BiLSTM',         BiLSTM(input_dim),                         seq_tr, seq_vl, seq_te),
]

print(f"  {'Model':<18}  {'Parameters':>12}")
print(f"  {'-'*18}  {'-'*12}")
for name, mdl, *_ in all_model_specs:
    p = sum(v.numel() for v in mdl.parameters() if v.requires_grad)
    print(f"  {name:<18}  {p:>12,}")


# -----------------------------------------------------------------------------
# 5. TRAINING UTILITIES
# -----------------------------------------------------------------------------
def mt_loss(p_eff, p_dhu, t_eff, t_dhu, w_eff=0.5, w_dhu=0.5):
    l_e = F.mse_loss(p_eff, t_eff)
    l_d = F.mse_loss(p_dhu, t_dhu)
    return w_eff * l_e + w_dhu * l_d, l_e.item(), l_d.item()


def train_one_epoch(model, loader, optimizer, device, we, wd):
    model.train()
    total = 0.0
    for Xb, yb in loader:
        Xb = Xb.to(device)
        te, td = yb[:, 0].to(device), yb[:, 1].to(device)
        optimizer.zero_grad()
        pe, pd = model(Xb)
        loss, _, _ = mt_loss(pe, pd, te, td, we, wd)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def eval_one_epoch(model, loader, device, we, wd):
    model.eval()
    total = 0.0
    p_eff_all, p_dhu_all, t_eff_all, t_dhu_all = [], [], [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            te, td = yb[:, 0].to(device), yb[:, 1].to(device)
            pe, pd = model(Xb)
            loss, _, _ = mt_loss(pe, pd, te, td, we, wd)
            total += loss.item()
            p_eff_all.extend(pe.cpu().numpy())
            p_dhu_all.extend(pd.cpu().numpy())
            t_eff_all.extend(te.cpu().numpy())
            t_dhu_all.extend(td.cpu().numpy())
    return (total / len(loader),
            np.array(p_eff_all), np.array(p_dhu_all),
            np.array(t_eff_all), np.array(t_dhu_all))


def orig_scale(arr, mean, std):
    return arr * std + mean


def compute_metrics(pred, true, mean, std):
    p = orig_scale(pred, mean, std)
    t = orig_scale(true, mean, std)
    return (mean_absolute_error(t, p),
            math.sqrt(mean_squared_error(t, p)),
            r2_score(t, p))


def train_model(name, model, tr_loader, vl_loader, te_loader, cfg):
    dev  = cfg['device']
    we, wd = cfg['eff_weight'], cfg['dhu_weight']
    model = model.to(dev)
    opt   = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    sched = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=8, min_lr=1e-6)

    best_val, patience_ctr, best_state = float('inf'), 0, None
    tr_losses, vl_losses = [], []
    t0 = time.time()

    print(f"\n  -- Training [{name}] --")
    for epoch in range(1, cfg['epochs'] + 1):
        tl = train_one_epoch(model, tr_loader, opt, dev, we, wd)
        vl, *_ = eval_one_epoch(model, vl_loader, dev, we, wd)
        tr_losses.append(tl); vl_losses.append(vl)
        sched.step(vl)

        if vl < best_val:
            best_val, patience_ctr = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        if epoch % 10 == 0:
            lr = opt.param_groups[0]['lr']
            print(f"    Epoch {epoch:3d} | train={tl:.4f} | val={vl:.4f} | lr={lr:.2e}")

        if patience_ctr >= cfg['patience']:
            print(f"    Early stop at epoch {epoch}")
            break

    train_time = time.time() - t0

    # -- Test-set evaluation -----------------------------------------------
    model.load_state_dict(best_state)
    _, pe, pd_, te_, td_ = eval_one_epoch(model, te_loader, dev, we, wd)

    eff_m, eff_s = tgt_scaler.mean_[0], tgt_scaler.scale_[0]
    dhu_m, dhu_s = tgt_scaler.mean_[1], tgt_scaler.scale_[1]

    eff_mae, eff_rmse, eff_r2 = compute_metrics(pe,  te_, eff_m, eff_s)
    dhu_mae, dhu_rmse, dhu_r2 = compute_metrics(pd_, td_, dhu_m, dhu_s)
    avg_r2 = (eff_r2 + dhu_r2) / 2

    print(f"    [TEST]  Efficiency → MAE={eff_mae:.3f}%  RMSE={eff_rmse:.3f}%  R²={eff_r2:.4f}")
    print(f"    [TEST]  DHU        → MAE={dhu_mae:.3f}%  RMSE={dhu_rmse:.3f}%  R²={dhu_r2:.4f}")
    print(f"    Avg R²={avg_r2:.4f}  |  Time={train_time:.1f}s")

    return dict(
        model_name=name, model=model,
        tr_losses=tr_losses, vl_losses=vl_losses,
        eff_mae=eff_mae, eff_rmse=eff_rmse, eff_r2=eff_r2,
        dhu_mae=dhu_mae, dhu_rmse=dhu_rmse, dhu_r2=dhu_r2,
        avg_r2=avg_r2, train_time=train_time,
        pred_eff=pe, pred_dhu=pd_, true_eff=te_, true_dhu=td_,
        eff_m=eff_m, eff_s=eff_s, dhu_m=dhu_m, dhu_s=dhu_s,
    )


# -----------------------------------------------------------------------------
# 6. TRAIN ALL MODELS
# -----------------------------------------------------------------------------
print("\n[STEP 3] Training All Models")

all_results = []
for name, mdl, tr, vl, te in all_model_specs:
    try:
        res = train_model(name, mdl, tr, vl, te, CONFIG)
        all_results.append(res)
    except Exception as e:
        print(f"  ERROR training {name}: {e}")


# -----------------------------------------------------------------------------
# 7. RESULTS TABLE
# -----------------------------------------------------------------------------
print("\n[STEP 4] Results Comparison")

rows = []
for r in all_results:
    rows.append({
        'Model'        : r['model_name'],
        'Eff_MAE(%)'   : round(r['eff_mae'],  3),
        'Eff_RMSE(%)'  : round(r['eff_rmse'], 3),
        'Eff_R²'       : round(r['eff_r2'],   4),
        'DHU_MAE(%)'   : round(r['dhu_mae'],  3),
        'DHU_RMSE(%)'  : round(r['dhu_rmse'], 3),
        'DHU_R²'       : round(r['dhu_r2'],   4),
        'Avg_R²'       : round(r['avg_r2'],   4),
        'Time(s)'      : round(r['train_time'], 1),
    })

res_df = (pd.DataFrame(rows)
            .sort_values('Avg_R²', ascending=False)
            .reset_index(drop=True))
res_df.insert(0, 'Rank', range(1, len(res_df) + 1))

print("\n" + res_df.to_string(index=False))

csv_path = os.path.join(CONFIG['results_dir'], 'model_comparison.csv')
res_df.to_csv(csv_path, index=False)
print(f"\n  Saved: {csv_path}")

best_name = res_df.iloc[0]['Model']
best_res  = next(r for r in all_results if r['model_name'] == best_name)
print(f"  Best model: {best_name}  (Avg R²={res_df.iloc[0]['Avg_R²']})")


# -----------------------------------------------------------------------------
# 8. VISUALISATION
# -----------------------------------------------------------------------------
print("\n[STEP 5] Generating Plots")

PALETTE = ['#1976D2', '#388E3C', '#F57C00', '#7B1FA2', '#D32F2F']
name2col = {r['model_name']: PALETTE[i] for i, r in enumerate(all_results)}

fig = plt.figure(figsize=(26, 22))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

model_order = res_df['Model'].tolist()

# -- Plot A: Validation Loss Curves -------------------------------------------
axA = fig.add_subplot(gs[0, :])
for r in all_results:
    axA.plot(range(1, len(r['vl_losses']) + 1), r['vl_losses'],
             label=r['model_name'], color=name2col[r['model_name']], linewidth=2)
axA.set_title('Validation Loss During Training (All Models)', fontsize=14, fontweight='bold')
axA.set_xlabel('Epoch'); axA.set_ylabel('Validation Loss (MSE)')
axA.legend(); axA.grid(True, alpha=0.3)

# -- Plot B: R² Comparison ----------------------------------------------------
axB = fig.add_subplot(gs[1, 0])
x   = np.arange(len(model_order)); w = 0.35
eff_r2 = [res_df.loc[res_df['Model'] == m, 'Eff_R²'].values[0] for m in model_order]
dhu_r2 = [res_df.loc[res_df['Model'] == m, 'DHU_R²'].values[0] for m in model_order]
axB.bar(x - w/2, eff_r2, w, label='Efficiency R²', color='#1976D2', alpha=0.85)
axB.bar(x + w/2, dhu_r2, w, label='DHU R²',        color='#D32F2F', alpha=0.85)
axB.set_title('R² Score (higher = better)', fontsize=12, fontweight='bold')
axB.set_xticks(x); axB.set_xticklabels(model_order, rotation=20, ha='right')
axB.set_ylabel('R²'); axB.legend(); axB.grid(True, alpha=0.3, axis='y')
axB.axhline(0, color='black', linewidth=0.8, linestyle='--')

# -- Plot C: MAE Comparison ---------------------------------------------------
axC = fig.add_subplot(gs[1, 1])
eff_mae = [res_df.loc[res_df['Model'] == m, 'Eff_MAE(%)'].values[0] for m in model_order]
dhu_mae = [res_df.loc[res_df['Model'] == m, 'DHU_MAE(%)'].values[0] for m in model_order]
axC.bar(x - w/2, eff_mae, w, label='Efficiency MAE', color='#388E3C', alpha=0.85)
axC.bar(x + w/2, dhu_mae, w, label='DHU MAE',        color='#7B1FA2', alpha=0.85)
axC.set_title('MAE % (lower = better)', fontsize=12, fontweight='bold')
axC.set_xticks(x); axC.set_xticklabels(model_order, rotation=20, ha='right')
axC.set_ylabel('MAE (%)'); axC.legend(); axC.grid(True, alpha=0.3, axis='y')

# -- Plot D: Training Time ----------------------------------------------------
axD = fig.add_subplot(gs[1, 2])
times = [res_df.loc[res_df['Model'] == m, 'Time(s)'].values[0] for m in model_order]
bars  = axD.barh(model_order, times, color=[name2col[m] for m in model_order], alpha=0.85)
axD.set_title('Training Time (seconds)', fontsize=12, fontweight='bold')
axD.set_xlabel('Seconds'); axD.grid(True, alpha=0.3, axis='x')
for bar, v in zip(bars, times):
    axD.text(bar.get_width() + max(times)*0.01, bar.get_y() + bar.get_height()/2,
             f'{v:.0f}s', va='center', fontsize=9)

# -- Plot E: Best Model -- Efficiency Predicted vs Actual ---------------------
axE  = fig.add_subplot(gs[2, 0])
pe_o = orig_scale(best_res['pred_eff'], best_res['eff_m'], best_res['eff_s'])
te_o = orig_scale(best_res['true_eff'], best_res['eff_m'], best_res['eff_s'])
axE.scatter(te_o, pe_o, alpha=0.35, s=12, color='#1976D2')
lim  = [min(te_o.min(), pe_o.min()) - 2, max(te_o.max(), pe_o.max()) + 2]
axE.plot(lim, lim, 'r--', lw=2, label='Ideal')
axE.set_title(f'{best_name} -- Efficiency: Pred vs Actual', fontsize=12, fontweight='bold')
axE.set_xlabel('Actual Efficiency (%)'); axE.set_ylabel('Predicted Efficiency (%)')
axE.legend(); axE.grid(True, alpha=0.3)

# -- Plot F: Best Model -- DHU Predicted vs Actual -----------------------------
axF  = fig.add_subplot(gs[2, 1])
pd_o = orig_scale(best_res['pred_dhu'], best_res['dhu_m'], best_res['dhu_s'])
td_o = orig_scale(best_res['true_dhu'], best_res['dhu_m'], best_res['dhu_s'])
axF.scatter(td_o, pd_o, alpha=0.35, s=12, color='#D32F2F')
lim  = [min(td_o.min(), pd_o.min()) - 0.5, max(td_o.max(), pd_o.max()) + 0.5]
axF.plot(lim, lim, 'r--', lw=2, label='Ideal')
axF.set_title(f'{best_name} -- DHU: Pred vs Actual', fontsize=12, fontweight='bold')
axF.set_xlabel('Actual DHU (%)'); axF.set_ylabel('Predicted DHU (%)')
axF.legend(); axF.grid(True, alpha=0.3)

# -- Plot G: Average R² Ranking -----------------------------------------------
axG  = fig.add_subplot(gs[2, 2])
avg_r2_vals = [res_df.loc[res_df['Model'] == m, 'Avg_R²'].values[0] for m in model_order]
bar_cols = ['gold' if m == best_name else '#607D8B' for m in model_order]
bars = axG.barh(model_order, avg_r2_vals, color=bar_cols, alpha=0.9, edgecolor='black', linewidth=0.8)
axG.set_title('Average R² Ranking (Eff + DHU)', fontsize=12, fontweight='bold')
axG.set_xlabel('Average R²'); axG.grid(True, alpha=0.3, axis='x')
axG.axvline(0, color='black', linewidth=0.8)
for bar, v in zip(bars, avg_r2_vals):
    c = 'black' if v < 0.05 else 'white'
    xp = max(bar.get_width() - 0.01, 0.005)
    axG.text(xp, bar.get_y() + bar.get_height()/2,
             f'{v:.4f}', va='center', ha='right', fontweight='bold', color=c, fontsize=9)

fig.suptitle(
    'Multi-Task Deep Learning for RMG Production -- Efficiency & DHU Prediction\n'
    f'Dataset: VistaQDailyProduction.csv  |  Best Model: {best_name}',
    fontsize=15, fontweight='bold', y=1.01)

plot_path = os.path.join(CONFIG['results_dir'], 'model_comparison.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {plot_path}")


# -----------------------------------------------------------------------------
# 9. SAVE BEST MODEL CHECKPOINT
# -----------------------------------------------------------------------------
ckpt_path = os.path.join(CONFIG['results_dir'], f'best_model_{best_name}.pth')
torch.save({
    'model_name'      : best_name,
    'model_state_dict': best_res['model'].state_dict(),
    'config'          : CONFIG,
    'feature_cols'    : feature_cols,
    'tgt_scaler_mean' : tgt_scaler.mean_.tolist(),
    'tgt_scaler_std'  : tgt_scaler.scale_.tolist(),
    'metrics': {
        'eff_mae' : best_res['eff_mae'],  'eff_rmse': best_res['eff_rmse'],
        'eff_r2'  : best_res['eff_r2'],   'dhu_mae' : best_res['dhu_mae'],
        'dhu_rmse': best_res['dhu_rmse'], 'dhu_r2'  : best_res['dhu_r2'],
        'avg_r2'  : best_res['avg_r2'],
    },
}, ckpt_path)
print(f"  Saved: {ckpt_path}")


# -----------------------------------------------------------------------------
# 10. FINAL SUMMARY
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  FINAL RESULTS SUMMARY")
print("=" * 70)
print(res_df[['Rank', 'Model', 'Eff_R²', 'DHU_R²', 'Avg_R²',
              'Eff_MAE(%)', 'DHU_MAE(%)', 'Time(s)']].to_string(index=False))
print()
print(f"  Winner       : {best_name}")
print(f"  Avg R²       : {best_res['avg_r2']:.4f}")
print(f"  Eff  MAE/R²  : {best_res['eff_mae']:.3f}% / {best_res['eff_r2']:.4f}")
print(f"  DHU  MAE/R²  : {best_res['dhu_mae']:.3f}% / {best_res['dhu_r2']:.4f}")
print()
print(f"  Output files in ./{CONFIG['results_dir']}/")
print(f"    model_comparison.csv")
print(f"    model_comparison.png")
print(f"    best_model_{best_name}.pth")
print("=" * 70)
