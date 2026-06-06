#!/usr/bin/env python3
"""
Figure generation for research paper.
Generates all 8 publication-quality figures:
  Fig 1 - EDA Dashboard
  Fig 2 - Feature Correlation Heatmap
  Fig 3 - MTL Framework Diagram
  Fig 4 - Five Model Architecture Diagrams
  Fig 5 - Training / Validation Loss Curves
  Fig 6 - Performance Comparison Charts
  Fig 7 - Predicted vs Actual (CNN1D)
  Fig 8 - Residual Analysis
"""

import os, sys, warnings, time, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
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

FIG_DIR = 'results/figures'
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

PALETTE = {
    'MLP':            '#1565C0',
    'DeepMLP':        '#2E7D32',
    'CNN1D':          '#E65100',
    'TabTransformer': '#6A1B9A',
    'BiLSTM':         '#B71C1C',
}
COLORS = list(PALETTE.values())

print("=" * 60)
print("  Figure Generation for Research Paper")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# DATA LOADING (shared with training)
# ─────────────────────────────────────────────────────────────
CONFIG = {
    'data_path': 'VistaQDailyProduction.csv',
    'random_seed': 42,
    'train_frac': 0.70, 'val_frac': 0.15,
    'batch_size': 64, 'epochs': 150, 'patience': 20,
    'lr': 1e-3, 'weight_decay': 1e-4, 'seq_len': 7,
    'eff_weight': 0.5, 'dhu_weight': 0.5,
    'eff_min': 5.0, 'eff_max': 130.0,
    'dhu_min': 0.1, 'dhu_max': 50.0,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}
torch.manual_seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])

def load_data(cfg):
    df = pd.read_csv(cfg['data_path'], encoding='utf-8-sig')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    num_raw = ['SMV','SampleSMV','CM','DayTarget','IETarget','ManPowerPresent',
               'PlannedHour','ActualHour','RunningWorkDay','AchievedEfficiency',
               'TargetEfficiency','dhu','OutputPCS','InputPCS','InspectionPCS',
               'DefectPCS','TTLInputMin','ttlOutputMin','RejectPCS']
    for c in num_raw:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df_raw = df[df['WorkspaceFactoryName'].str.contains('Ltd', na=False)].copy()
    df_clean = df_raw[
        df_raw['AchievedEfficiency'].between(cfg['eff_min'], cfg['eff_max']) &
        df_raw['dhu'].between(cfg['dhu_min'], cfg['dhu_max'])
    ].copy()
    df_clean = df_clean.sort_values('Date').reset_index(drop=True)
    df_clean['DayOfWeek']       = df_clean['Date'].dt.dayofweek.fillna(0).astype(float)
    df_clean['Month']           = df_clean['Date'].dt.month.fillna(10).astype(float)
    df_clean['SMV_per_Worker']  = df_clean['SMV'] / df_clean['ManPowerPresent'].clip(lower=1)
    df_clean['HourUtil']        = df_clean['ActualHour'] / df_clean['PlannedHour'].clip(lower=0.1)
    df_clean['OutputPerWorker'] = df_clean['OutputPCS'] / df_clean['ManPowerPresent'].clip(lower=1)
    df_clean['Eff_Gap']         = df_clean['TargetEfficiency'] - df_clean['AchievedEfficiency']
    cat_cols = ['WorkspaceFactoryName','WorkspaceBuildingName','BuyerName']
    for col in cat_cols:
        df_clean[col] = df_clean[col].fillna('Unknown')
        le = LabelEncoder()
        df_clean[col+'_enc'] = le.fit_transform(df_clean[col].astype(str))
    num_feats = ['SMV','DayTarget','IETarget','ManPowerPresent','PlannedHour',
                 'ActualHour','RunningWorkDay','CM','TargetEfficiency','TTLInputMin',
                 'ttlOutputMin','DayOfWeek','Month','SMV_per_Worker','HourUtil',
                 'OutputPerWorker','Eff_Gap']
    for c in num_feats:
        if c in df_clean.columns:
            df_clean[c] = df_clean[c].fillna(df_clean[c].median())
    cat_feats  = [c+'_enc' for c in cat_cols]
    all_feats  = num_feats + cat_feats
    feat_scaler = RobustScaler()
    X = feat_scaler.fit_transform(df_clean[all_feats].values).astype(np.float32)
    tgt_scaler = StandardScaler()
    y = tgt_scaler.fit_transform(df_clean[['AchievedEfficiency','dhu']].values).astype(np.float32)
    return X, y, all_feats, num_feats, tgt_scaler, df_clean

print("\n[1/8] Loading data...")
X, y, feature_cols, num_feats, tgt_scaler, df = load_data(CONFIG)
n = len(X)
te  = int(n * CONFIG['train_frac'])
ve  = int(n * (CONFIG['train_frac'] + CONFIG['val_frac']))
X_train, y_train = X[:te],  y[:te]
X_val,   y_val   = X[te:ve], y[te:ve]
X_test,  y_test  = X[ve:],   y[ve:]
input_dim = X.shape[1]
print(f"    Loaded {n:,} samples | features={input_dim}")

# ─────────────────────────────────────────────────────────────
# FIG 1 — EDA DASHBOARD
# ─────────────────────────────────────────────────────────────
print("\n[2/8] Figure 1 — EDA Dashboard...")
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle('Exploratory Data Analysis — VistaQ Daily Production Dataset',
             fontsize=13, fontweight='bold', y=1.01)

eff = df['AchievedEfficiency']
dhu_col = df['dhu']

# (a) Efficiency distribution
ax = axes[0, 0]
ax.hist(eff, bins=40, color='#1565C0', alpha=0.75, edgecolor='white', linewidth=0.4)
ax.axvline(eff.mean(), color='red', linestyle='--', linewidth=1.4, label=f'Mean = {eff.mean():.1f}%')
ax.axvline(eff.median(), color='orange', linestyle=':', linewidth=1.4, label=f'Median = {eff.median():.1f}%')
ax.set_xlabel('Achieved Efficiency (%)')
ax.set_ylabel('Frequency')
ax.set_title('(a) Distribution of Achieved Efficiency')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (b) DHU distribution
ax = axes[0, 1]
ax.hist(dhu_col, bins=40, color='#E65100', alpha=0.75, edgecolor='white', linewidth=0.4)
ax.axvline(dhu_col.mean(), color='red', linestyle='--', linewidth=1.4, label=f'Mean = {dhu_col.mean():.2f}%')
ax.axvline(dhu_col.median(), color='orange', linestyle=':', linewidth=1.4, label=f'Median = {dhu_col.median():.2f}%')
ax.set_xlabel('DHU (%)')
ax.set_ylabel('Frequency')
ax.set_title('(b) Distribution of DHU (Defects per 100 Units)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (c) Efficiency by Factory — box plot
ax = axes[0, 2]
factories = df['WorkspaceFactoryName'].unique()
data_by_factory = [df[df['WorkspaceFactoryName'] == f]['AchievedEfficiency'].values for f in factories]
short_names = [f.replace(' Ltd.', '').replace(' Limited', '') for f in factories]
bp = ax.boxplot(data_by_factory, patch_artist=True, notch=False,
                medianprops=dict(color='black', linewidth=1.5))
colors_box = ['#1565C0', '#2E7D32', '#B71C1C']
for patch, color in zip(bp['boxes'], colors_box[:len(factories)]):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax.set_xticklabels(short_names, fontsize=8)
ax.set_ylabel('Achieved Efficiency (%)')
ax.set_title('(c) Efficiency Distribution by Factory')
ax.grid(True, alpha=0.3, axis='y')

# (d) DHU by Factory
ax = axes[1, 0]
data_dhu = [df[df['WorkspaceFactoryName'] == f]['dhu'].values for f in factories]
bp2 = ax.boxplot(data_dhu, patch_artist=True, notch=False,
                 medianprops=dict(color='black', linewidth=1.5))
for patch, color in zip(bp2['boxes'], colors_box[:len(factories)]):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax.set_xticklabels(short_names, fontsize=8)
ax.set_ylabel('DHU (%)')
ax.set_title('(d) DHU Distribution by Factory')
ax.grid(True, alpha=0.3, axis='y')

# (e) Top buyers
ax = axes[1, 1]
buyer_counts = df['BuyerName'].value_counts().head(8)
bars = ax.barh(buyer_counts.index[::-1], buyer_counts.values[::-1],
               color='#6A1B9A', alpha=0.8, edgecolor='white')
ax.set_xlabel('Number of Records')
ax.set_title('(e) Records by Buyer (Top 8)')
for bar, val in zip(bars, buyer_counts.values[::-1]):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            str(val), va='center', fontsize=8)
ax.grid(True, alpha=0.3, axis='x')

# (f) Efficiency vs DHU scatter
ax = axes[1, 2]
factory_list = df['WorkspaceFactoryName'].unique()
cmap_f = ['#1565C0','#2E7D32','#B71C1C']
for i, f in enumerate(factory_list):
    sub = df[df['WorkspaceFactoryName'] == f]
    ax.scatter(sub['AchievedEfficiency'], sub['dhu'],
               alpha=0.25, s=8, color=cmap_f[i % len(cmap_f)],
               label=f.replace(' Ltd.', '').replace(' Limited', ''))
ax.set_xlabel('Achieved Efficiency (%)')
ax.set_ylabel('DHU (%)')
ax.set_title('(f) Efficiency vs. DHU by Factory')
ax.legend(fontsize=7, markerscale=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig1_eda_dashboard.png')
plt.close()
print("    Saved fig1_eda_dashboard.png")

# ─────────────────────────────────────────────────────────────
# FIG 2 — FEATURE CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────
print("\n[3/8] Figure 2 — Correlation Heatmap...")
corr_cols = ['SMV','DayTarget','IETarget','ManPowerPresent','PlannedHour',
             'ActualHour','RunningWorkDay','TargetEfficiency',
             'SMV_per_Worker','HourUtil','OutputPerWorker','Eff_Gap',
             'AchievedEfficiency','dhu']
corr_labels = ['SMV','Day Target','IE Target','Manpower','Plan.Hours',
               'Act.Hours','Work Days','Tgt.Eff.','SMV/Worker',
               'Hour Util.','Out/Worker','Eff.Gap','Eff.(Target)','DHU']
corr_df = df[corr_cols].copy()
corr_df.columns = corr_labels
corr_matrix = corr_df.corr()

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, ax=ax,
            annot_kws={'size': 7.5},
            cbar_kws={'shrink': 0.8, 'label': 'Pearson Correlation'})
ax.set_title('Feature Correlation Matrix\n(Lower Triangle — Pearson r)',
             fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig2_correlation_heatmap.png')
plt.close()
print("    Saved fig2_correlation_heatmap.png")

# ─────────────────────────────────────────────────────────────
# FIG 3 — MTL FRAMEWORK DIAGRAM
# ─────────────────────────────────────────────────────────────
print("\n[4/8] Figure 3 — MTL Framework Diagram...")

def draw_box(ax, x, y, w, h, text, fc='#E3F2FD', ec='#1565C0',
             fontsize=9, bold=False, radius=0.04, text_color='black'):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f'round,pad=0.01,rounding_size={radius}',
                         fc=fc, ec=ec, linewidth=1.8, zorder=3)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, color=text_color,
            wrap=True, zorder=4)

def draw_arrow_h(ax, x1, x2, y, color='#424242', lw=1.5):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))

fig, ax = plt.subplots(figsize=(14, 5.5))
ax.set_xlim(0, 14); ax.set_ylim(0, 5.5)
ax.axis('off')
ax.set_facecolor('#FAFAFA')
fig.patch.set_facecolor('#FAFAFA')

# Input block
draw_box(ax, 0.2, 0.8, 2.2, 3.8, '  Input Features  \n\n'
         '17 Numerical:\nSMV, Manpower,\nPlanned Hours,\n'
         'IE Target, ...\n\n3 Categorical:\nFactory, Building\nBuyer',
         fc='#E3F2FD', ec='#1565C0', fontsize=8)

draw_arrow_h(ax, 2.4, 3.0, 2.7, '#1565C0')

# Shared Encoder block
draw_box(ax, 3.0, 0.4, 3.5, 4.6,
         'Shared Backbone Encoder\n\n'
         'MLP:  FC(256) + FC(128)\n'
         'DeepMLP:  FC[512,256,128,64]\n'
         '           +BN+GELU+Dropout\n'
         'CNN1D:  Conv1D x3 + AvgPool\n'
         'TabTransformer: Feat.Tokens\n'
         '                + Self-Attn x3\n'
         'BiLSTM:  Bi-LSTM(128, 2L)\n'
         '          Sliding Window (7)',
         fc='#FFF3E0', ec='#E65100', fontsize=8, bold=False)

# Arrows from encoder to both heads
ax.annotate('', xy=(7.5, 3.8), xytext=(6.5, 2.7),
            arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.8))
ax.annotate('', xy=(7.5, 1.5), xytext=(6.5, 2.7),
            arrowprops=dict(arrowstyle='->', color='#B71C1C', lw=1.8))

# Task 1 head
draw_box(ax, 7.5, 3.1, 3.0, 1.5,
         'Task 1 Head\n\nFC(64)+ReLU\nFC(1)\n',
         fc='#E8F5E9', ec='#2E7D32', fontsize=8)
draw_arrow_h(ax, 10.5, 11.1, 3.85, '#2E7D32')
draw_box(ax, 11.1, 3.3, 2.6, 1.1,
         'Output 1\nEfficiency (%)',
         fc='#C8E6C9', ec='#1B5E20', fontsize=9, bold=True)

# Task 2 head
draw_box(ax, 7.5, 0.8, 3.0, 1.5,
         'Task 2 Head\n\nFC(64)+ReLU\nFC(1)\n',
         fc='#FFEBEE', ec='#B71C1C', fontsize=8)
draw_arrow_h(ax, 10.5, 11.1, 1.55, '#B71C1C')
draw_box(ax, 11.1, 0.95, 2.6, 1.1,
         'Output 2\nDHU (%)',
         fc='#FFCDD2', ec='#7F0000', fontsize=9, bold=True)

# Loss function annotation
ax.text(7.0, 5.1, 'Joint Loss:  L = 0.5 * MSE(Efficiency) + 0.5 * MSE(DHU)',
        ha='center', va='center', fontsize=9, style='italic',
        bbox=dict(fc='#FFFDE7', ec='#F57F17', lw=1.2, pad=4, boxstyle='round,pad=0.4'))

ax.set_title('Figure 3. Multi-Task Learning Framework for Garment Production Prediction',
             fontsize=11, fontweight='bold', pad=8)
fig.savefig(f'{FIG_DIR}/fig3_mtl_framework.png', facecolor='#FAFAFA')
plt.close()
print("    Saved fig3_mtl_framework.png")

# ─────────────────────────────────────────────────────────────
# FIG 4 — ARCHITECTURE DIAGRAMS (5 models)
# ─────────────────────────────────────────────────────────────
print("\n[5/8] Figure 4 — Architecture Diagrams...")

LCOLORS = {
    'input':   ('#BBDEFB', '#0D47A1'),
    'fc':      ('#C8E6C9', '#1B5E20'),
    'bn':      ('#FFF9C4', '#F57F17'),
    'act':     ('#FFE0B2', '#E65100'),
    'drop':    ('#F3E5F5', '#6A1B9A'),
    'conv':    ('#B2EBF2', '#006064'),
    'pool':    ('#D7CCC8', '#3E2723'),
    'lstm':    ('#FFCDD2', '#B71C1C'),
    'attn':    ('#E8EAF6', '#283593'),
    'norm':    ('#FFF9C4', '#F57F17'),
    'head':    ('#DCEDC8', '#33691E'),
    'output':  ('#F8BBD9', '#880E4F'),
}

def draw_v_box(ax, cx, y, w, h, label, ltype='fc', fontsize=7.5):
    fc, ec = LCOLORS.get(ltype, ('#ECEFF1','#455A64'))
    box = FancyBboxPatch((cx - w/2, y), w, h,
                         boxstyle='round,pad=0.01,rounding_size=0.03',
                         fc=fc, ec=ec, linewidth=1.4, zorder=3)
    ax.add_patch(box)
    ax.text(cx, y + h/2, label, ha='center', va='center',
            fontsize=fontsize, zorder=4, wrap=False)

def draw_v_arr(ax, cx, y1, y2, color='#424242'):
    ax.annotate('', xy=(cx, y2), xytext=(cx, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.3))

def draw_branch(ax, cx_src, y_src, cx_l, y_l, cx_r, y_r):
    ax.plot([cx_src, cx_l], [y_src, y_r], '-', color='#424242', lw=1.3)
    ax.annotate('', xy=(cx_l, y_l), xytext=(cx_l, y_r),
                arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.3))
    ax.plot([cx_src, cx_r], [y_src, y_r], '-', color='#424242', lw=1.3)
    ax.annotate('', xy=(cx_r, y_l), xytext=(cx_r, y_r),
                arrowprops=dict(arrowstyle='->', color='#B71C1C', lw=1.3))

fig, axes = plt.subplots(1, 5, figsize=(18, 9))
fig.suptitle('Figure 4. Deep Learning Architecture Diagrams for Multi-Task Production Prediction',
             fontsize=12, fontweight='bold', y=1.01)

W, H = 1.7, 0.38
GAP = 0.10
CX  = 1.0

# ── MLP ──────────────────────────────────────────────────────
ax = axes[0]
ax.set_xlim(0, 2); ax.set_ylim(0, 8.5); ax.axis('off')
ax.set_title('(a) MLP\n(Baseline)', fontsize=10, fontweight='bold', color=PALETTE['MLP'])
layers = [
    ('Input\n20 Features', 'input', 7.8),
    ('FC 256 + ReLU', 'fc', 6.8),
    ('FC 128 + ReLU', 'fc', 5.8),
    ('Shared Rep.\n(128-d)', 'fc', 4.6),
]
for lbl, ltype, y in layers:
    draw_v_box(ax, CX, y, W, H, lbl, ltype)
for i in range(len(layers)-1):
    draw_v_arr(ax, CX, layers[i][2], layers[i+1][2]+H, '#424242')
ax.plot([CX, CX-0.5], [4.6, 3.8], '-', color='#424242', lw=1.2)
ax.plot([CX, CX+0.5], [4.6, 3.8], '-', color='#424242', lw=1.2)
draw_v_box(ax, CX-0.5, 3.4, 0.82, H, 'FC 64+ReLU', 'head')
draw_v_box(ax, CX+0.5, 3.4, 0.82, H, 'FC 64+ReLU', 'head')
ax.annotate('', xy=(CX-0.5, 3.4), xytext=(CX-0.5, 3.8),
            arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.2))
ax.annotate('', xy=(CX+0.5, 3.4), xytext=(CX+0.5, 3.8),
            arrowprops=dict(arrowstyle='->', color='#B71C1C', lw=1.2))
draw_v_box(ax, CX-0.5, 2.7, 0.82, H, 'FC 1', 'output')
draw_v_box(ax, CX+0.5, 2.7, 0.82, H, 'FC 1', 'output')
for cx_ in [CX-0.5, CX+0.5]:
    draw_v_arr(ax, cx_, 3.4, 3.1, '#555')
draw_v_box(ax, CX-0.5, 2.1, 0.82, 0.45, 'Efficiency\n(%)', 'output')
draw_v_box(ax, CX+0.5, 2.1, 0.82, 0.45, 'DHU\n(%)', 'output')
for cx_ in [CX-0.5, CX+0.5]:
    draw_v_arr(ax, cx_, 2.7, 2.55, '#555')
ax.text(CX, 1.4, '54,914 params', ha='center', fontsize=8, style='italic', color='gray')

# ── DeepMLP ──────────────────────────────────────────────────
ax = axes[1]
ax.set_xlim(0, 2); ax.set_ylim(0, 8.5); ax.axis('off')
ax.set_title('(b) DeepMLP\n(BatchNorm + GELU)', fontsize=10, fontweight='bold', color=PALETTE['DeepMLP'])
dlayers = [
    ('Input 20','input',8.0),
    ('FC 512','fc',7.2), ('BN + GELU + Drop','bn',6.6),
    ('FC 256','fc',5.8), ('BN + GELU + Drop','bn',5.2),
    ('FC 128','fc',4.4), ('BN + GELU + Drop','bn',3.8),
    ('FC 64','fc',3.0),  ('BN + GELU + Drop','bn',2.4),
]
for lbl, lt, y in dlayers:
    draw_v_box(ax, CX, y, W, 0.34, lbl, lt, fontsize=7)
for i in range(len(dlayers)-1):
    draw_v_arr(ax, CX, dlayers[i][2], dlayers[i+1][2]+0.34, '#424242')
ax.plot([CX, CX-0.4], [2.4, 1.7], '-', color='#424242', lw=1.2)
ax.plot([CX, CX+0.4], [2.4, 1.7], '-', color='#424242', lw=1.2)
draw_v_box(ax, CX-0.4, 1.3, 0.72, 0.32, 'FC 32+GELU', 'head', fontsize=7)
draw_v_box(ax, CX+0.4, 1.3, 0.72, 0.32, 'FC 32+GELU', 'head', fontsize=7)
draw_v_box(ax, CX-0.4, 0.7, 0.72, 0.32, 'Efficiency', 'output', fontsize=7)
draw_v_box(ax, CX+0.4, 0.7, 0.72, 0.32, 'DHU', 'output', fontsize=7)
for cx_ in [CX-0.4, CX+0.4]:
    for y1, y2 in [(1.7, 1.62), (1.3, 1.02)]:
        draw_v_arr(ax, cx_, y1, y2, '#555')
ax.text(CX, 0.2, '188,354 params', ha='center', fontsize=8, style='italic', color='gray')

# ── CNN1D ────────────────────────────────────────────────────
ax = axes[2]
ax.set_xlim(0, 2); ax.set_ylim(0, 8.5); ax.axis('off')
ax.set_title('(c) CNN1D\n(1-D Convolution)', fontsize=10, fontweight='bold', color=PALETTE['CNN1D'])
clayers = [
    ('Input (20,)\nReshape (1,20)', 'input', 7.8),
    ('Conv1D(1->64,k=3)\n+BN + ReLU', 'conv', 6.8),
    ('Conv1D(64->128,k=3)\n+BN + ReLU', 'conv', 5.7),
    ('Conv1D(128->256,k=3)\n+BN + ReLU', 'conv', 4.6),
    ('AdaptiveAvgPool\n(256,)', 'pool', 3.7),
    ('Dropout(0.3)', 'drop', 2.9),
]
for lbl, lt, y in clayers:
    draw_v_box(ax, CX, y, W, 0.42, lbl, lt, fontsize=7)
for i in range(len(clayers)-1):
    draw_v_arr(ax, CX, clayers[i][2], clayers[i+1][2]+0.42, '#424242')
ax.plot([CX, CX-0.4], [2.9, 2.1], '-', color='#424242', lw=1.2)
ax.plot([CX, CX+0.4], [2.9, 2.1], '-', color='#424242', lw=1.2)
draw_v_box(ax, CX-0.4, 1.7, 0.72, 0.34, 'FC 64+ReLU', 'head', fontsize=7)
draw_v_box(ax, CX+0.4, 1.7, 0.72, 0.34, 'FC 64+ReLU', 'head', fontsize=7)
draw_v_box(ax, CX-0.4, 1.0, 0.72, 0.34, 'Efficiency', 'output', fontsize=7)
draw_v_box(ax, CX+0.4, 1.0, 0.72, 0.34, 'DHU', 'output', fontsize=7)
for cx_ in [CX-0.4, CX+0.4]:
    for y1, y2 in [(2.1, 2.04), (1.7, 1.34)]:
        draw_v_arr(ax, cx_, y1, y2, '#555')
ax.text(CX, 0.4, '157,442 params', ha='center', fontsize=8, style='italic', color='gray')

# ── TabTransformer ────────────────────────────────────────────
ax = axes[3]
ax.set_xlim(0, 2); ax.set_ylim(0, 8.5); ax.axis('off')
ax.set_title('(d) TabTransformer\n(Self-Attention)', fontsize=10, fontweight='bold', color=PALETTE['TabTransformer'])
tlayers = [
    ('Input 20 Features', 'input', 8.0),
    ('Feature Proj.\nLinear(1, 64)', 'fc', 7.1),
    ('+ Positional\nEmbedding', 'norm', 6.2),
    ('Transformer Block x3\n(4 heads, FFN=256)', 'attn', 5.1),
    ('LayerNorm', 'norm', 4.3),
    ('Mean Pooling\n(20 tokens -> 64)', 'pool', 3.5),
]
for lbl, lt, y in tlayers:
    draw_v_box(ax, CX, y, W, 0.44, lbl, lt, fontsize=7)
for i in range(len(tlayers)-1):
    draw_v_arr(ax, CX, tlayers[i][2], tlayers[i+1][2]+0.44, '#424242')
ax.plot([CX, CX-0.4], [3.5, 2.7], '-', color='#424242', lw=1.2)
ax.plot([CX, CX+0.4], [3.5, 2.7], '-', color='#424242', lw=1.2)
draw_v_box(ax, CX-0.4, 2.3, 0.72, 0.34, 'FC 64+GELU', 'head', fontsize=7)
draw_v_box(ax, CX+0.4, 2.3, 0.72, 0.34, 'FC 64+GELU', 'head', fontsize=7)
draw_v_box(ax, CX-0.4, 1.6, 0.72, 0.34, 'Efficiency', 'output', fontsize=7)
draw_v_box(ax, CX+0.4, 1.6, 0.72, 0.34, 'DHU', 'output', fontsize=7)
for cx_ in [CX-0.4, CX+0.4]:
    for y1, y2 in [(2.7, 2.64), (2.3, 1.94)]:
        draw_v_arr(ax, cx_, y1, y2, '#555')
ax.text(CX, 0.9, '159,938 params', ha='center', fontsize=8, style='italic', color='gray')

# ── BiLSTM ────────────────────────────────────────────────────
ax = axes[4]
ax.set_xlim(0, 2); ax.set_ylim(0, 8.5); ax.axis('off')
ax.set_title('(e) BiLSTM\n(Bidirectional LSTM)', fontsize=10, fontweight='bold', color=PALETTE['BiLSTM'])
llayers = [
    ('Input Sequence\n(7 x 20)', 'input', 7.9),
    ('Sliding Window\nConstruction', 'norm', 7.0),
    ('BiLSTM Layer 1\nhidden=128 (x2)', 'lstm', 6.0),
    ('BiLSTM Layer 2\nhidden=128 (x2)', 'lstm', 5.0),
    ('Last Time Step\n(256-d)', 'pool', 4.1),
    ('LayerNorm\n+ Dropout(0.3)', 'drop', 3.2),
]
for lbl, lt, y in llayers:
    draw_v_box(ax, CX, y, W, 0.44, lbl, lt, fontsize=7)
for i in range(len(llayers)-1):
    draw_v_arr(ax, CX, llayers[i][2], llayers[i+1][2]+0.44, '#424242')
ax.plot([CX, CX-0.4], [3.2, 2.4], '-', color='#424242', lw=1.2)
ax.plot([CX, CX+0.4], [3.2, 2.4], '-', color='#424242', lw=1.2)
draw_v_box(ax, CX-0.4, 2.0, 0.72, 0.34, 'FC 64+ReLU', 'head', fontsize=7)
draw_v_box(ax, CX+0.4, 2.0, 0.72, 0.34, 'FC 64+ReLU', 'head', fontsize=7)
draw_v_box(ax, CX-0.4, 1.3, 0.72, 0.34, 'Efficiency', 'output', fontsize=7)
draw_v_box(ax, CX+0.4, 1.3, 0.72, 0.34, 'DHU', 'output', fontsize=7)
for cx_ in [CX-0.4, CX+0.4]:
    for y1, y2 in [(2.4, 2.34), (2.0, 1.64)]:
        draw_v_arr(ax, cx_, y1, y2, '#555')
ax.text(CX, 0.6, '582,402 params', ha='center', fontsize=8, style='italic', color='gray')

legend_els = [
    mpatches.Patch(fc=LCOLORS['input'][0],  ec=LCOLORS['input'][1],  label='Input Layer'),
    mpatches.Patch(fc=LCOLORS['fc'][0],     ec=LCOLORS['fc'][1],     label='Fully Connected'),
    mpatches.Patch(fc=LCOLORS['bn'][0],     ec=LCOLORS['bn'][1],     label='BatchNorm / LayerNorm'),
    mpatches.Patch(fc=LCOLORS['conv'][0],   ec=LCOLORS['conv'][1],   label='Conv1D'),
    mpatches.Patch(fc=LCOLORS['lstm'][0],   ec=LCOLORS['lstm'][1],   label='LSTM'),
    mpatches.Patch(fc=LCOLORS['attn'][0],   ec=LCOLORS['attn'][1],   label='Attention'),
    mpatches.Patch(fc=LCOLORS['pool'][0],   ec=LCOLORS['pool'][1],   label='Pooling'),
    mpatches.Patch(fc=LCOLORS['drop'][0],   ec=LCOLORS['drop'][1],   label='Dropout'),
    mpatches.Patch(fc=LCOLORS['head'][0],   ec=LCOLORS['head'][1],   label='Task Head'),
    mpatches.Patch(fc=LCOLORS['output'][0], ec=LCOLORS['output'][1], label='Output'),
]
fig.legend(handles=legend_els, loc='lower center', ncol=5,
           fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.04))
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig4_architectures.png')
plt.close()
print("    Saved fig4_architectures.png")

# ─────────────────────────────────────────────────────────────
# TRAINING — Save losses and predictions
# ─────────────────────────────────────────────────────────────
print("\n[6/8] Training all models (saves losses + predictions)...")

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = torch.FloatTensor(X), torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class SequenceDataset(Dataset):
    def __init__(self, X, y, seq_len=7):
        self.X, self.y, self.seq_len = torch.FloatTensor(X), torch.FloatTensor(y), seq_len
        self.idx = list(range(seq_len - 1, len(X)))
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        return self.X[j - self.seq_len + 1:j + 1], self.y[j]

def make_loaders(Xtr, ytr, Xv, yv, Xte, yte, bs, sl, lstm=False):
    Cls = SequenceDataset if lstm else TabularDataset
    kw  = {'seq_len': sl} if lstm else {}
    return (DataLoader(Cls(Xtr, ytr, **kw), bs, shuffle=not lstm),
            DataLoader(Cls(Xv,  yv,  **kw), bs, shuffle=False),
            DataLoader(Cls(Xte, yte, **kw), bs, shuffle=False))

tab_tr, tab_vl, tab_te = make_loaders(X_train,y_train,X_val,y_val,X_test,y_test,64,7,False)
seq_tr, seq_vl, seq_te = make_loaders(X_train,y_train,X_val,y_val,X_test,y_test,64,7,True)

class MLP(nn.Module):
    def __init__(self, d, h=(256,128)):
        super().__init__()
        layers, cur = [], d
        for hh in h:
            layers += [nn.Linear(cur, hh), nn.ReLU()]; cur = hh
        self.bb = nn.Sequential(*layers)
        self.eh = nn.Sequential(nn.Linear(cur,64), nn.ReLU(), nn.Linear(64,1))
        self.dh = nn.Sequential(nn.Linear(cur,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self, x): f=self.bb(x); return self.eh(f).squeeze(-1), self.dh(f).squeeze(-1)

class DeepMLP(nn.Module):
    def __init__(self, d, h=(512,256,128,64), drop=0.3):
        super().__init__()
        self.proj = nn.Linear(d, h[0])
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Linear(h[i],h[i+1]), nn.BatchNorm1d(h[i+1]), nn.GELU(), nn.Dropout(drop))
            for i in range(len(h)-1)])
        self.eh = nn.Sequential(nn.Linear(h[-1],32), nn.GELU(), nn.Linear(32,1))
        self.dh = nn.Sequential(nn.Linear(h[-1],32), nn.GELU(), nn.Linear(32,1))
    def forward(self, x):
        x = F.gelu(self.proj(x))
        for b in self.blocks: x = b(x)
        return self.eh(x).squeeze(-1), self.dh(x).squeeze(-1)

class CNN1D(nn.Module):
    def __init__(self, d, drop=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1,64,3,padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64,128,3,padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128,256,3,padding=1), nn.BatchNorm1d(256), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool1d(1); self.drop = nn.Dropout(drop)
        self.eh = nn.Sequential(nn.Linear(256,64),nn.ReLU(),nn.Dropout(drop),nn.Linear(64,1))
        self.dh = nn.Sequential(nn.Linear(256,64),nn.ReLU(),nn.Dropout(drop),nn.Linear(64,1))
    def forward(self, x):
        x = self.pool(self.conv(x.unsqueeze(1))).squeeze(-1); x = self.drop(x)
        return self.eh(x).squeeze(-1), self.dh(x).squeeze(-1)

class _TB(nn.Module):
    def __init__(self, d, heads, ffn, drop):
        super().__init__()
        self.attn=nn.MultiheadAttention(d,heads,dropout=drop,batch_first=True)
        self.n1=nn.LayerNorm(d); self.n2=nn.LayerNorm(d)
        self.ff=nn.Sequential(nn.Linear(d,ffn),nn.GELU(),nn.Dropout(drop),nn.Linear(ffn,d))
        self.dr=nn.Dropout(drop)
    def forward(self, x):
        a,_=self.attn(x,x,x); x=self.n1(x+self.dr(a)); return self.n2(x+self.dr(self.ff(x)))

class TabTransformer(nn.Module):
    def __init__(self, d, dm=64, heads=4, nl=3, drop=0.2):
        super().__init__()
        self.fp=nn.Linear(1,dm); self.pe=nn.Parameter(torch.randn(1,d,dm)*0.02)
        self.blocks=nn.ModuleList([_TB(dm,heads,dm*4,drop) for _ in range(nl)])
        self.norm=nn.LayerNorm(dm)
        self.eh=nn.Sequential(nn.Linear(dm,64),nn.GELU(),nn.Dropout(drop),nn.Linear(64,1))
        self.dh=nn.Sequential(nn.Linear(dm,64),nn.GELU(),nn.Dropout(drop),nn.Linear(64,1))
    def forward(self, x):
        x=self.fp(x.unsqueeze(-1))+self.pe
        for b in self.blocks: x=b(x)
        x=self.norm(x).mean(1)
        return self.eh(x).squeeze(-1), self.dh(x).squeeze(-1)

class BiLSTM(nn.Module):
    def __init__(self, d, hid=128, nl=2, drop=0.3):
        super().__init__()
        self.lstm=nn.LSTM(d,hid,nl,batch_first=True,dropout=drop if nl>1 else 0,bidirectional=True)
        self.norm=nn.LayerNorm(hid*2); self.drop=nn.Dropout(drop)
        self.eh=nn.Sequential(nn.Linear(hid*2,64),nn.ReLU(),nn.Dropout(drop),nn.Linear(64,1))
        self.dh=nn.Sequential(nn.Linear(hid*2,64),nn.ReLU(),nn.Dropout(drop),nn.Linear(64,1))
    def forward(self, x):
        out,_=self.lstm(x); x=self.norm(self.drop(out[:,-1,:]))
        return self.eh(x).squeeze(-1), self.dh(x).squeeze(-1)

def mt_loss(pe, pd, te, td, we=0.5, wd=0.5):
    return we*F.mse_loss(pe,te) + wd*F.mse_loss(pd,td)

def run_epoch(model, loader, opt, device, train=True):
    model.train() if train else model.eval()
    tot, pes, pds, tes, tds = 0., [], [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for Xb, yb in loader:
            Xb=Xb.to(device); te_=yb[:,0].to(device); td_=yb[:,1].to(device)
            if train: opt.zero_grad()
            pe_, pd_ = model(Xb)
            loss = mt_loss(pe_, pd_, te_, td_)
            if train: loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            tot += loss.item()
            pes.extend(pe_.detach().cpu().numpy()); pds.extend(pd_.detach().cpu().numpy())
            tes.extend(te_.cpu().numpy()); tds.extend(td_.cpu().numpy())
    return tot/len(loader), np.array(pes), np.array(pds), np.array(tes), np.array(tds)

def train_all():
    dev = CONFIG['device']
    specs = [
        ('MLP',            MLP(input_dim),            tab_tr,tab_vl,tab_te),
        ('DeepMLP',        DeepMLP(input_dim),         tab_tr,tab_vl,tab_te),
        ('CNN1D',          CNN1D(input_dim),            tab_tr,tab_vl,tab_te),
        ('TabTransformer', TabTransformer(input_dim),   tab_tr,tab_vl,tab_te),
        ('BiLSTM',         BiLSTM(input_dim),           seq_tr,seq_vl,seq_te),
    ]
    all_res = {}
    for name, mdl, tr, vl, te in specs:
        print(f"    Training {name}...", end='', flush=True)
        mdl = mdl.to(dev)
        opt   = optim.AdamW(mdl.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = ReduceLROnPlateau(opt,'min',factor=0.5,patience=8,min_lr=1e-6)
        best_vl, pat, best_st = float('inf'), 0, None
        tr_losses, vl_losses = [], []
        t0 = time.time()
        for ep in range(1, CONFIG['epochs']+1):
            tl,_,_,_,_ = run_epoch(mdl, tr, opt, dev, train=True)
            vl_,_,_,_,_ = run_epoch(mdl, vl, opt, dev, train=False)
            sched.step(vl_); tr_losses.append(tl); vl_losses.append(vl_)
            if vl_ < best_vl:
                best_vl, pat = vl_, 0
                best_st = {k:v.clone() for k,v in mdl.state_dict().items()}
            else:
                pat += 1
            if pat >= CONFIG['patience']: break
        mdl.load_state_dict(best_st)
        _,pe,pd_,tes,tds = run_epoch(mdl, te, opt, dev, train=False)
        em, es = tgt_scaler.mean_[0], tgt_scaler.scale_[0]
        dm, ds = tgt_scaler.mean_[1], tgt_scaler.scale_[1]
        def metrics(p, t, m, s):
            po=p*s+m; to_=t*s+m
            return (mean_absolute_error(to_,po),
                    math.sqrt(mean_squared_error(to_,po)),
                    r2_score(to_,po))
        emae,ermse,er2 = metrics(pe, tes, em, es)
        dmae,drmse,dr2 = metrics(pd_, tds, dm, ds)
        print(f" done ({time.time()-t0:.0f}s) | EffR2={er2:.3f} DHUR2={dr2:.3f}")
        all_res[name] = dict(
            tr_losses=tr_losses, vl_losses=vl_losses,
            pred_eff=pe, pred_dhu=pd_, true_eff=tes, true_dhu=tds,
            eff_mae=emae, eff_rmse=ermse, eff_r2=er2,
            dhu_mae=dmae, dhu_rmse=drmse, dhu_r2=dr2,
            avg_r2=(er2+dr2)/2,
            em=em, es=es, dm=dm, ds=ds,
            train_time=time.time()-t0
        )
    return all_res

results = train_all()
model_order = sorted(results.keys(), key=lambda k: -results[k]['avg_r2'])

# ─────────────────────────────────────────────────────────────
# FIG 5 — TRAINING/VALIDATION LOSS CURVES
# ─────────────────────────────────────────────────────────────
print("\n[7/8] Figure 5 — Loss Curves + Figure 6 — Performance Comparison...")
fig, axes = plt.subplots(1, 5, figsize=(17, 4), sharey=False)
fig.suptitle('Figure 5. Training and Validation Loss Convergence for All Models',
             fontsize=11, fontweight='bold')
for i, (name, ax) in enumerate(zip(
        ['MLP','DeepMLP','CNN1D','TabTransformer','BiLSTM'], axes)):
    r  = results[name]
    ep = range(1, len(r['tr_losses'])+1)
    ax.plot(ep, r['tr_losses'], color=PALETTE[name], linewidth=1.6, label='Train')
    ax.plot(ep, r['vl_losses'], color=PALETTE[name], linewidth=1.6,
            linestyle='--', alpha=0.75, label='Val')
    ax.set_title(f'({chr(97+i)}) {name}', fontsize=9, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=8); ax.set_ylabel('Loss (MSE)' if i==0 else '', fontsize=8)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig5_loss_curves.png')
plt.close()

# ─────────────────────────────────────────────────────────────
# FIG 6 — PERFORMANCE COMPARISON
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('Figure 6. Multi-Model Performance Comparison on Test Set',
             fontsize=12, fontweight='bold')

names = model_order
cols  = [PALETTE[n] for n in names]

eff_r2   = [results[n]['eff_r2']   for n in names]
dhu_r2   = [results[n]['dhu_r2']   for n in names]
avg_r2   = [results[n]['avg_r2']   for n in names]
eff_mae  = [results[n]['eff_mae']  for n in names]
dhu_mae  = [results[n]['dhu_mae']  for n in names]
eff_rmse = [results[n]['eff_rmse'] for n in names]
dhu_rmse = [results[n]['dhu_rmse'] for n in names]
times    = [results[n]['train_time'] for n in names]

x = np.arange(len(names)); w = 0.32

# (a) R2 comparison
ax = axes[0,0]
b1 = ax.bar(x - w/2, eff_r2, w, label='Efficiency R\u00b2', color='#1565C0', alpha=0.85, edgecolor='white')
b2 = ax.bar(x + w/2, dhu_r2, w, label='DHU R\u00b2',        color='#E65100', alpha=0.85, edgecolor='white')
ax.axhline(0, color='black', lw=0.8, linestyle='--')
ax.set_title('(a) R\u00b2 Score Comparison (higher = better)', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha='right')
ax.set_ylabel('R\u00b2'); ax.legend(); ax.grid(True, alpha=0.3, axis='y')
for bar in b1:
    v = bar.get_height()
    if v > 0: ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f'{v:.3f}', ha='center', fontsize=7)
for bar in b2:
    v = bar.get_height()
    if v > 0: ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f'{v:.3f}', ha='center', fontsize=7)

# (b) MAE comparison
ax = axes[0,1]
ax.bar(x - w/2, eff_mae, w, label='Efficiency MAE (%)', color='#2E7D32', alpha=0.85, edgecolor='white')
ax.bar(x + w/2, dhu_mae, w, label='DHU MAE (%)',        color='#6A1B9A', alpha=0.85, edgecolor='white')
ax.set_title('(b) MAE Comparison (lower = better)', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha='right')
ax.set_ylabel('MAE (%)'); ax.legend(); ax.grid(True, alpha=0.3, axis='y')

# (c) Average R2 ranking
ax = axes[1,0]
bar_cols = ['gold' if n == model_order[0] else c for n, c in zip(names, cols)]
bars = ax.bar(names, avg_r2, color=bar_cols, edgecolor='black', linewidth=0.7, alpha=0.9)
ax.set_title('(c) Average R\u00b2 (Efficiency + DHU) / 2', fontweight='bold')
ax.set_ylabel('Average R\u00b2'); ax.grid(True, alpha=0.3, axis='y')
ax.set_xticklabels(names, rotation=20, ha='right')
for bar, v in zip(bars, avg_r2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f'{v:.4f}', ha='center', fontsize=8, fontweight='bold')
ax.set_ylim(min(0, min(avg_r2))-0.05, max(avg_r2)+0.12)
best_patch = mpatches.Patch(color='gold', label=f'Best: {model_order[0]}')
ax.legend(handles=[best_patch], fontsize=8)

# (d) RMSE comparison
ax = axes[1,1]
ax.bar(x - w/2, eff_rmse, w, label='Efficiency RMSE (%)', color='#0277BD', alpha=0.85, edgecolor='white')
ax.bar(x + w/2, dhu_rmse, w, label='DHU RMSE (%)',        color='#AD1457', alpha=0.85, edgecolor='white')
ax.set_title('(d) RMSE Comparison (lower = better)', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha='right')
ax.set_ylabel('RMSE (%)'); ax.legend(); ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig6_performance_comparison.png')
plt.close()
print("    Saved fig5_loss_curves.png + fig6_performance_comparison.png")

# ─────────────────────────────────────────────────────────────
# FIG 7 — PREDICTED VS ACTUAL (CNN1D + BiLSTM side by side)
# ─────────────────────────────────────────────────────────────
print("\n[8/8] Figure 7 — Predicted vs Actual & Figure 8 — Residuals...")
best = model_order[0]
r    = results[best]
em, es, dm, ds = r['em'], r['es'], r['dm'], r['ds']

pe_o  = r['pred_eff'] * es + em
te_o  = r['true_eff'] * es + em
pd_o  = r['pred_dhu'] * ds + dm
td_o  = r['true_dhu'] * ds + dm

bilstm_r = results['BiLSTM']
pe_bi = bilstm_r['pred_eff'] * bilstm_r['es'] + bilstm_r['em']
te_bi = bilstm_r['true_eff'] * bilstm_r['es'] + bilstm_r['em']

fig, axes = plt.subplots(2, 2, figsize=(13, 11))
fig.suptitle(f'Figure 7. Predicted vs. Actual Values — {best} (Best Overall) and BiLSTM (Best Efficiency)',
             fontsize=11, fontweight='bold')

def pva_plot(ax, true, pred, title, color, xlabel, ylabel):
    ax.scatter(true, pred, alpha=0.35, s=14, color=color, edgecolors='none')
    lo = min(true.min(), pred.min()) - 2
    hi = max(true.max(), pred.max()) + 2
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1.8, label='Ideal (y=x)')
    z = np.polyfit(true, pred, 1)
    p = np.poly1d(z)
    xline = np.linspace(lo, hi, 100)
    ax.plot(xline, p(xline), 'k-', lw=1.2, alpha=0.6, label='Trend')
    r2v = r2_score(true, pred)
    maev = mean_absolute_error(true, pred)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f'{title}\nR\u00b2 = {r2v:.4f}  |  MAE = {maev:.3f}%', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.text(0.04, 0.94, f'n = {len(true)}', transform=ax.transAxes, fontsize=8, va='top')

pva_plot(axes[0,0], te_o, pe_o,
         f'(a) {best} — Efficiency',
         PALETTE[best], 'Actual Efficiency (%)', 'Predicted Efficiency (%)')
pva_plot(axes[0,1], td_o, pd_o,
         f'(b) {best} — DHU',
         PALETTE[best], 'Actual DHU (%)', 'Predicted DHU (%)')
pva_plot(axes[1,0], te_bi, pe_bi,
         '(c) BiLSTM — Efficiency (Best R\u00b2)',
         PALETTE['BiLSTM'], 'Actual Efficiency (%)', 'Predicted Efficiency (%)')
res_dhu_bi = results['BiLSTM']
pd_bi = res_dhu_bi['pred_dhu'] * res_dhu_bi['ds'] + res_dhu_bi['dm']
td_bi = res_dhu_bi['true_dhu'] * res_dhu_bi['ds'] + res_dhu_bi['dm']
pva_plot(axes[1,1], td_bi, pd_bi,
         '(d) BiLSTM — DHU',
         PALETTE['BiLSTM'], 'Actual DHU (%)', 'Predicted DHU (%)')

plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig7_predicted_vs_actual.png')
plt.close()

# ─────────────────────────────────────────────────────────────
# FIG 8 — RESIDUAL ANALYSIS
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle(f'Figure 8. Residual Analysis — {best} (All Models\' Efficiency Residual Distribution)',
             fontsize=11, fontweight='bold')

# (a) CNN1D Efficiency residuals vs predicted
res_eff = pe_o - te_o
ax = axes[0,0]
ax.scatter(pe_o, res_eff, alpha=0.3, s=12, color=PALETTE[best], edgecolors='none')
ax.axhline(0, color='red', lw=1.5, linestyle='--')
ax.set_xlabel('Predicted Efficiency (%)'); ax.set_ylabel('Residual (%)')
ax.set_title(f'(a) {best} — Efficiency Residuals vs. Fitted', fontweight='bold')
ax.grid(True, alpha=0.3)

# (b) CNN1D DHU residuals vs predicted
res_dhu_v = pd_o - td_o
ax = axes[0,1]
ax.scatter(pd_o, res_dhu_v, alpha=0.3, s=12, color=PALETTE[best], edgecolors='none')
ax.axhline(0, color='red', lw=1.5, linestyle='--')
ax.set_xlabel('Predicted DHU (%)'); ax.set_ylabel('Residual (%)')
ax.set_title(f'(b) {best} — DHU Residuals vs. Fitted', fontweight='bold')
ax.grid(True, alpha=0.3)

# (c) Residual distributions — Efficiency (all models)
ax = axes[1,0]
for n in model_order:
    rr = results[n]
    res_e = (rr['pred_eff'] * rr['es'] + rr['em']) - (rr['true_eff'] * rr['es'] + rr['em'])
    ax.hist(res_e, bins=35, alpha=0.5, label=n, color=PALETTE[n], edgecolor='none', density=True)
ax.axvline(0, color='black', lw=1.2, linestyle='--')
ax.set_xlabel('Residual — Efficiency (%)'); ax.set_ylabel('Density')
ax.set_title('(c) Efficiency Residual Distribution (All Models)', fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (d) Residual distributions — DHU (all models)
ax = axes[1,1]
for n in model_order:
    rr = results[n]
    res_d = (rr['pred_dhu'] * rr['ds'] + rr['dm']) - (rr['true_dhu'] * rr['ds'] + rr['dm'])
    ax.hist(res_d, bins=35, alpha=0.5, label=n, color=PALETTE[n], edgecolor='none', density=True)
ax.axvline(0, color='black', lw=1.2, linestyle='--')
ax.set_xlabel('Residual — DHU (%)'); ax.set_ylabel('Density')
ax.set_title('(d) DHU Residual Distribution (All Models)', fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig8_residual_analysis.png')
plt.close()
print("    Saved fig7_predicted_vs_actual.png + fig8_residual_analysis.png")

# Save results summary for paper generator
import json
summary = {}
for name, r in results.items():
    summary[name] = {
        'eff_mae': round(r['eff_mae'],3), 'eff_rmse': round(r['eff_rmse'],3),
        'eff_r2':  round(r['eff_r2'],4),
        'dhu_mae': round(r['dhu_mae'],3), 'dhu_rmse': round(r['dhu_rmse'],3),
        'dhu_r2':  round(r['dhu_r2'],4),
        'avg_r2':  round(r['avg_r2'],4),
        'train_time': round(r['train_time'],1),
        'n_epochs': len(r['tr_losses']),
    }
summary['best_model']   = model_order[0]
summary['model_order']  = model_order
summary['n_train']      = len(X_train)
summary['n_val']        = len(X_val)
summary['n_test']       = len(X_test)
summary['n_features']   = input_dim
summary['n_samples']    = n
summary['eff_mean']     = round(float(df['AchievedEfficiency'].mean()), 2)
summary['eff_std']      = round(float(df['AchievedEfficiency'].std()),  2)
summary['dhu_mean']     = round(float(df['dhu'].mean()), 2)
summary['dhu_std']      = round(float(df['dhu'].std()),  2)

with open(f'{FIG_DIR}/results_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n  All 8 figures saved to: {FIG_DIR}/")
print(f"  Best model: {model_order[0]}  (Avg R2={summary[model_order[0]]['avg_r2']})")
print("=" * 60)
