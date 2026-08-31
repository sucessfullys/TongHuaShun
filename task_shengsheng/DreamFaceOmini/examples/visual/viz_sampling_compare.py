"""
Precise comparison of timestep sampling strategies (from actual source code):
  1. DreamFaceOmini: uniform index(1) + exp-shift table + Gaussian loss weight
  2. LongCat:        logit-normal(B) in σ-space + dynamic shift + no weight
  3. FireRed:        logit-normal(B) + linear σ-table shift + SD3 loss weight
  4. Seedream:       logit-normal(B) + resolution-aware shift (3 resolutions)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math

plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "text.color": "#e0e0e0",
    "xtick.color": "#b0b0b0",
    "ytick.color": "#b0b0b0",
    "grid.color": "#2a2a5a",
    "grid.alpha": 0.5,
    "legend.facecolor": "#16213e",
    "legend.edgecolor": "#444",
    "font.size": 11,
})

N_SAMPLES = 100_000
N_STEPS = 1000
BATCH_SIZE = 4

def exp_shift_arr(sigma, mu):
    """Vectorized exp-shift: exp(mu) / (exp(mu) + 1/σ - 1)"""
    e_mu = math.exp(mu)
    return e_mu / (e_mu + 1.0 / sigma - 1.0)

def calculate_shift(seq_len, base_seq_len=256, max_seq_len=4096,
                    base_shift=0.5, max_shift=1.15):
    """SD3/FLUX-style dynamic mu based on image sequence length."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return seq_len * m + b

def weighted_hist(ax, sigmas, weights, bins, color, label, alpha=0.6):
    """Plot a weighted histogram as a line (efficient, no KDE)."""
    hist_w, edges = np.histogram(sigmas, bins=bins, weights=weights, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.fill_between(centers, hist_w, alpha=alpha * 0.5, color=color)
    ax.plot(centers, hist_w, color=color, lw=2, label=label)

# ==========================================================
# 1. DreamFaceOmini (from flow_match.py + loss.py)
# ==========================================================
MU_OURS = 0.8
sigmas_table = np.linspace(1.0, 1.0 / N_STEPS, N_STEPS)
sigmas_table = exp_shift_arr(sigmas_table, MU_OURS)
timesteps_table = sigmas_table * N_STEPS

ids_ours = np.random.randint(0, N_STEPS, size=N_SAMPLES)
sigmas_ours = sigmas_table[ids_ours]

gauss_weights_all = np.exp(-2 * ((timesteps_table - N_STEPS / 2) / N_STEPS) ** 2)
gauss_weights_all -= gauss_weights_all.min()
gauss_weights_all *= (N_STEPS / gauss_weights_all.sum())
weights_ours = gauss_weights_all[ids_ours]

# ==========================================================
# 2. LongCat (from train_edit.py)
# ==========================================================
MU_LONGCAT = 0.8
z_lc = np.random.randn(N_SAMPLES)
sigmas_longcat = 1.0 / (1.0 + np.exp(-z_lc))
sigmas_longcat = exp_shift_arr(sigmas_longcat, MU_LONGCAT)

# ==========================================================
# 3. FireRed (from forward_step.py)
# ==========================================================
LOGIT_MEAN_FR, LOGIT_STD_FR = 0.0, 1.0
MU_FIRERED = 0.83

z_fr = np.random.randn(N_SAMPLES)
u_fr = 1.0 / (1.0 + np.exp(-(LOGIT_MEAN_FR + LOGIT_STD_FR * z_fr)))
fr_sigmas_linear = np.linspace(1.0, 1.0 / N_STEPS, N_STEPS)
fr_sigmas_shifted = exp_shift_arr(fr_sigmas_linear, MU_FIRERED)
indices_fr = (u_fr * N_STEPS).astype(int).clip(0, N_STEPS - 1)
sigmas_firered = fr_sigmas_shifted[indices_fr]
sd3_w_fr = 1.0 / np.clip(sigmas_firered, 1e-4, None)

# ==========================================================
# 4. Seedream (from Seedream 3.0 paper: logit-normal + resolution-aware shift)
#    Key: mu is NOT fixed — it's computed from image resolution via calculate_shift
# ==========================================================
RESOLUTIONS = {
    "512²":  (512,  512),
    "1024²": (1024, 1024),
    "2048²": (2048, 2048),
}
SEED_COLORS = ["#a29bfe", "#6c5ce7", "#2d1b69"]

seed_sigmas = {}
seed_mus = {}
for res_label, (h, w) in RESOLUTIONS.items():
    latent_h, latent_w = h // 8, w // 8
    seq_len = (latent_h // 2) * (latent_w // 2)
    mu = calculate_shift(seq_len)
    seed_mus[res_label] = mu
    z = np.random.randn(N_SAMPLES)
    s = 1.0 / (1.0 + np.exp(-z))
    s = exp_shift_arr(s, mu)
    seed_sigmas[res_label] = s

# ==========================================================
# Plot: 4×2 layout
# ==========================================================
fig, axes = plt.subplots(4, 2, figsize=(16, 22))

colors = ["#00d2ff", "#ff6b6b", "#ffd93d"]
labels = ["DreamFaceOmini", "LongCat", "FireRed"]
bins = np.linspace(0, 1, 150)

# ── Panel 1: Raw σ sampling distribution (3 methods) ─────
ax = axes[0, 0]
ax.hist(sigmas_ours, bins=bins, density=True, alpha=0.45, color=colors[0], label=labels[0])
ax.hist(sigmas_longcat, bins=bins, density=True, alpha=0.45, color=colors[1], label=labels[1])
ax.hist(sigmas_firered, bins=bins, density=True, alpha=0.45, color=colors[2], label=labels[2])
ax.set_xlabel("σ (noise level)")
ax.set_ylabel("Density")
ax.set_title("① Raw σ Sampling Distribution", fontsize=13, fontweight="bold")
ax.legend(framealpha=0.8, fontsize=9)
ax.set_xlim(0, 1)
ax.grid(True)

# ── Panel 2: Seedream resolution-aware shift ──────────────
ax = axes[0, 1]
for (res_label, s), sc in zip(seed_sigmas.items(), SEED_COLORS):
    mu = seed_mus[res_label]
    ax.hist(s, bins=bins, density=True, alpha=0.4, color=sc,
            label=f"Seedream {res_label} (μ={mu:.2f})")
ax.hist(sigmas_longcat, bins=bins, density=True, alpha=0.25, color=colors[1],
        label=f"LongCat 1024² (μ={MU_LONGCAT:.2f})", histtype="step", lw=1.5)
ax.set_xlabel("σ (noise level)")
ax.set_ylabel("Density")
ax.set_title("② Seedream: Resolution-Aware σ Shift", fontsize=13, fontweight="bold")
ax.legend(framealpha=0.8, fontsize=8)
ax.set_xlim(0, 1)
ax.grid(True)

# ── Panel 3: Effective training focus (sampling × weight) ─
ax = axes[1, 0]
weighted_hist(ax, sigmas_ours, weights_ours, bins, colors[0],
              f"{labels[0]}\n(uniform idx × Gaussian)")
weighted_hist(ax, sigmas_longcat, np.ones_like(sigmas_longcat), bins, colors[1],
              f"{labels[1]}\n(logit-normal, no weight)")
weighted_hist(ax, sigmas_firered, sd3_w_fr, bins, colors[2],
              f"{labels[2]}\n(logit-normal × 1/σ)")
seed_1024 = seed_sigmas["1024²"]
weighted_hist(ax, seed_1024, np.ones_like(seed_1024), bins, SEED_COLORS[1],
              f"Seedream 1024²\n(logit-normal, no weight)")
ax.set_xlabel("σ (noise level)")
ax.set_ylabel("Effective Gradient Contribution (a.u.)")
ax.set_title("③ Effective Training Focus (sampling × loss weight)", fontsize=13, fontweight="bold")
ax.legend(fontsize=7, framealpha=0.8, loc="upper right")
ax.set_xlim(0, 1)
ax.grid(True)

# ── Panel 4: Loss weight curves ──────────────────────────
ax = axes[1, 1]
sigma_range = np.linspace(0.01, 0.99, 500)
ts_range = sigma_range * N_STEPS

gauss_w = np.exp(-2 * ((ts_range - N_STEPS / 2) / N_STEPS) ** 2)
gauss_w -= gauss_w.min()
gauss_w /= gauss_w.max()

sd3_w = 1.0 / sigma_range
sd3_w /= sd3_w.max()

ax.plot(sigma_range, gauss_w, color=colors[0], lw=2.5,
        label=f"{labels[0]}: Gaussian weight")
ax.axhline(1.0, color=colors[1], lw=2.5, ls="--",
           label=f"{labels[1]}: No weight (=1)")
ax.plot(sigma_range, sd3_w, color=colors[2], lw=2.5,
        label=f"{labels[2]}: 1/σ weight (SD3)")
ax.axhline(1.0, color=SEED_COLORS[1], lw=2, ls=":",
           label="Seedream: No weight (=1)")
ax.set_xlabel("σ (noise level)")
ax.set_ylabel("Normalized Loss Weight")
ax.set_title("④ Loss Weighting Functions", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, framealpha=0.8)
ax.set_xlim(0, 1)
ax.set_ylim(-0.05, 1.15)
ax.grid(True)

# ── Panel 5: ID loss weight interaction ──────────────────
ax = axes[2, 0]
sigma2_w = sigma_range ** 2
sigma2_w_norm = sigma2_w / sigma2_w.max()
one_minus_sigma = 1.0 - sigma_range

ax.plot(sigma_range, gauss_w, color=colors[0], lw=2, alpha=0.6, ls="--",
        label="Gaussian training_weight")
ax.plot(sigma_range, sigma2_w_norm, color="#ff9f43", lw=2.5,
        label="σ² (FireRed ID weight)")
ax.plot(sigma_range, one_minus_sigma, color="#a29bfe", lw=2.5,
        label="(1−σ) (original ID weight)")
combined = gauss_w * sigma2_w_norm
combined /= combined.max()
ax.plot(sigma_range, combined, color="#ff6348", lw=2.5, ls="-.",
        label="Gaussian × σ² (effective ID focus)")
ax.set_xlabel("σ (noise level)")
ax.set_ylabel("Normalized Weight")
ax.set_title("⑤ ID Loss Weight × Training Weight Interaction", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, framealpha=0.8)
ax.set_xlim(0, 1)
ax.set_ylim(-0.05, 1.15)
ax.grid(True)

# ── Panel 6: Per-batch σ diversity ────────────────────────
ax = axes[2, 1]
rng = np.random.default_rng(42)

n_batches = 500
ours_std, longcat_std, firered_std, seed_std = [], [], [], []
mu_seed_1024 = seed_mus["1024²"]
for _ in range(n_batches):
    ours_std.append(0.0)

    z = rng.standard_normal(BATCH_SIZE)
    s = 1.0 / (1.0 + np.exp(-z))
    s = exp_shift_arr(s, MU_LONGCAT)
    longcat_std.append(float(np.std(s)))

    z = rng.standard_normal(BATCH_SIZE)
    u = 1.0 / (1.0 + np.exp(-z))
    idx = (u * N_STEPS).astype(int).clip(0, N_STEPS - 1)
    firered_std.append(float(np.std(fr_sigmas_shifted[idx])))

    z = rng.standard_normal(BATCH_SIZE)
    s = 1.0 / (1.0 + np.exp(-z))
    s = exp_shift_arr(s, mu_seed_1024)
    seed_std.append(float(np.std(s)))

box_labels = ["DreamFaceOmini", "LongCat", "FireRed", "Seedream"]
box_colors = [colors[0], colors[1], colors[2], SEED_COLORS[1]]
bp = ax.boxplot(
    [ours_std, longcat_std, firered_std, seed_std],
    tick_labels=box_labels, patch_artist=True, widths=0.5,
    medianprops=dict(color="white", lw=2),
    whiskerprops=dict(color="#888"),
    capprops=dict(color="#888"),
    flierprops=dict(marker=".", markersize=2, markerfacecolor="#666"),
)
for patch, c in zip(bp["boxes"], box_colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)
    patch.set_edgecolor("white")

ax.set_ylabel("Intra-batch σ std dev")
ax.set_title(f"⑥ Per-batch σ Diversity (batch_size={BATCH_SIZE})",
             fontsize=13, fontweight="bold")
ax.annotate("Ours: one σ per batch → zero diversity",
            xy=(1, 0.005), fontsize=8, color=colors[0], ha="center")
ax.grid(True, axis="y")

# ── Panel 7: Resolution → mu mapping ─────────────────────
ax = axes[3, 0]
res_range = np.array([256, 384, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048])
seq_lens = ((res_range // 8) // 2) ** 2
mus = np.array([calculate_shift(sl) for sl in seq_lens])

ax.plot(res_range, mus, color=SEED_COLORS[1], lw=3, marker="o", markersize=6,
        label="Seedream/SD3: calculate_shift(seq_len)")
ax.axhline(MU_OURS, color=colors[0], lw=2, ls="--",
           label=f"DreamFaceOmini: fixed μ={MU_OURS}")
ax.axhline(MU_LONGCAT, color=colors[1], lw=2, ls=":",
           label=f"LongCat: fixed μ={MU_LONGCAT} (per-resolution)")
ax.axhline(MU_FIRERED, color=colors[2], lw=2, ls="-.",
           label=f"FireRed: dynamic μ=0.5~1.15 (per-batch)")

for r, m in zip(res_range[::2], mus[::2]):
    ax.annotate(f"μ={m:.2f}", (r, m), textcoords="offset points",
                xytext=(0, 12), ha="center", fontsize=8, color=SEED_COLORS[0])

ax.set_xlabel("Image Resolution (px)")
ax.set_ylabel("Shift μ")
ax.set_title("⑦ Resolution → Shift μ Mapping", fontsize=13, fontweight="bold")
ax.legend(fontsize=8, framealpha=0.8, loc="upper left")
ax.set_xlim(200, 2100)
ax.grid(True)

# ── Panel 8: Summary comparison table ────────────────────
ax = axes[3, 1]
ax.axis("off")

table_data = [
    ["σ sampling",  "randint → table", "sigmoid(N(0,1))", "sigmoid(μ+σ·N(0,1))", "sigmoid(N(0,1))"],
    ["σ per step",  "1 (whole batch)", "B (per sample)",  "B (per sample)",       "B (per sample)"],
    ["shift μ",     "fixed 0.8",       "fixed 0.8",       "dynamic 0.5~1.15",    "dynamic, res-aware"],
    ["loss weight", "Gaussian bell",   "None (=1)",       "SD3 (1/σ)",            "None (=1)†"],
    ["robust MSE",  "Yes (thr=50)",    "No",              "Yes (thr=50)",         "Unknown"],
    ["accel.",      "—",               "—",               "—",                    "Importance samp.‡"],
]
col_labels = ["", "DreamFaceOmini", "LongCat", "FireRed", "Seedream"]

tbl = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    colWidths=[0.12, 0.22, 0.20, 0.23, 0.23],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.0, 1.8)

all_colors = colors + [SEED_COLORS[1]]
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("#444")
    if row == 0:
        cell.set_facecolor("#2a2a5a")
        cell.set_text_props(color="white", fontweight="bold")
    elif col == 0:
        cell.set_facecolor("#1e2a45")
        cell.set_text_props(color="#aaa", fontweight="bold")
    else:
        cell.set_facecolor("#16213e")
        cell.set_text_props(color="#ddd")
    if row > 0 and 1 <= col <= 4:
        cell.set_text_props(color=all_colors[col - 1])

ax.set_title("⑧ Code-level Comparison", fontsize=13, fontweight="bold")
ax.text(0.5, -0.02,
        "† Seedream 3.0 paper does not detail loss weighting; shown as uniform\n"
        "‡ Seedream acceleration: learned importance-aware timestep sampling (SSD + neural net)",
        transform=ax.transAxes, fontsize=7, ha="center", va="top", color="#888")

fig.suptitle(
    "Timestep Sampling Strategy — Precise Code-level Comparison\n"
    "DreamFaceOmini  vs  LongCat  vs  FireRed  vs  Seedream",
    fontsize=16, fontweight="bold", y=0.995,
)
plt.tight_layout(rect=[0, 0, 1, 0.96])

out_path = "viz_sampling_compare.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")
