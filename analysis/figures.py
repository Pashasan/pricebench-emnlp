"""
Generate all PriceBench figures from the data CSVs in outputs/data/.

Figures (outputs/figures/):
  fig_triage.png         -- first-option rates per model, engaged vs locked
  fig_decile_curves.png  -- non-parametric price response, engaged only, shared y
  fig_quality_heatmap.png -- standardized quality coefs, engaged
  fig_personality.png    -- scatter: price sensitivity vs quality sensitivity
  fig_brand_heatmap.png  -- model x brand coefficients, engaged
  fig_funform.png        -- log vs linear overlay on non-parametric, shared y
                            (only models with genuine negative price response)
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FixedLocator, FuncFormatter

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA = os.path.join(BASE, 'outputs', 'data')
FIG  = os.path.join(BASE, 'outputs', 'figures')
os.makedirs(FIG, exist_ok=True)

mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

# =============================================================================
# Color scheme
# -----------------------------------------------------------------------------
# A single coherent palette used across all figures. Choices:
#  - Provider colors: 8 visually distinct, harmonious hues with good print
#    legibility and reasonable colorblind separation (the three blues differ
#    in saturation/lightness so a CVD reader can still tell them apart).
#  - Compare pairs: blue + warm rust for binary/ternary, gold + violet for
#    linear/log. These are intentionally different pairs so the two comparisons
#    do not visually echo each other across plots.
#  - Heatmaps: matplotlib RdBu_r (diverging, centered on zero).
#  - Neutral grays: GRID for light reference lines, ANNOT for axis text,
#    TEXT for annotations; no other shades.
# =============================================================================

PROVIDER_COLORS = {
    'OpenAI':    '#2A9D8F',  # teal
    'Anthropic': '#C26A4A',  # terracotta
    'Google':    '#3D6FB5',  # navy blue
    'Meta':      '#5DA5DA',  # sky blue
    'Alibaba':   '#E9A93C',  # gold
    'Microsoft': '#8576C2',  # violet
    'Mistral':   '#C73E1D',  # brick red
    'DeepSeek':  '#4C5B70',  # slate
    'Other':     '#8A8A8A',  # neutral gray
}

# Two-condition comparison pairs (deliberately distinct from each other).
COMPARE_PRIMARY   = '#2E5A88'  # deep blue
COMPARE_SECONDARY = '#C26A4A'  # terracotta

LIN_COLOR = '#B8862C'  # mustard gold (linear-price fit)
LOG_COLOR = '#5D478B'  # dark violet (log-price fit)

# Neutral grayscale (use these and nothing else).
GRID_LIGHT = '#d0d0d0'  # very light gridlines
GRID       = '#b0b0b0'  # light gridlines
ANNOT      = '#666666'  # secondary axis text, annotations
TEXT       = '#333333'  # primary text on plots
LOCKED     = '#bbbbbb'  # bars for position-locked models

triage = pd.read_csv(os.path.join(DATA, 'triage.csv'))
personality = pd.read_csv(os.path.join(DATA, 'personality.csv'))
deciles = pd.read_csv(os.path.join(DATA, 'decile_pooled.csv'))
brands = pd.read_csv(os.path.join(DATA, 'brand_matrix.csv'), index_col=0)

engaged = triage[triage['verdict'] == 'engaged']['model'].tolist()
ENGAGED_SET = set(engaged)

# Shared axis helpers ---------------------------------------------------------
PRICE_TICKS = [100, 200, 400, 800]


def _format_price(v, _pos=None):
    if v >= 1000:
        return f'${v/1000:.0f}k'
    return f'${v:.0f}'


def _configure_log_x(ax):
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(FixedLocator(PRICE_TICKS))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_major_formatter(FuncFormatter(_format_price))


# ============================================================================
# Triage
# ============================================================================
def fig_triage():
    df = triage.dropna(subset=['first_rate']).sort_values('first_rate')
    fig, ax = plt.subplots(figsize=(11, 9))
    colors = []
    for _, r in df.iterrows():
        if r['verdict'] == 'engaged':
            colors.append(PROVIDER_COLORS.get(r['provider'], PROVIDER_COLORS['Other']))
        else:
            colors.append(LOCKED)
    y = np.arange(len(df))
    bars = ax.barh(y, df['first_rate'] * 100, color=colors,
                   edgecolor='white', height=0.75)
    ax.set_yticks(y)
    ax.set_yticklabels(df['model'], fontsize=12)
    ax.axvline(50, color=TEXT, lw=1.0, linestyle='-')
    ax.axvspan(0, 15, color='#f0f0f0', zorder=0)
    ax.axvspan(85, 100, color='#f0f0f0', zorder=0)

    # Locked-band labels placed to the OUTSIDE of the bands so they don't
    # overlap bars or risk being misread as belonging to any single row.
    ax.text(7.5, len(df) + 0.7, 'locked\n(recency)',
            ha='center', va='bottom', color=ANNOT, fontsize=11)
    ax.text(92.5, len(df) + 0.7, 'locked\n(primacy)',
            ha='center', va='bottom', color=ANNOT, fontsize=11)
    # Make room at the top of the axes for those labels
    ax.set_ylim(-0.5, len(df) + 2.4)

    for bar, (_, r) in zip(bars, df.iterrows()):
        x = bar.get_width()
        ax.text(x + 1.5 if x < 90 else x - 1.5, bar.get_y() + bar.get_height()/2,
                f'{x:.1f}%', va='center',
                ha='left' if x < 90 else 'right',
                fontsize=11, color=TEXT, fontweight='medium')

    ax.set_xlim(0, 100)
    ax.set_xlabel('First-shown selection rate (%), pooled across orderings',
                  fontsize=12)
    ax.tick_params(axis='x', labelsize=11)
    ax.text(50, -1.45, '50% = balanced', ha='center', color=ANNOT,
            fontsize=11)
    ax.set_title('Triage: which models actually engage with content?',
                 loc='left', fontweight='bold', fontsize=14, pad=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_triage.png'), dpi=170,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ============================================================================
# Non-parametric decile curves (engaged, shared y)
# ============================================================================
def fig_decile_curves():
    eng = personality[personality['verdict'] == 'engaged'].copy()
    eng['abs_b'] = eng['b_logp'].abs()
    eng = eng.sort_values('abs_b', ascending=False)
    models_to_plot = [m for m in eng['model'].tolist() if m in ENGAGED_SET]

    n = len(models_to_plot)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.4),
                              sharex=True, sharey=True)
    axes = np.array(axes).reshape(rows, cols)

    # Compute shared y limits so all panels are directly comparable
    sub_eng = deciles[deciles['model'].isin(models_to_plot)]
    y_lo = (sub_eng['coef'] - sub_eng['se']).min() - 0.3
    y_hi = (sub_eng['coef'] + sub_eng['se']).max() + 0.3

    triage_idx = triage.set_index('model')
    for i, name in enumerate(models_to_plot):
        ax = axes[i // cols, i % cols]
        sub = deciles[deciles['model'] == name].sort_values('decile')
        if sub.empty:
            ax.set_axis_off()
            continue
        x = sub['median_price'].values
        y = sub['coef'].values
        se = sub['se'].values
        prov = triage_idx.loc[name, 'provider']
        color = PROVIDER_COLORS.get(prov, PROVIDER_COLORS['Other'])
        ax.plot(x, y, '-o', color=color, lw=1.7, ms=3.8)
        ax.fill_between(x, y - se, y + se, color=color, alpha=0.18, lw=0)
        ax.axhline(0, color=GRID, lw=0.6, ls=':')
        ax.set_title(name, fontsize=12, pad=3)
        r2 = eng[eng['model'] == name]['r2_log'].values
        if len(r2):
            ax.text(0.97, 0.04, f'R²={r2[0]:.2f}', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=10, color=ANNOT)
        _configure_log_x(ax)
        ax.tick_params(labelsize=10)

    for j in range(n, rows * cols):
        axes[j // cols, j % cols].set_axis_off()

    for c in range(cols):
        axes[-1, c].set_xlabel('Price (log scale)', fontsize=11)
    for r in range(rows):
        axes[r, 0].set_ylabel('Utility vs cheapest decile', fontsize=11)

    for ax in axes.flat:
        if ax.has_data():
            ax.set_ylim(y_lo, y_hi)

    fig.suptitle('Non-parametric price response: all 23 engaged models, '
                 'shared y-axis, sorted by |β_ln(p)|',
                 fontsize=14, fontweight='bold', y=1.005)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_decile_curves.png'), dpi=170,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ============================================================================
# Quality sensitivity heatmap
# ============================================================================
def fig_quality_heatmap():
    eng = personality[personality['verdict'] == 'engaged'].copy()
    eng = eng.sort_values('b_logp', ascending=True).reset_index(drop=True)
    cols = ['b_d_review_score', 'b_d_stars', 'b_d_cancel',
            'b_d_breakfast', 'b_d_bed']
    labels = ['ΔReview', 'ΔStars', 'ΔCancel', 'ΔBreakfast', 'ΔBed']
    M = eng[cols].values.astype(float)
    M_z = (M - M.mean(0)) / (M.std(0) + 1e-9)

    fig, ax = plt.subplots(figsize=(9.0, max(5.5, 0.36 * len(eng))))
    im = ax.imshow(M_z, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12, fontweight='medium')
    ax.set_yticks(range(len(eng)))
    ax.set_yticklabels(eng['model'], fontsize=10.5)
    for i in range(len(eng)):
        for j in range(len(cols)):
            v = M[i, j]
            shade = 'white' if abs(M_z[i, j]) > 1.1 else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=10, color=shade, fontweight='medium')
    ax.set_title('Quality sensitivity per model (raw logit coefs, '
                 'cells colored by within-attribute z-score)',
                 fontsize=11.5, loc='left', fontweight='bold', pad=10)
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label('z-score across models', fontsize=10)
    cb.ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_quality_heatmap.png'), dpi=170,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ============================================================================
# Personality map
# ============================================================================
def _nudge_labels(positions, min_dx=1.10, min_dy=0.35, passes=80):
    """Iteratively push overlapping labels apart along y so each label
    has at least min_dy vertical clearance from every neighbor within
    min_dx in the x direction. Labels are anchored to the right of the
    point, so the effective collision box is wider than the point itself."""
    out = [list(p) for p in positions]
    for _ in range(passes):
        moved = False
        order = sorted(range(len(out)), key=lambda i: (out[i][1], out[i][0]))
        for a_idx in range(len(order)):
            for b_idx in range(a_idx + 1, len(order)):
                i, j = order[a_idx], order[b_idx]
                dx = abs(out[i][0] - out[j][0])
                dy = out[j][1] - out[i][1]
                if dx < min_dx and dy < min_dy:
                    shift = (min_dy - dy) / 2 + 0.005
                    out[i][1] -= shift
                    out[j][1] += shift
                    moved = True
        if not moved:
            break
    return [tuple(p) for p in out]


def fig_personality():
    from adjustText import adjust_text
    eng = personality[personality['verdict'] == 'engaged'].copy()
    eng['abs_b'] = eng['b_logp'].abs()
    eng['quality'] = eng['b_d_review_score']

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    triage_idx = triage.set_index('model')

    texts = []
    pts_x, pts_y = [], []
    for _, r in eng.iterrows():
        x, y = r['abs_b'], r['quality']
        prov = triage_idx.loc[r['model'], 'provider']
        color = PROVIDER_COLORS.get(prov, PROVIDER_COLORS['Other'])
        ax.scatter(x, y, s=110, color=color, edgecolor='white', lw=1.2,
                   alpha=0.92, zorder=3)
        texts.append(ax.text(x, y, r['model'], fontsize=11.0, color=TEXT,
                              fontweight='medium', zorder=4))
        pts_x.append(x); pts_y.append(y)

    ax.set_xlabel('Price sensitivity  |β_ln(p)|   '
                  '(higher = more price-averse, holding quality fixed)',
                  fontsize=12)
    ax.set_ylabel('Quality sensitivity  β_review_score',
                  fontsize=12)
    ax.axhline(0, color=GRID, lw=0.6, ls=':')

    xmid = float(eng['abs_b'].median())
    ymid = float(eng['quality'].median())
    ax.axvline(xmid, color=GRID_LIGHT, lw=0.5, ls='--')
    ax.axhline(ymid, color=GRID_LIGHT, lw=0.5, ls='--')

    # Headroom: top for quadrant labels, bottom for nudged labels.
    x_lo, x_hi = ax.get_xlim(); y_lo, y_hi = ax.get_ylim()
    y_range = y_hi - y_lo
    ax.set_ylim(y_lo - y_range * 0.20, y_hi + y_range * 0.10)
    ax.set_xlim(x_lo - (x_hi - x_lo) * 0.02, x_hi + (x_hi - x_lo) * 0.02)

    # Quadrant labels placed at the top as annotations above the data area
    # so they don't overlap with points or the legend.
    ax.set_title(
        'PriceBench price-quality map: all 23 engaged LLMs',
        loc='left', fontweight='bold', fontsize=14, pad=32)
    quad_kw = dict(fontsize=10, color=ANNOT, fontstyle='italic',
                   transform=ax.transAxes)
    ax.text(xmid / ax.get_xlim()[1] / 2 + 0.005, 1.02,
            'lets price slide + chases quality', ha='left', **quad_kw)
    ax.text(1.0 - (xmid / ax.get_xlim()[1]) * 0.0 - 0.005, 1.02,
            'sharp on price + chases quality', ha='right', **quad_kw)

    # Lower quadrant hints at the bottom of the data area, tucked against axes
    ax.text(0.005, 0.01, 'indifferent / weak price signal',
            ha='left', va='bottom', fontsize=10, color=GRID,
            fontstyle='italic', transform=ax.transAxes)
    ax.text(0.995, 0.01, 'sharp on price + lets quality slide',
            ha='right', va='bottom', fontsize=10, color=GRID,
            fontstyle='italic', transform=ax.transAxes)

    # Provider legend — place OUTSIDE data area (to the right) so it never
    # overlaps points.
    seen, handles = set(), []
    order_providers = ['OpenAI', 'Anthropic', 'Google', 'Meta', 'Alibaba',
                        'Microsoft', 'Mistral', 'DeepSeek']
    for prov in order_providers:
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=PROVIDER_COLORS[prov],
                       markersize=10, label=prov))
    ax.legend(handles=handles, loc='center left',
              bbox_to_anchor=(1.005, 0.5), frameon=False, fontsize=9)

    adjust_text(
        texts,
        x=pts_x, y=pts_y,
        ax=ax,
        expand=(1.25, 1.45),
        force_text=(0.6, 0.9),
        force_static=(0.4, 0.6),
        arrowprops=dict(arrowstyle='-', color=GRID_LIGHT, lw=0.6, alpha=0.8),
        only_move={'text': 'xy', 'static': 'xy', 'explode': 'xy', 'pull': 'xy'},
        max_move=60,
        time_lim=3.0,
    )

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_personality.png'), dpi=170,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ============================================================================
# Willingness-to-pay — the scale-free price/quality trade-off
# ============================================================================
def fig_wtp():
    """Scale-free persona axis. The raw personality map mixes two things:
    DECISIVENESS (how sharply a model discriminates at all -- the logit scale
    multiplies every coefficient, so decisive models look large on price AND
    quality) and the TRADE-OFF DIRECTION (price vs quality). Willingness to pay
    -- dollars per review-score point -- is a ratio of coefficients, so the
    logit scale cancels and only the trade-off survives. WTP at a reference
    listing price p0 is (beta_review / |beta_ln(p)|) * p0, identical in form
    across models. Restricted to engaged models whose price AND review
    coefficients are both well identified (p < 0.001)."""
    P0 = 250.0
    e = personality[(personality['verdict'] == 'engaged')
                    & (personality['b_logp_p'] < 0.001)
                    & (personality['b_d_review_score_p'] < 0.001)].copy()
    a, b = e['b_d_review_score'].values, e['b_logp'].abs().values
    sa, sb = e['b_d_review_score_se'].values, e['b_logp_se'].values
    e['wtp'] = (a / b) * P0
    # delta-method SE on the ratio (treats the two coefs as independent; the
    # joint covariance is not in the CSV, so these bars are approximate).
    e['wtp_se'] = np.sqrt((sa / b) ** 2 + (a * sb / b ** 2) ** 2) * P0
    e = e.sort_values('wtp').reset_index(drop=True)

    triage_idx = triage.set_index('model')
    colors = [PROVIDER_COLORS.get(triage_idx.loc[m, 'provider'],
                                  PROVIDER_COLORS['Other']) for m in e['model']]

    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(e))
    ax.barh(y, e['wtp'], xerr=e['wtp_se'], color=colors, edgecolor='white',
            height=0.74, error_kw=dict(ecolor=ANNOT, lw=0.9, capsize=2), zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(e['model'], fontsize=11)

    med = float(e['wtp'].median())
    ax.axvline(med, color=GRID, lw=0.8, ls='--', zorder=1)
    ax.text(med, len(e) - 0.35, f' median ${med:.0f}', color=ANNOT,
            fontsize=9, va='top', ha='left')

    for yi, (w, s) in enumerate(zip(e['wtp'], e['wtp_se'])):
        ax.text(w + s + 10, yi, f'${w:.0f}', va='center', ha='left',
                fontsize=9.5, color=TEXT)

    ax.set_xlabel('Willingness to pay for one review-score point '
                  '(\$ at a \$250 listing)', fontsize=11)
    ax.set_xlim(0, float((e['wtp'] + e['wtp_se']).max()) * 1.14)
    ax.set_ylim(-0.7, len(e) - 0.3)

    # Right-margin direction label (outside the data area, no bar overlap).
    # rotation=90 so it reads bottom->top, matching the ascending sort:
    # bargain hunters at the bottom (low WTP), quality chasers at the top.
    ax.text(1.035, 0.5, 'bargain hunters  →  quality chasers',
            transform=ax.transAxes, rotation=90, va='center', ha='center',
            fontsize=10.5, color=ANNOT, fontstyle='italic')

    ax.set_title('Scale-free price/quality trade-off: '
                 'willingness to pay per quality point',
                 loc='left', fontweight='bold', fontsize=13, pad=12)

    order_providers = ['OpenAI', 'Anthropic', 'Google', 'Meta', 'Alibaba',
                       'Microsoft', 'Mistral', 'DeepSeek']
    present = {triage_idx.loc[m, 'provider'] for m in e['model']}
    handles = [plt.Line2D([0], [0], marker='s', color='w',
               markerfacecolor=PROVIDER_COLORS[p], markersize=10, label=p)
               for p in order_providers if p in present]
    ax.legend(handles=handles, loc='lower right', frameon=False,
              fontsize=9, ncol=2)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_wtp.png'), dpi=170,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'wtp figure: n={len(e)}, '
          f'range ${e["wtp"].min():.0f}-${e["wtp"].max():.0f} '
          f'({e["wtp"].max()/e["wtp"].min():.0f}x)')


# ============================================================================
# Brand heatmap
# ============================================================================
def fig_brand_heatmap():
    M = brands.copy()
    order = M.abs().sum(axis=1).sort_values(ascending=True).index
    M = M.loc[order]
    fig, ax = plt.subplots(figsize=(8.0, max(5.5, 0.36 * len(M))))
    vmax = max(abs(M.values.min()), abs(M.values.max()))
    im = ax.imshow(M.values, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(M.columns, fontsize=12, fontweight='medium')
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels(M.index, fontsize=10.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            color = 'white' if abs(v) > 0.55 * vmax else 'black'
            ax.text(j, i, f'{v:+.2f}', ha='center', va='center',
                    fontsize=10, color=color, fontweight='medium')
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label('logit coef (vs Independent)', fontsize=10)
    cb.ax.tick_params(labelsize=9)
    ax.set_title('Brand preferences after controlling for price + quality + room attributes\n'
                 '(positive = preferred over an Independent at same controls)',
                 fontsize=11.5, loc='left', fontweight='bold', pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_brand_heatmap.png'), dpi=170,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ============================================================================
# Log vs linear vs nonparametric
# ============================================================================
def fig_funform():
    """
    Filter to models with a genuine negative price response -- i.e. the
    log-price coefficient is negative and statistically significant at p<0.05
    AND the non-parametric D10 coefficient is meaningfully below zero (< -0.3).
    This excludes models like Gemma3 1B (positive coef) and Phi-2 2.7B (flat)
    where the log-vs-linear question is not well posed.
    """
    d_last = deciles.sort_values('decile').groupby('model').tail(1)[
        ['model', 'coef']
    ].rename(columns={'coef': 'd10_coef'})
    merged = personality.merge(d_last, on='model', how='left')

    eng = merged[(merged['verdict'] == 'engaged')
                  & (merged['b_logp'] < 0)
                  & (merged['b_logp_p'] < 0.05)
                  & (merged['d10_coef'] < -0.3)].copy()
    eng['abs_b'] = eng['b_logp'].abs()
    eng = eng.sort_values('abs_b', ascending=False)
    models_to_plot = eng['model'].tolist()

    # Qwen3 30B-A3B (the weakest negative responder, |b_logp| = 0.40) is omitted
    # from this main-text figure for space; its result is stated in the caption.
    # Dropping it leaves exactly 20 panels (4 full rows of 5) rather than a lone
    # 21st panel rolling onto a 5th row.
    models_to_plot = [m for m in models_to_plot if m != 'Qwen3 30B-A3B']

    n = len(models_to_plot)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.05),
                              sharex=True, sharey=True)
    axes = np.array(axes).reshape(rows, cols)

    # Shared y range over these models
    sub = deciles[deciles['model'].isin(models_to_plot)]
    y_lo = (sub['coef'] - sub['se']).min() - 0.5
    y_hi = (sub['coef'] + sub['se']).max() + 0.5

    for i, name in enumerate(models_to_plot):
        ax = axes[i // cols, i % cols]
        ss = deciles[deciles['model'] == name].sort_values('decile')
        x = ss['median_price'].values
        y = ss['coef'].values
        se = ss['se'].values
        ax.fill_between(x, y - se, y + se, color=ANNOT, alpha=0.15, lw=0)
        ax.plot(x, y, 'o', ms=4.5, color=TEXT)

        row = eng[eng['model'] == name].iloc[0]
        x_ref = x[0]
        log_pred = row['b_logp'] * (np.log(x) - np.log(x_ref))
        lin_pred = row['b_linp'] * (x - x_ref)
        best = row['fit_form']

        # Winning curve bold; losing curve pale.
        LIN_STYLE = dict(color=LIN_COLOR, ls='--')
        LOG_STYLE = dict(color=LOG_COLOR)
        if best == 'linear':
            ax.plot(x, lin_pred, lw=2.0, **LIN_STYLE)
            ax.plot(x, log_pred, lw=1.0, alpha=0.35, **LOG_STYLE)
        elif best == 'log':
            ax.plot(x, log_pred, lw=2.0, **LOG_STYLE)
            ax.plot(x, lin_pred, lw=1.0, alpha=0.35, **LIN_STYLE)
        else:
            ax.plot(x, log_pred, lw=1.5, **LOG_STYLE)
            ax.plot(x, lin_pred, lw=1.5, **LIN_STYLE)

        d_aic = row['aic_lin'] - row['aic_log']  # negative = linear wins
        marker = best if best != 'tie' else 'tie'
        ax.set_title(f'{name}  [best: {marker}]', fontsize=9.5, pad=3)
        # 0 displayed unsigned so near-tie panels never print "-0".
        d_aic_txt = '0' if round(d_aic) == 0 else f'{d_aic:+.0f}'
        # Coefficients + AIC gap stacked in the bottom-left: the response falls
        # toward the bottom-right, so that corner is clear of the curve.
        ax.text(0.04, 0.05,
                f'$\\beta_{{\\ln p}}={row["b_logp"]:.2f}$\n'
                f'$\\beta_{{p}}={row["b_linp"]:.4f}$\n'
                f'$\Delta$AIC$={d_aic_txt}$',
                transform=ax.transAxes, ha='left', va='bottom',
                fontsize=10.0, color=ANNOT, linespacing=1.4)
        ax.axhline(0, color=GRID_LIGHT, lw=0.5, ls=':')
        _configure_log_x(ax)
        ax.tick_params(labelsize=10)

    for j in range(n, rows * cols):
        axes[j // cols, j % cols].set_axis_off()
    for c in range(cols):
        axes[-1, c].set_xlabel('Price (log scale)', fontsize=11)
    for r in range(rows):
        axes[r, 0].set_ylabel('Utility vs cheapest decile', fontsize=11)

    for ax in axes.flat:
        if ax.has_data():
            ax.set_ylim(y_lo, y_hi)

    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=TEXT,
                   markersize=6, label='non-parametric (decile dummies)'),
        plt.Line2D([0], [0], color=LOG_COLOR, lw=2,
                   label='log-price fit'),
        plt.Line2D([0], [0], color=LIN_COLOR, ls='--', lw=2,
                   label='linear-price fit'),
    ]
    fig.legend(handles=handles, loc='upper center',
               bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, fontsize=11)
    fig.suptitle('Log vs linear price fit: winner bold, loser pale. '
                 'Shared y-axis.',
                 fontsize=14, fontweight='bold', y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_funform.png'), dpi=170,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Print the filter result for reporting
    fit_counts = eng['fit_form'].value_counts().to_dict()
    print(f'funform figure: {n} models passed filter; fit counts = {fit_counts}')


# ============================================================================
# Binary vs ternary forest plot (log + linear price)
# ============================================================================
def fig_binary_vs_ternary():
    """Two-panel forest plot: log-price beta (left) and linear-price beta
    (right), each with binary and ternary estimates per engaged model.
    Models are sorted on a SHARED y-axis by binary log-price coefficient
    from most-negative (top) to least."""
    bin_df = pd.read_csv(os.path.join(DATA, 'personality.csv'))
    ter_df = pd.read_csv(os.path.join(DATA, 'personality_ternary.csv'))
    bin_ = bin_df[bin_df['verdict'] == 'engaged'][
        ['model', 'b_logp', 'b_logp_se', 'b_linp', 'b_linp_se']
    ].rename(columns={'b_logp': 'b_logp_bin', 'b_logp_se': 'se_logp_bin',
                       'b_linp': 'b_linp_bin', 'b_linp_se': 'se_linp_bin'})
    ter = ter_df[['model', 'b_logp', 'b_logp_se', 'b_linp', 'b_linp_se']].rename(
        columns={'b_logp': 'b_logp_ter', 'b_logp_se': 'se_logp_ter',
                 'b_linp': 'b_linp_ter', 'b_linp_se': 'se_linp_ter'})
    m = bin_.merge(ter, on='model', how='left')
    m = m.sort_values('b_logp_bin', ascending=True).reset_index(drop=True)
    # most-negative first row; invert y-axis later so it ends up on top

    from scipy.stats import spearmanr, pearsonr
    valid = m['b_logp_ter'].notna()
    rho_log, _ = spearmanr(m.loc[valid, 'b_logp_bin'].abs(),
                            m.loc[valid, 'b_logp_ter'].abs())
    r_log, _ = pearsonr(m.loc[valid, 'b_logp_bin'],
                         m.loc[valid, 'b_logp_ter'])
    rho_lin, _ = spearmanr(m.loc[valid, 'b_linp_bin'].abs(),
                            m.loc[valid, 'b_linp_ter'].abs())
    r_lin, _ = pearsonr(m.loc[valid, 'b_linp_bin'],
                         m.loc[valid, 'b_linp_ter'])

    fig, (ax_log, ax_lin) = plt.subplots(
        1, 2, figsize=(12.5, 8), sharey=True,
        gridspec_kw={'wspace': 0.08})

    y = np.arange(len(m))

    # --- Log panel ---
    for i, row in m.iterrows():
        if not pd.isna(row['b_logp_ter']):
            ax_log.plot([row['b_logp_bin'], row['b_logp_ter']],
                        [y[i], y[i]], '-', color=GRID_LIGHT, lw=0.8, zorder=1)
    ax_log.errorbar(m['b_logp_bin'], y, xerr=m['se_logp_bin'],
                     fmt='o', color=COMPARE_PRIMARY, ms=6, capsize=2,
                     label='binary', zorder=3)
    ax_log.errorbar(m.loc[valid, 'b_logp_ter'], y[valid],
                     xerr=m.loc[valid, 'se_logp_ter'],
                     fmt='s', color=COMPARE_SECONDARY, ms=5.5, capsize=2,
                     label='ternary', zorder=3)
    ax_log.axvline(0, color=GRID, lw=0.6, ls=':')
    ax_log.set_yticks(y)
    ax_log.set_yticklabels(m['model'], fontsize=9)
    ax_log.invert_yaxis()
    ax_log.set_xlabel(r'$\beta_{\ln(p)}$  log-price coefficient',
                       fontsize=10.5)
    ax_log.set_title(
        f'Log-price   (Spearman ρ = {rho_log:.2f}, Pearson r = {r_log:.2f})',
        fontsize=10.5, loc='left', pad=8)
    ax_log.legend(loc='lower left', frameon=False, fontsize=9)

    # --- Linear panel ---
    for i, row in m.iterrows():
        if not pd.isna(row['b_linp_ter']):
            ax_lin.plot([row['b_linp_bin'], row['b_linp_ter']],
                        [y[i], y[i]], '-', color=GRID_LIGHT, lw=0.8, zorder=1)
    ax_lin.errorbar(m['b_linp_bin'], y, xerr=m['se_linp_bin'],
                     fmt='o', color=COMPARE_PRIMARY, ms=6, capsize=2,
                     zorder=3)
    ax_lin.errorbar(m.loc[valid, 'b_linp_ter'], y[valid],
                     xerr=m.loc[valid, 'se_linp_ter'],
                     fmt='s', color=COMPARE_SECONDARY, ms=5.5, capsize=2,
                     zorder=3)
    ax_lin.axvline(0, color=GRID, lw=0.6, ls=':')
    ax_lin.set_xlabel(r'$\beta_{p}$  linear-price coefficient (per-dollar)',
                       fontsize=10.5)
    ax_lin.set_title(
        f'Linear price   (Spearman ρ = {rho_lin:.2f}, Pearson r = {r_lin:.2f})',
        fontsize=10.5, loc='left', pad=8)

    fig.suptitle(
        'Binary vs ternary price coefficients per model '
        '(models sorted by binary log-price, shared y-axis)',
        fontsize=12, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_binary_vs_ternary.png'), dpi=170,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ============================================================================
# Parameter count (capability) vs |beta_ln(p)|
# ============================================================================
# Open-weight parameter counts (billions, best public estimate).
# For MoE models we use ACTIVE parameters since that drives compute.
# Proprietary model counts are not public; we use rough tier-based estimates
# that approximately match API pricing ladders, and flag them as estimates.
MODEL_PARAMS_B = {
    # Open-weight (known)
    'Qwen3 0.6B': 0.6, 'Qwen3 1.7B': 1.7, 'Qwen3 4B': 4.0, 'Qwen3 8B': 8.0,
    'Qwen3 30B-A3B': 3.0,  # MoE, 3B active
    'Gemma3 1B': 1.0, 'Gemma3 4B': 4.0, 'Gemma3 12B': 12.0, 'Gemma3 27B': 27.0,
    'Gemma2 9B': 9.0,
    'Llama3.2 1B': 1.0, 'Llama3.2 3B': 3.0, 'Llama3 8B': 8.0,
    'Llama3.1 8B': 8.0,
    'Phi-2 2.7B': 2.7, 'Phi-3 Mini': 3.8, 'Phi-3 Medium': 14.0,
    'Phi-4 Mini': 3.8, 'Phi-4 14B': 14.0,
    'Mistral 7B': 7.0, 'Mistral-Nemo 12B': 12.0,
    'DeepSeek-R1 1.5B': 1.5, 'DeepSeek-R1 7B': 7.0,
    # Proprietary (ESTIMATES from pricing tiers)
    'GPT-4.1 Nano': 8.0,   'GPT-4.1 Mini': 40.0,
    'GPT-5.4 Nano': 10.0,  'GPT-5.4 Mini': 60.0,  'GPT-5.4': 400.0,
    'Claude Haiku 4.5': 25.0,
}
PROPRIETARY = {'GPT-4.1 Nano', 'GPT-4.1 Mini', 'GPT-5.4 Nano',
               'GPT-5.4 Mini', 'GPT-5.4', 'Claude Haiku 4.5'}


def fig_scaling():
    """Scatter: model size vs binary log-price sensitivity among engaged models.
    Filled circles = open-weight (real size), open squares = proprietary
    (tier-based size estimate). Within-family connecting lines."""
    p = pd.read_csv(os.path.join(DATA, 'personality.csv'))
    e = p[(p['verdict'] == 'engaged') & (p['b_logp_p'] < 0.05)].copy()
    e['params_b'] = e['model'].map(MODEL_PARAMS_B)
    e = e.dropna(subset=['params_b']).copy()
    e['absb'] = e['b_logp'].abs()
    e['is_prop'] = e['model'].isin(PROPRIETARY)

    from scipy.stats import pearsonr, spearmanr
    open_e = e[~e['is_prop']]
    r_open, p_open = pearsonr(np.log10(open_e['params_b']), open_e['absb'])
    rho_open, _ = spearmanr(np.log10(open_e['params_b']), open_e['absb'])
    r_all, _ = pearsonr(np.log10(e['params_b']), e['absb'])

    from adjustText import adjust_text
    fig, ax = plt.subplots(figsize=(14, 9))
    triage_idx = triage.set_index('model')

    # Within-family connecting lines
    families = {
        'Qwen3': ['Qwen3 4B', 'Qwen3 8B', 'Qwen3 30B-A3B'],
        'Gemma3': ['Gemma3 1B', 'Gemma3 4B', 'Gemma3 12B', 'Gemma3 27B'],
        'Phi-3': ['Phi-3 Mini', 'Phi-3 Medium'],
        'Phi-4': ['Phi-4 Mini', 'Phi-4 14B'],
        'DeepSeek-R1': ['DeepSeek-R1 1.5B', 'DeepSeek-R1 7B'],
        'GPT-4.1': ['GPT-4.1 Nano', 'GPT-4.1 Mini'],
        'GPT-5.4': ['GPT-5.4 Nano', 'GPT-5.4 Mini', 'GPT-5.4'],
    }
    for fam, members in families.items():
        sub = e[e['model'].isin(members)].sort_values('params_b')
        if len(sub) < 2:
            continue
        prov = triage_idx.loc[sub.iloc[0]['model'], 'provider']
        color = PROVIDER_COLORS.get(prov, PROVIDER_COLORS['Other'])
        ax.plot(sub['params_b'], sub['absb'], '-',
                color=color, lw=1.2, alpha=0.25, zorder=1)

    # Points
    texts = []
    pts_x, pts_y = [], []
    for _, r in e.iterrows():
        prov = triage_idx.loc[r['model'], 'provider']
        color = PROVIDER_COLORS.get(prov, PROVIDER_COLORS['Other'])
        if r['is_prop']:
            ax.scatter(r['params_b'], r['absb'], s=140, marker='s',
                       facecolor='white', edgecolor=color, lw=2.0, zorder=3)
        else:
            ax.scatter(r['params_b'], r['absb'], s=160, marker='o',
                       color=color, edgecolor='white', lw=1.2, zorder=3)
        texts.append(ax.text(r['params_b'], r['absb'], r['model'],
                              fontsize=11.5, color=TEXT, fontweight='medium',
                              zorder=4))
        pts_x.append(r['params_b']); pts_y.append(r['absb'])

    ax.set_xscale('log')
    ax.set_xlabel('Parameter count (B, log scale).   '
                  'Squares = proprietary tier-based estimates.',
                  fontsize=12)
    ax.set_ylabel(r'|$\beta_{\ln(p)}$|   (binary log-price sensitivity)',
                   fontsize=12)
    ax.tick_params(axis='both', labelsize=11)
    ax.set_title(
        f'Capability (parameter count) vs price sensitivity   '
        f'(open-weight only: r = {r_open:.2f}, ρ = {rho_open:.2f}, '
        f'n = {len(open_e)})',
        loc='left', fontweight='bold', fontsize=14, pad=14)

    # Provider legend
    seen, handles = set(), []
    for _, r in e.iterrows():
        prov = triage_idx.loc[r['model'], 'provider']
        if prov in seen:
            continue
        seen.add(prov)
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=PROVIDER_COLORS[prov],
                       markersize=12, label=prov))
    ax.legend(handles=handles, loc='upper left', frameon=False,
              fontsize=11, ncol=2)

    # Headroom for label placement at top and bottom edges (log y space).
    y_lo, y_hi = ax.get_ylim()
    ax.set_ylim(y_lo - 0.4, y_hi + 0.6)

    adjust_text(
        texts,
        x=pts_x, y=pts_y,
        ax=ax,
        expand=(1.3, 1.5),
        force_text=(0.7, 1.0),
        force_static=(0.5, 0.7),
        arrowprops=dict(arrowstyle='-', color=GRID_LIGHT, lw=0.6, alpha=0.8),
        only_move={'text': 'xy', 'static': 'xy', 'explode': 'xy', 'pull': 'xy'},
        max_move=80,
        time_lim=4.0,
    )

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_scaling.png'), dpi=170,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f'  open-weight n={len(open_e)}  Pearson r={r_open:.3f} '
          f'(p={p_open:.3g})  Spearman rho={rho_open:.3f}')
    print(f'  all-engaged n={len(e)}  Pearson r={r_all:.3f}')


if __name__ == '__main__':
    print('triage...');              fig_triage()
    print('decile curves...');       fig_decile_curves()
    print('quality heatmap...');     fig_quality_heatmap()
    print('personality...');         fig_personality()
    print('wtp (trade-off)...');     fig_wtp()
    print('brand heatmap...');       fig_brand_heatmap()
    print('functional form...');     fig_funform()
    print('binary vs ternary...');   fig_binary_vs_ternary()
    print('scaling...');             fig_scaling()
    print('done, figures in', FIG)
