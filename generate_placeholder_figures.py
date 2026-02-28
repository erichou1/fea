"""
Generate all placeholder figures for the SASTO research paper.
Creates publication-quality figures using matplotlib for:
  1. fig1_pipeline.png - SASTO pipeline overview flowchart
  2. fig2_architecture.png - Surrogate3DResNet architecture diagram
  3. fig_wireframe_pipeline.png - 3DWire wireframe -> volumetric -> voxelized
  4. fig_model_comparison.png - Wireframe -> Volumetric -> SASTO-U -> SASTO-PA
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Common style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# ============================================================
# Helper: render voxel iso-surface with part colors
# ============================================================
def render_voxels_3d(ax, occ, part_labels=None, alpha=0.95, elev=25, azim=-60,
                     title=None, downsample=2):
    """Render 3D voxel volume with part-type coloring using scatter plot."""
    # Downsample for speed
    occ_ds = occ[::downsample, ::downsample, ::downsample]

    if part_labels is not None:
        part_ds = part_labels[::downsample, ::downsample, ::downsample]
    else:
        part_ds = np.ones_like(occ_ds)

    # Color map: 0=empty, 1=exterior(blue), 2=interior(green), 3=roof(orange), 4=floor(red)
    # Strong, saturated, non-pastel colors
    part_colors = {
        0: (0.50, 0.50, 0.50, alpha),    # generic gray
        1: (0.08, 0.30, 0.70, alpha),    # exterior wall - strong blue
        2: (0.15, 0.60, 0.15, alpha),    # interior wall - strong green
        3: (0.85, 0.45, 0.00, alpha),    # roof - strong orange
        4: (0.75, 0.10, 0.10, alpha),    # floor - strong red
    }

    occupied = np.where(occ_ds > 0)
    if len(occupied[0]) == 0:
        return

    xs, ys, zs = occupied
    colors = []
    for i in range(len(xs)):
        p = part_ds[xs[i], ys[i], zs[i]]
        colors.append(part_colors.get(p, part_colors[0]))

    ax.scatter(xs, ys, zs, c=colors, s=3.0, marker='s', linewidths=0, depthshade=True)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(0, occ_ds.shape[0])
    ax.set_ylim(0, occ_ds.shape[1])
    ax.set_zlim(0, occ_ds.shape[2])
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold', pad=-5)


def render_wireframe_3d(ax, vertices, lines, title=None, elev=25, azim=-60):
    """Render 3D wireframe from vertices and edge indices."""
    for edge in lines:
        v0, v1 = vertices[edge[0]], vertices[edge[1]]
        ax.plot3D(*zip(v0, v1), color='#333333', linewidth=0.8)

    # Plot vertices as dots
    ax.scatter3D(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                 color='#CC3333', s=8, zorder=5, depthshade=False)

    ax.view_init(elev=elev, azim=azim)
    margin = 0.1
    for i, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
        lo, hi = vertices[:, i].min(), vertices[:, i].max()
        setter(lo - margin, hi + margin)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold', pad=-5)


# ============================================================
# Figure 1: SASTO Pipeline Overview
# ============================================================
def generate_pipeline_figure():
    print("Generating fig1_pipeline.png ...")
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.5)
    ax.set_axis_off()

    # --- Color palette (saturated, non-pastel) ---
    BLUE_BG     = '#1B5E9E'   # deep blue for offline
    BLUE_DARK   = '#0D3B66'   # dark blue for offline header
    ORANGE_BG   = '#BF5700'   # burnt orange for online
    ORANGE_DARK = '#8C3D00'   # dark orange for online header
    GREEN_BG    = '#1E7B3A'   # deep green for output
    GRAY_TEXT   = '#333333'

    def draw_box(ax, x, y, w, h, text, facecolor, edgecolor, fontsize=8.5,
                 fontcolor='white', fontweight='normal'):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                             facecolor=facecolor, edgecolor=edgecolor,
                             linewidth=1.8, zorder=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, color=fontcolor, fontweight=fontweight, zorder=3)

    def draw_arrow(ax, x1, y1, x2, y2, color='#333333', lw=1.8, style='-'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                    linestyle=style), zorder=1)

    # ---- Phase headers ----
    ax.text(0.15, 5.15, 'OFFLINE TRAINING PHASE', fontsize=12, fontweight='bold',
            color=BLUE_DARK, va='center', fontfamily='sans-serif')
    ax.axhline(y=4.95, xmin=0.01, xmax=0.99, color=BLUE_DARK, linewidth=1.0, alpha=0.4)

    ax.text(0.15, 2.7, 'ONLINE OPTIMIZATION PHASE', fontsize=12, fontweight='bold',
            color=ORANGE_DARK, va='center', fontfamily='sans-serif')
    ax.axhline(y=2.5, xmin=0.01, xmax=0.99, color=ORANGE_DARK, linewidth=1.0, alpha=0.4)

    # ---- OFFLINE PHASE (top row) ----
    row_y = 3.7
    row_h = 1.0
    boxes_top = [
        (0.2,  row_y, 2.6, row_h, '3DWire\nWireframes\n(14,293)', BLUE_BG),
        (3.5,  row_y, 2.6, row_h, 'Volumetric\nGeneration\n(4-part STL)', BLUE_BG),
        (6.8,  row_y, 2.6, row_h, 'FEA Simulation\n(SfePy, ASCE 7-22)\nσ, u, C', BLUE_BG),
        (10.2, row_y, 3.5, row_h, 'Deep Ensemble Training\n(5× Surrogate3DResNet)\n11,178 samples', '#0A2F5C'),
    ]
    for x, y, w, h, text, fc in boxes_top:
        draw_box(ax, x, y, w, h, text, fc, '#0A2F5C', fontsize=8.5, fontcolor='white')

    # Arrows between top boxes (straight, no overlap)
    draw_arrow(ax, 2.8, row_y+row_h/2, 3.5, row_y+row_h/2, color=BLUE_DARK)
    draw_arrow(ax, 6.1, row_y+row_h/2, 6.8, row_y+row_h/2, color=BLUE_DARK)
    draw_arrow(ax, 9.4, row_y+row_h/2, 10.2, row_y+row_h/2, color=BLUE_DARK)

    # ---- ONLINE PHASE (bottom row) ----
    bot_y = 1.0
    bot_h = 1.2
    boxes_bot = [
        (0.2,  bot_y, 2.0, bot_h, '128³ Voxel Grid\n+ Part Labels', '#D46A00'),
        (2.8,  bot_y, 2.4, bot_h, 'Phase 1:\nSensitivity Erosion\n(99% removal)', ORANGE_BG),
        (5.8,  bot_y, 2.0, bot_h, 'Phase 2:\nEndgame\n(single voxel)', ORANGE_BG),
        (8.4,  bot_y, 2.0, bot_h, 'Phase 3:\nSwap Refinement', ORANGE_BG),
        (11.1, bot_y, 2.7, bot_h, 'Post-Process\n+ STL Export\n(marching cubes)', GREEN_BG),
    ]
    for x, y, w, h, text, fc in boxes_bot:
        ec = '#5C2D00' if fc != GREEN_BG else '#0A4D1E'
        draw_box(ax, x, y, w, h, text, fc, ec, fontsize=8.5, fontcolor='white')

    # Arrows between bottom boxes
    draw_arrow(ax, 2.2, bot_y+bot_h/2, 2.8, bot_y+bot_h/2, color=ORANGE_DARK)
    draw_arrow(ax, 5.2, bot_y+bot_h/2, 5.8, bot_y+bot_h/2, color=ORANGE_DARK)
    draw_arrow(ax, 7.8, bot_y+bot_h/2, 8.4, bot_y+bot_h/2, color=ORANGE_DARK)
    draw_arrow(ax, 10.4, bot_y+bot_h/2, 11.1, bot_y+bot_h/2, color='#333333')

    # Connection: ensemble -> Phase 1 (dashed, surrogate predictions)
    ax.annotate('', xy=(4.0, bot_y+bot_h), xytext=(11.95, row_y),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5,
                                linestyle='dashed', connectionstyle='arc3,rad=0.25'),
                zorder=1)
    ax.text(8.7, 3.1, 'Surrogate predictions', fontsize=8, ha='center',
            va='center', color='#333333', style='italic',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='#999999', linewidth=0.7, alpha=0.9))

    # Key features (below bottom row)
    features = [
        (4.0, 0.35, '6-connectivity check'),
        (7.0, 0.35, 'Conservative μ+kσ bounds'),
        (9.8, 0.35, 'Adaptive batch (trust region)'),
    ]
    for x, y, text in features:
        ax.text(x, y, text, ha='center', va='center', fontsize=7.5,
                color='#444444', style='italic',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#F0F0F0',
                          edgecolor='#888888', linewidth=0.6))

    plt.tight_layout(pad=0.5)
    fig.savefig(os.path.join(FIGURES_DIR, 'fig1_pipeline.png'))
    fig.savefig(os.path.join(FIGURES_DIR, 'fig1_pipeline.pdf'))
    plt.close(fig)
    print("  -> Saved fig1_pipeline.png/pdf")


# ============================================================
# Figure 2: Surrogate3DResNet Architecture
# ============================================================
def generate_architecture_figure():
    print("Generating fig2_architecture.png ...")
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4.5)
    ax.set_axis_off()

    # --- Color palette (saturated, non-pastel) ---
    CONV_BLUE   = '#1B4F8A'   # deep blue for conv stages
    CONV_DARK   = '#0D2E52'   # very dark blue border
    SE_AMBER    = '#B8600A'   # deep amber for SE blocks
    POOL_GREEN  = '#1A6B35'   # dark green for pooling
    FEAT_RED    = '#9D2235'   # deep red for feature branch
    CONCAT_GRAY = '#4A4A4A'   # dark gray for concat
    HEAD_GREEN  = '#0D5820'   # dark green for prediction head

    def draw_box(ax, x, y, w, h, text, fc, ec, fs=9, fc_text='white'):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                             facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fs, color=fc_text, zorder=3)

    ax.text(7, 4.25, 'Surrogate3DResNet Architecture (Single Ensemble Member, ~8.76M params)',
            fontsize=12, fontweight='bold', ha='center', va='center', color='#222222')

    # Stages with decreasing visual height to show downsampling
    stages = [
        (0.2,  1.4, 1.3, 2.2, 'Input\n7×128³', '#2B5EA7',  CONV_DARK, 10),
        (1.8,  1.5, 1.2, 2.0, 'Conv 1\n64×64³\nBN+GELU',   CONV_BLUE, CONV_DARK, 9),
        (3.3,  1.6, 1.2, 1.8, 'Conv 2\n128×32³\nBN+GELU',  CONV_BLUE, CONV_DARK, 9),
        (4.8,  1.7, 1.2, 1.6, 'Conv 3\n256×16³\nBN+GELU',  '#163F6E', CONV_DARK, 9),
        (6.3,  1.8, 1.2, 1.4, 'Conv 4\n512×8³\nBN+GELU',   '#0E2D50', CONV_DARK, 9),
        (7.8,  1.7, 1.3, 1.6, 'SE-ResBlock\n×3\nSE(r=4)',  SE_AMBER,  '#7A3F06', 9),
        (9.4,  1.8, 1.3, 1.4, 'Dual Pool\nAvg+Max\n512-d', POOL_GREEN,'#0A4D1E', 9),
    ]

    for x, y, w, h, text, fc, ec, fs in stages:
        draw_box(ax, x, y, w, h, text, fc, ec, fs)

    # Arrows between stages
    arrow_pairs = [(1.5, 1.8), (3.0, 3.3), (4.5, 4.8), (6.0, 6.3), (7.5, 7.8), (9.1, 9.4)]
    for x1, x2 in arrow_pairs:
        ax.annotate('', xy=(x2, 2.5), xytext=(x1, 2.5),
                    arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5), zorder=1)

    # Feature vector branch (bottom)
    draw_box(ax, 0.2, 0.15, 1.3, 0.9, 'Feature\nVector\n10-d', FEAT_RED, '#5A1220', 9)
    draw_box(ax, 1.8, 0.15, 1.2, 0.9, 'Feature\nMLP\n128-d', FEAT_RED, '#5A1220', 9)

    ax.annotate('', xy=(1.8, 0.6), xytext=(1.5, 0.6),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.2), zorder=1)

    # Concat box
    draw_box(ax, 10.9, 1.3, 0.9, 2.4, '⊕\nConcat\n640-d', CONCAT_GRAY, '#222222', 9)

    # Arrows to concat
    ax.annotate('', xy=(10.9, 2.7), xytext=(10.7, 2.5),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.2), zorder=1)
    ax.annotate('', xy=(10.9, 1.8), xytext=(3.0, 1.05),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2,
                                linestyle='dashed', connectionstyle='arc3,rad=-0.25'),
                zorder=1)

    # Prediction head
    draw_box(ax, 12.1, 1.5, 1.7, 2.0, 'Pred Head\n640→512→256\n+Skip\n→ 3 outputs\n(σ, u, C)',
             HEAD_GREEN, '#0A4D1E', 8)

    ax.annotate('', xy=(12.1, 2.5), xytext=(11.8, 2.5),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5), zorder=1)

    # Dimension annotations (above boxes)
    dims = ['128³', '64³', '32³', '16³', '8³']
    dim_xs = [0.85, 2.4, 3.9, 5.4, 6.9]
    for d, dx in zip(dims, dim_xs):
        ax.text(dx, 3.85, d, ha='center', va='center', fontsize=7.5,
                color='#555555', fontweight='bold')

    plt.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIGURES_DIR, 'fig2_architecture.png'))
    fig.savefig(os.path.join(FIGURES_DIR, 'fig2_architecture.pdf'))
    plt.close(fig)
    print("  -> Saved fig2_architecture.png/pdf")


# ============================================================
# Figure 3: Wireframe-to-Volume Pipeline
# ============================================================
def generate_wireframe_pipeline_figure():
    print("Generating fig_wireframe_pipeline.png ...")

    # Load reference case data
    wire_data = np.load('optimization/data/3dwire_raw/00472.npz')
    vertices = wire_data['vertices']
    lines = wire_data['lines']

    occ = np.load('fea_ml/runs/v3/optimization_128/fixed_occ.npz')['data']
    part = np.load('fea_ml/runs/v3/optimization_128/fixed_part.npz')['data']

    fig = plt.figure(figsize=(15, 4.5))

    # Panel (a): Wireframe
    ax1 = fig.add_subplot(131, projection='3d')
    render_wireframe_3d(ax1, vertices, lines, title='(a) 3DWire Wireframe', elev=20, azim=-55)

    # Panel (b): Volumetric model with part colors
    ax2 = fig.add_subplot(132, projection='3d')
    render_voxels_3d(ax2, occ, part, alpha=0.95, title='(b) Volumetric Model\n(4 part types)',
                     elev=20, azim=-55, downsample=3)

    # Panel (c): Voxelized 128^3 (single color, showing grid structure)
    ax3 = fig.add_subplot(133, projection='3d')
    render_voxels_3d(ax3, occ, part_labels=None, alpha=0.85,
                     title='(c) Voxelized 128³ Grid', elev=20, azim=-55, downsample=3)

    # Add legend for part types
    legend_elements = [
        mpatches.Patch(color=(0.08, 0.30, 0.70), label='Exterior Wall'),
        mpatches.Patch(color=(0.15, 0.60, 0.15), label='Interior Wall'),
        mpatches.Patch(color=(0.85, 0.45, 0.00), label='Roof'),
        mpatches.Patch(color=(0.75, 0.10, 0.10), label='Floor'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig_wireframe_pipeline.png'))
    fig.savefig(os.path.join(FIGURES_DIR, 'fig_wireframe_pipeline.pdf'))
    plt.close(fig)
    print("  -> Saved fig_wireframe_pipeline.png/pdf")


# ============================================================
# Figure 4: Model Comparison (wireframe -> volumetric -> SASTO-U -> SASTO-PA)
# ============================================================
def generate_model_comparison_figure():
    print("Generating fig_model_comparison.png ...")

    # Load data
    wire_data = np.load('optimization/data/3dwire_raw/00472.npz')
    vertices = wire_data['vertices']
    lines = wire_data['lines']

    occ_orig = np.load('fea_ml/runs/v3/optimization_128/fixed_occ.npz')['data']
    part = np.load('fea_ml/runs/v3/optimization_128/fixed_part.npz')['data']

    # v12 = uniform (SASTO-U, ~34.3%), v11 = part-aware (SASTO-PA, ~45.0%)
    occ_uniform = np.load('fea_ml/runs/v3/optimization_128/optimized_occ_v12.npz')['data']
    occ_partaware = np.load('fea_ml/runs/v3/optimization_128/optimized_occ_v11.npz')['data']

    elev, azim = 20, -55

    fig = plt.figure(figsize=(16, 4.5))

    # Panel (a): Wireframe
    ax1 = fig.add_subplot(141, projection='3d')
    render_wireframe_3d(ax1, vertices, lines, title='(a) 3DWire Wireframe', elev=elev, azim=azim)

    # Panel (b): Original volumetric model
    ax2 = fig.add_subplot(142, projection='3d')
    render_voxels_3d(ax2, occ_orig, part, alpha=0.95,
                     title='(b) Original Volumetric', elev=elev, azim=azim, downsample=3)

    # Panel (c): SASTO-U optimized (uniform)
    ax3 = fig.add_subplot(143, projection='3d')
    # Use part labels on optimized result
    part_u = part * occ_uniform  # only show parts where still occupied
    render_voxels_3d(ax3, occ_uniform, part_u, alpha=0.95,
                     title='(c) SASTO-U (34.3%)', elev=elev, azim=azim, downsample=3)

    # Panel (d): SASTO-PA optimized (part-aware)
    ax4 = fig.add_subplot(144, projection='3d')
    part_pa = part * occ_partaware
    render_voxels_3d(ax4, occ_partaware, part_pa, alpha=0.95,
                     title='(d) SASTO-PA (45.0%)', elev=elev, azim=azim, downsample=3)

    # Legend
    legend_elements = [
        mpatches.Patch(color=(0.08, 0.30, 0.70), label='Exterior Wall'),
        mpatches.Patch(color=(0.15, 0.60, 0.15), label='Interior Wall'),
        mpatches.Patch(color=(0.85, 0.45, 0.00), label='Roof'),
        mpatches.Patch(color=(0.75, 0.10, 0.10), label='Floor'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig_model_comparison.png'))
    fig.savefig(os.path.join(FIGURES_DIR, 'fig_model_comparison.pdf'))
    plt.close(fig)
    print("  -> Saved fig_model_comparison.png/pdf")


# ============================================================
# Run all
# ============================================================
if __name__ == '__main__':
    generate_pipeline_figure()
    generate_architecture_figure()
    generate_wireframe_pipeline_figure()
    generate_model_comparison_figure()
    print("\nAll placeholder figures generated successfully!")
