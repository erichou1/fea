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
import graphviz
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

    ax.scatter(xs, ys, zs, c=colors, s=8.0, marker='s', linewidths=0, depthshade=True)
    ax.view_init(elev=elev, azim=azim)
    # Tight limits around occupied region so model fills the panel
    pad = 1
    x_lo, x_hi = xs.min() - pad, xs.max() + pad
    y_lo, y_hi = ys.min() - pad, ys.max() + pad
    z_lo, z_hi = zs.min() - pad, zs.max() + pad
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_zlim(z_lo, z_hi)
    ax.set_axis_off()
    # Aspect ratio matching occupied extents
    rx, ry, rz = x_hi - x_lo, y_hi - y_lo, z_hi - z_lo
    rmax = max(rx, ry, rz)
    ax.set_box_aspect([rx / rmax, ry / rmax, rz / rmax])
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=-5)


def render_wireframe_3d(ax, vertices, lines, title=None, elev=25, azim=-60):
    """Render 3D wireframe from vertices and edge indices."""
    for edge in lines:
        v0, v1 = vertices[edge[0]], vertices[edge[1]]
        ax.plot3D(*zip(v0, v1), color='#333333', linewidth=2.5)

    # Plot vertices as dots
    ax.scatter3D(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                 color='#CC3333', s=30, zorder=5, depthshade=False)

    ax.view_init(elev=elev, azim=azim)
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    ranges = hi - lo
    pad = ranges * 0.02  # minimal proportional padding
    for i, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
        setter(lo[i] - pad[i], hi[i] + pad[i])
    # Match aspect to data so wireframe fills the panel
    ax.set_box_aspect(ranges / ranges.max() if ranges.max() > 0 else [1, 1, 1])
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=-5)


# ============================================================
# Figure 1: SASTO Pipeline Overview  (Graphviz)
# ============================================================
def generate_pipeline_figure():
    print("Generating fig1_pipeline.png ...")

    dot = graphviz.Digraph('pipeline', engine='dot')
    dot.attr(rankdir='TB', dpi='300', size='7,10!',
             fontname='Arial', bgcolor='white',
             nodesep='0.5', ranksep='0.7', margin='0.3')

    dot.attr('node', shape='box', style='filled,rounded',
             fontname='Arial', fontsize='20',
             fontcolor='white', penwidth='3.0', margin='0.30,0.20')
    dot.attr('edge', fontname='Arial', fontsize='16',
             penwidth='3.0', arrowsize='1.3')

    # ── Offline cluster ──
    with dot.subgraph(name='cluster_offline') as c:
        c.attr(label='  OFFLINE TRAINING PHASE  ', labelloc='t',
               style='filled,rounded', fillcolor='#EDF2F9',
               color='#0D3B66', fontcolor='#0D3B66',
               fontsize='22', fontname='Arial',
               penwidth='3.0')

        c.node('wire', '3DWire\nWireframes\n(14,293)',
               fillcolor='#1B5E9E', color='#0A2F5C')
        c.node('vol', 'Volumetric\nGeneration\n(4-part STL)',
               fillcolor='#1B5E9E', color='#0A2F5C')
        c.node('fea', 'FEA Simulation\n(SfePy, ASCE 7-22)\nσ, u, C',
               fillcolor='#1B5E9E', color='#0A2F5C')
        c.node('ensemble', 'Deep Ensemble Training\n(5× Surrogate3DResNet)\n11,178 samples',
               fillcolor='#0A2F5C', color='#051A33')

        c.edge('wire', 'vol', color='#0D3B66')
        c.edge('vol', 'fea', color='#0D3B66')
        c.edge('fea', 'ensemble', color='#0D3B66')

    # ── Online cluster ──
    with dot.subgraph(name='cluster_online') as c:
        c.attr(label='  ONLINE OPTIMIZATION PHASE  ', labelloc='t',
               style='filled,rounded', fillcolor='#FDF3EA',
               color='#8C3D00', fontcolor='#8C3D00',
               fontsize='22', fontname='Arial',
               penwidth='3.0')

        c.node('voxel', '128³ Voxel Grid\n+ Part Labels',
               fillcolor='#D46A00', color='#5C2D00')
        c.node('phase1', 'Phase 1:\nSensitivity Erosion\n(99% removal)',
               fillcolor='#BF5700', color='#5C2D00')
        c.node('phase2', 'Phase 2:\nEndgame\n(single voxel)',
               fillcolor='#BF5700', color='#5C2D00')
        c.node('phase3', 'Phase 3:\nSwap Refinement',
               fillcolor='#BF5700', color='#5C2D00')
        c.node('export', 'Post-Process\n+ STL Export\n(marching cubes)',
               fillcolor='#1E7B3A', color='#0A4D1E')

        c.edge('voxel', 'phase1', color='#8C3D00')
        c.edge('phase1', 'phase2', color='#8C3D00')
        c.edge('phase2', 'phase3', color='#8C3D00')
        c.edge('phase3', 'export', color='#444444')

    # Cross-phase: ensemble → Phase 1
    dot.edge('ensemble', 'phase1',
             style='dashed', color='#666666', fontcolor='#333333',
             label='  Surrogate predictions  ',
             penwidth='1.5')

    # Render PNG and PDF
    output_base = os.path.join(FIGURES_DIR, 'fig1_pipeline')
    for fmt in ('png', 'pdf'):
        dot.format = fmt
        dot.render(output_base, cleanup=True)
    print("  -> Saved fig1_pipeline.png/pdf")


# ============================================================
# Figure 2: Surrogate3DResNet Architecture  (Graphviz)
# ============================================================
def generate_architecture_figure():
    print("Generating fig2_architecture.png ...")

    dot = graphviz.Digraph('architecture', engine='dot')
    dot.attr(rankdir='TB', dpi='300', size='8,10!',
             fontname='Arial', bgcolor='white',
             nodesep='0.4', ranksep='0.6', margin='0.3',
             label='Surrogate3DResNet Architecture  (Single Ensemble Member, ~8.76M params)',
             labelloc='t', fontsize='20', labeljust='c')

    dot.attr('node', shape='box', style='filled,rounded',
             fontname='Arial', fontsize='18',
             fontcolor='white', penwidth='2.8', margin='0.24,0.18')
    dot.attr('edge', fontname='Arial', fontsize='14',
             penwidth='2.8', arrowsize='1.2')

    # ── 3D CNN Backbone ──
    with dot.subgraph(name='cluster_cnn') as c:
        c.attr(label='3D CNN Backbone', labelloc='t',
               style='filled,rounded', fillcolor='#EDF2F9',
               color='#0D2E52', fontcolor='#0D2E52',
               fontsize='15', penwidth='2.0')

        c.node('input', 'Input\n7 × 128³',
               fillcolor='#2B5EA7', color='#0D2E52')
        c.node('conv1', 'Conv 1\n64 × 64³\nBN + GELU',
               fillcolor='#1B4F8A', color='#0D2E52')
        c.node('conv2', 'Conv 2\n128 × 32³\nBN + GELU',
               fillcolor='#1B4F8A', color='#0D2E52')
        c.node('conv3', 'Conv 3\n256 × 16³\nBN + GELU',
               fillcolor='#163F6E', color='#0D2E52')
        c.node('conv4', 'Conv 4\n512 × 8³\nBN + GELU',
               fillcolor='#0E2D50', color='#0D2E52')

        c.edge('input', 'conv1', color='#333333')
        c.edge('conv1', 'conv2', color='#333333')
        c.edge('conv2', 'conv3', color='#333333')
        c.edge('conv3', 'conv4', color='#333333')

    # ── SE-ResBlock + Pooling ──
    with dot.subgraph(name='cluster_se') as c:
        c.attr(label='SE-Res + Pool', labelloc='t',
               style='filled,rounded', fillcolor='#FDF3EA',
               color='#7A3F06', fontcolor='#7A3F06',
               fontsize='15', penwidth='2.0')

        c.node('se', 'SE-ResBlock\n× 3\nSE(r=4)',
               fillcolor='#B8600A', color='#7A3F06')
        c.node('pool', 'Dual Pool\nAvg + Max\n512-d',
               fillcolor='#1A6B35', color='#0A4D1E')

        c.edge('se', 'pool', color='#333333')

    dot.edge('conv4', 'se', color='#333333')

    # ── Feature Branch ──
    with dot.subgraph(name='cluster_feat') as c:
        c.attr(label='Feature Branch', labelloc='t',
               style='filled,rounded', fillcolor='#FCEEF1',
               color='#5A1220', fontcolor='#5A1220',
               fontsize='15', penwidth='2.0')

        c.node('feat', 'Feature Vector\n10-d',
               fillcolor='#9D2235', color='#5A1220')
        c.node('mlp', 'Feature MLP\n128-d',
               fillcolor='#9D2235', color='#5A1220')

        c.edge('feat', 'mlp', color='#333333')

    # ── Concat + Head ──
    dot.node('concat', '⊕ Concat\n640-d',
             fillcolor='#4A4A4A', color='#222222')
    dot.node('head', 'Prediction Head\n640 → 512 → 256 + Skip\n→ 3 outputs (σ, u, C)',
             fillcolor='#0D5820', color='#0A4D1E')

    dot.edge('pool', 'concat', color='#333333')
    dot.edge('mlp', 'concat', color='#555555', style='dashed',
             label='  128-d  ', fontcolor='#555555')
    dot.edge('concat', 'head', color='#333333')

    # Render
    output_base = os.path.join(FIGURES_DIR, 'fig2_architecture')
    for fmt in ('png', 'pdf'):
        dot.format = fmt
        dot.render(output_base, cleanup=True)
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

    fig = plt.figure(figsize=(18, 7))

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

    fig = plt.figure(figsize=(20, 7))

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
