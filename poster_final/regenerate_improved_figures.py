#!/usr/bin/env python3
"""
Extract individual house renders from the existing high-quality composite figures
produced by render_figures.py. Also creates the reference-style Visual Abstract.

Uses: figures/fig_model_comparison.png (4700x2397) — wireframe + 3 models x 2 rows
      figures/fig_type_comparison.png (2956x4183) — 3 models x 4 views
      figures/fig12_stl_comparison.png (2679x2293) — orig vs PA, solid + cutaway
      figures/fig_wireframe_pipeline.png (4312x2535) — wireframe to voxel pipeline
      figures/fig_cross_section_comparison.png (4118x1463) — 3 cutaways
"""

import os
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ── Paths ────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(BASE, "figures")
OUT = os.path.dirname(os.path.abspath(__file__))
RENDERS_HQ = os.path.join(OUT, "renders_hq")
os.makedirs(RENDERS_HQ, exist_ok=True)

# ── Poster color palette ────────────────────────────────────
TEAL   = "#008C9E"
RED    = "#D7263D"
GOLD   = "#CFA535"
CARD   = "#F7F9FC"
DARK   = "#0B1736"
BLUE   = "#0A3D9A"
NAVY   = "#062B7A"
SPINE  = "#999999"
LIGHT_BLUE = "#E8EEF8"
WHITE  = "#FFFFFF"
EXT_WALL = "#4A7FC1"
INT_WALL = "#E8833A"
ROOF_COLOR = "#6AAF6E"
GREEN  = "#2E8B57"
BG_RENDER = "#E6E8EB"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "text.color": DARK,
    "figure.facecolor": CARD,
    "axes.facecolor": CARD,
    "savefig.facecolor": CARD,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.png"),
                bbox_inches="tight", pad_inches=0.08, dpi=300,
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"  -> {name}.png")


# ═══════════════════════════════════════════════════════════════
# STEP 1: Extract individual renders from composite figures
# ═══════════════════════════════════════════════════════════════

def extract_renders():
    """Extract individual house renders from composite figures."""
    print("\n=== Extracting renders from composite figures ===")

    # ── From fig_model_comparison.png (4700x2397) ──
    mc = os.path.join(FIGS, "fig_model_comparison.png")
    if os.path.isfile(mc):
        img = Image.open(mc)
        w, h = img.size
        print(f"  fig_model_comparison: {w}x{h}")

        wire_end = int(w * 0.286)
        col_w = (w - wire_end) // 3
        mid_h = h // 2

        wireframe = img.crop((0, 0, wire_end, h))
        wireframe.save(os.path.join(RENDERS_HQ, "wireframe_panel.png"))
        print(f"    wireframe_panel: {wireframe.size}")

        for i, name in enumerate(["original_solid", "sasto_u_solid", "sasto_pa_solid"]):
            x0 = wire_end + i * col_w
            x1 = wire_end + (i + 1) * col_w
            crop = img.crop((x0, 0, x1, mid_h))
            crop.save(os.path.join(RENDERS_HQ, f"{name}.png"))
            print(f"    {name}: {crop.size}")

        for i, name in enumerate(["original_cutaway", "sasto_u_cutaway", "sasto_pa_cutaway"]):
            x0 = wire_end + i * col_w
            x1 = wire_end + (i + 1) * col_w
            crop = img.crop((x0, mid_h, x1, h))
            crop.save(os.path.join(RENDERS_HQ, f"{name}.png"))
            print(f"    {name}: {crop.size}")

    # ── From fig12_stl_comparison.png (2679x2293) ──
    sc = os.path.join(FIGS, "fig12_stl_comparison.png")
    if os.path.isfile(sc):
        img = Image.open(sc)
        w, h = img.size
        print(f"  fig12_stl_comparison: {w}x{h}")
        mid_w, mid_h = w // 2, h // 2

        crops = {
            "stl_orig_solid": (0, 0, mid_w, mid_h),
            "stl_pa_solid": (mid_w, 0, w, mid_h),
            "stl_orig_cutaway": (0, mid_h, mid_w, h),
            "stl_pa_cutaway": (mid_w, mid_h, w, h),
        }
        for name, box in crops.items():
            crop = img.crop(box)
            crop.save(os.path.join(RENDERS_HQ, f"{name}.png"))
            print(f"    {name}: {crop.size}")

    # ── From fig_cross_section_comparison.png (4118x1463) ──
    cs = os.path.join(FIGS, "fig_cross_section_comparison.png")
    if os.path.isfile(cs):
        img = Image.open(cs)
        w, h = img.size
        print(f"  fig_cross_section_comparison: {w}x{h}")
        col_w = w // 3
        for i, name in enumerate(["xs_original", "xs_sasto_u", "xs_sasto_pa"]):
            crop = img.crop((i * col_w, 0, (i+1) * col_w, h))
            crop.save(os.path.join(RENDERS_HQ, f"{name}.png"))
            print(f"    {name}: {crop.size}")

    # ── Copy full composite figures for poster use ──
    from shutil import copy2
    for fname in ["fig_wireframe_pipeline.png", "fig_model_comparison.png",
                   "fig_cross_section_comparison.png", "fig_type_comparison.png",
                   "fig_optimized_gallery.png", "fig_diverse_stl_gallery.png",
                   "fig12_stl_comparison.png"]:
        src = os.path.join(FIGS, fname)
        if os.path.isfile(src):
            copy2(src, os.path.join(RENDERS_HQ, fname))
            print(f"    copied {fname}")


def load_hq(name, max_size=None):
    """Load an extracted HQ render."""
    path = os.path.join(RENDERS_HQ, f"{name}.png")
    if not os.path.isfile(path):
        return None
    img = Image.open(path).convert("RGBA")
    if max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    return np.array(img)


def embed_image(ax, img_array, x, y, zoom=0.15):
    if img_array is None:
        return
    im = OffsetImage(img_array, zoom=zoom)
    ab = AnnotationBbox(im, (x, y), frameon=False, pad=0)
    ax.add_artist(ab)


# ═══════════════════════════════════════════════════════════════
# STEP 2: VISUAL ABSTRACT — Reference-style with bracket + visuals
# ═══════════════════════════════════════════════════════════════

def fig_visual_abstract():
    """
    Reference-style Visual Abstract: two columns with curly bracket.
    Left: "a) User Workflow" — 3 steps with real house render thumbnails
    Right: "b) Model Creation" — 5 steps with process icons + renders
    Connected by bracket like Rishab Jain's poster.
    """
    fig = plt.figure(figsize=(11.5, 8.5), facecolor=CARD)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    # ── Top banner ──
    banner = FancyBboxPatch((0.3, 7.8), 10.9, 0.55,
                            boxstyle="round,pad=0.08",
                            facecolor=NAVY, edgecolor=NAVY)
    ax.add_patch(banner)
    ax.text(5.75, 8.08, "SASTO: Surrogate-Accelerated Structural Topology Optimization",
            ha="center", va="center", fontsize=13, fontweight="bold", color=WHITE)

    # ── Column headers with decorative underline ──
    ax.text(2.75, 7.50, "a) User Workflow",
            ha="center", fontsize=13, fontweight="bold", color=DARK)
    ax.plot([0.4, 5.1], [7.35, 7.35], color=TEAL, lw=2.5, solid_capstyle="round")

    ax.text(8.65, 7.50, "b) Model Creation",
            ha="center", fontsize=13, fontweight="bold", color=DARK)
    ax.plot([6.4, 10.9], [7.35, 7.35], color=GOLD, lw=2.5, solid_capstyle="round")

    # ── Center bracket (vertical with pointed midpoint) ──
    bx = 5.75
    by_top = 7.2
    by_bot = 0.8
    by_mid = (by_top + by_bot) / 2

    # Draw a stylized bracket { }
    from matplotlib.path import Path as MPath
    from matplotlib.patches import PathPatch

    # Left half of bracket
    verts_L = [
        (bx - 0.08, by_top), (bx - 0.15, by_top - 0.15),
        (bx - 0.15, by_mid + 0.3), (bx - 0.25, by_mid),
        (bx - 0.15, by_mid - 0.3), (bx - 0.15, by_bot + 0.15),
        (bx - 0.08, by_bot),
    ]
    codes_L = [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3, MPath.CURVE3,
               MPath.CURVE3, MPath.CURVE3, MPath.CURVE3]
    path_L = MPath(verts_L, codes_L)
    pp_L = PathPatch(path_L, facecolor='none', edgecolor='#AAAAAA', lw=2)
    ax.add_patch(pp_L)

    # Right half of bracket (mirror)
    verts_R = [
        (bx + 0.08, by_top), (bx + 0.15, by_top - 0.15),
        (bx + 0.15, by_mid + 0.3), (bx + 0.25, by_mid),
        (bx + 0.15, by_mid - 0.3), (bx + 0.15, by_bot + 0.15),
        (bx + 0.08, by_bot),
    ]
    path_R = MPath(verts_R, codes_L)
    pp_R = PathPatch(path_R, facecolor='none', edgecolor='#AAAAAA', lw=2)
    ax.add_patch(pp_R)

    # ── LEFT COLUMN: User Workflow ──

    def flow_box(x, y, w, h, title, border_color, banner_h=0.40):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.06",
                              facecolor=WHITE, edgecolor=border_color, lw=1.8)
        ax.add_patch(rect)
        ban = FancyBboxPatch((x + 0.02, y + h - banner_h + 0.02), w - 0.04, banner_h - 0.04,
                                boxstyle="round,pad=0.04",
                                facecolor=border_color, edgecolor=border_color)
        ax.add_patch(ban)
        ax.text(x + w/2, y + h - banner_h/2, title,
                ha="center", va="center", fontsize=9.5, fontweight="bold",
                color=WHITE)

    def flow_arrow_v(x, y1, y2, color=GOLD):
        ax.annotate("", xy=(x, y2), xytext=(x, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=2.5, mutation_scale=18))

    lx, lw = 0.4, 4.7

    # Box 1: Input Building Design
    b1_y, b1_h = 5.5, 1.70
    flow_box(lx, b1_y, lw, b1_h, "1. Input: 3D Building Wireframe", TEAL)

    thumb_orig = load_hq("original_solid", max_size=400)
    if thumb_orig is not None:
        embed_image(ax, thumb_orig, 1.6, b1_y + 0.55, zoom=0.18)
    ax.text(3.4, b1_y + 0.85, "3DWire wireframe\nskeleton input", ha="center",
            fontsize=9, color=DARK)
    ax.text(3.4, b1_y + 0.25, "Volumetric parts:\next, int, roof, floor",
            ha="center", fontsize=8, color=SPINE, fontstyle="italic")

    flow_arrow_v(lx + lw/2, b1_y, b1_y - 0.15, GOLD)

    # Box 2: SASTO Optimization
    b2_y, b2_h = 3.1, 2.05
    flow_box(lx, b2_y, lw, b2_h, "2. SASTO Optimization (~50 sec)", RED)

    steps = [
        "Backprop sensitivity through ensemble",
        "Rank & remove safe voxels (6-conn)",
        "Trust-region constraint checking",
        "Phase 1->2->3 convergence",
    ]
    for i, step in enumerate(steps):
        bcy = b2_y + 1.35 - i * 0.30
        circ = Circle((lx + 0.18, bcy), 0.06, facecolor=RED, edgecolor=RED)
        ax.add_patch(circ)
        ax.text(lx + 0.32, bcy, step, fontsize=8, color=DARK, va="center")

    flow_arrow_v(lx + lw/2, b2_y, b2_y - 0.15, GOLD)

    # Box 3: Output Optimized STL
    b3_y, b3_h = 0.6, 2.15
    flow_box(lx, b3_y, lw, b3_h, "3. Output: Optimized Watertight STL", GREEN)

    thumb_pa = load_hq("sasto_pa_solid", max_size=400)
    if thumb_pa is not None:
        embed_image(ax, thumb_pa, 1.6, b3_y + 0.55, zoom=0.18)
    ax.text(3.4, b3_y + 1.05, "Watertight mesh\nfor 3D printing", ha="center",
            fontsize=9, color=DARK)

    badges = [
        ("-23.5% material", RED),
        ("0 violations", TEAL),
        ("50s median", GOLD),
    ]
    bx_start = 2.7
    for j, (btxt, bclr) in enumerate(badges):
        bpill = FancyBboxPatch((bx_start + j * 1.2, b3_y + 0.12), 1.10, 0.30,
                               boxstyle="round,pad=0.04",
                               facecolor=bclr, edgecolor=bclr, alpha=0.9)
        ax.add_patch(bpill)
        ax.text(bx_start + j * 1.2 + 0.55, b3_y + 0.27, btxt,
                ha="center", va="center", fontsize=7, fontweight="bold", color=WHITE)

    # ── RIGHT COLUMN: Model Creation ──

    right_steps = [
        ("1. Retrieve 14,293\n   Building Wireframes", GOLD,
         "3DWire dataset [Lin 2024]", "wireframe_panel"),
        ("2. Generate FEA\n   Simulations", TEAL,
         "Gmsh mesh -> SfePy solver", None),
        ("3. Voxelize to\n   128^3 Grid", EXT_WALL,
         "7-channel: 4 parts + 3 loads", None),
        ("4. Train Deep\n   Ensemble (x5)", RED,
         "5x8.76M params, Huber loss", None),
        ("5. Conformal\n   Calibration", GREEN,
         "P(violation) <= 0.09%", None),
    ]

    rx, rw = 6.4, 4.5
    n_steps = len(right_steps)
    box_h = 0.95
    step_gap = 0.35
    start_y = 7.1

    for i, (text, color, detail, thumb_name) in enumerate(right_steps):
        by = start_y - i * (box_h + step_gap)

        rect = FancyBboxPatch((rx, by), rw, box_h,
                              boxstyle="round,pad=0.06",
                              facecolor=WHITE, edgecolor=color, lw=1.6)
        ax.add_patch(rect)

        # Step number circle
        circ = Circle((rx - 0.18, by + box_h/2), 0.18,
                      facecolor=color, edgecolor=color)
        ax.add_patch(circ)
        ax.text(rx - 0.18, by + box_h/2, str(i + 1),
                ha="center", va="center", fontsize=10,
                fontweight="bold", color=WHITE)

        if thumb_name:
            thumb = load_hq(thumb_name, max_size=200)
            if thumb is not None:
                embed_image(ax, thumb, rx + 0.55, by + box_h/2, zoom=0.10)

        text_x = rx + 1.0 if thumb_name else rx + 0.40
        ax.text(text_x, by + box_h * 0.68, text,
                fontsize=9, fontweight="bold", color=DARK, va="center")
        ax.text(text_x, by + box_h * 0.18, detail,
                fontsize=7.5, color=SPINE, fontstyle="italic", va="center")

        if i < n_steps - 1:
            next_top = start_y - (i + 1) * (box_h + step_gap) + box_h
            flow_arrow_v(rx + rw/2, by, next_top, color)

    save(fig, "fig01_visual_abstract_pipeline")


# ═══════════════════════════════════════════════════════════════
# STEP 3: Regenerate key poster figures using HQ renders
# ═══════════════════════════════════════════════════════════════

def fig_uniform_vs_optimized():
    """Side-by-side using high-quality renders."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 5.5))
    fig.patch.set_facecolor(CARD)

    for ax in (ax1, ax2):
        ax.axis("off")

    ax1.set_title("Conventional: Uniform Thickness", fontsize=12,
                  fontweight="bold", color=RED, pad=8)
    ax2.set_title("SASTO-PA: Part-Aware Optimized", fontsize=12,
                  fontweight="bold", color=TEAL, pad=8)

    orig = load_hq("stl_orig_solid", max_size=1000)
    optim = load_hq("stl_pa_solid", max_size=1000)

    if orig is not None:
        ax1.imshow(orig)
        ax1.text(0.5, 0.02, "All walls: 2-4 voxels (~156-316 mm)",
                 transform=ax1.transAxes, ha="center", fontsize=9,
                 color=RED, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=WHITE,
                           edgecolor=RED, alpha=0.9))
    else:
        ax1.text(0.5, 0.5, "[Original house render]",
                 transform=ax1.transAxes, ha="center", va="center",
                 fontsize=11, color=SPINE, fontstyle="italic")

    if optim is not None:
        ax2.imshow(optim)
        ax2.text(0.5, 0.02, "Ext: 156mm / Int: 78mm -> -45.0% material",
                 transform=ax2.transAxes, ha="center", fontsize=9,
                 color=TEAL, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=WHITE,
                           edgecolor=TEAL, alpha=0.9))
    else:
        ax2.text(0.5, 0.5, "[Optimized house render]",
                 transform=ax2.transAxes, ha="center", va="center",
                 fontsize=11, color=SPINE, fontstyle="italic")

    fig.text(0.5, 0.5, "vs.", ha="center", va="center",
             fontsize=22, fontweight="bold", color=RED,
             transform=fig.transFigure)

    fig.text(0.5, 0.01, "23.5% +/- 7.8% mean material reduction across 1,114 geometries",
             ha="center", fontsize=12, fontweight="bold", color=RED)

    save(fig, "fig02_uniform_vs_optimized")


def fig_ref_comparison():
    """Full reference case comparison using HQ renders — copy best existing."""
    from shutil import copy2
    src = os.path.join(FIGS, "fig12_stl_comparison.png")
    if os.path.isfile(src):
        dst = os.path.join(OUT, "fig_ref_comparison.png")
        copy2(src, dst)
        print(f"  -> fig_ref_comparison.png (from fig12_stl_comparison)")


def fig_connectivity():
    """6-conn vs 26-conn with actual cutaway renders."""
    fig = plt.figure(figsize=(11.0, 6.5))
    fig.patch.set_facecolor(CARD)

    ax1 = fig.add_axes([0.02, 0.12, 0.46, 0.78])
    ax2 = fig.add_axes([0.52, 0.12, 0.46, 0.78])

    for ax in (ax1, ax2):
        ax.axis("off")

    # ── Left: 26-connectivity FAILS ──
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)

    title_box = FancyBboxPatch((0.3, 7.0), 9.4, 0.8,
                               boxstyle="round,pad=0.08",
                               facecolor=RED, edgecolor=RED, alpha=0.9)
    ax1.add_patch(title_box)
    ax1.text(5.0, 7.4, "26-Connectivity — FAILS",
             ha="center", va="center", fontsize=14, fontweight="bold",
             color=WHITE)

    bg_box = FancyBboxPatch((0.5, 1.5), 9.0, 5.2,
                            boxstyle="round,pad=0.08",
                            facecolor="#F8F0F0", edgecolor="#DDCCCC", lw=0.8)
    ax1.add_patch(bg_box)

    np.random.seed(42)
    main_rect = plt.Rectangle((2.0, 2.0), 6.0, 3.5, facecolor="#E8E8E8",
                              edgecolor="#BBBBBB", lw=1.5, linestyle="--")
    ax1.add_patch(main_rect)
    ax1.text(5.0, 3.8, "Mesh splits into\nthousands of\nfloating fragments",
             ha="center", va="center", fontsize=10, color=RED,
             fontweight="bold", fontstyle="italic")

    for _ in range(50):
        fx = np.random.uniform(1.0, 9.0)
        fy = np.random.uniform(1.8, 6.4)
        fs = np.random.uniform(0.08, 0.30)
        angle = np.random.uniform(0, 60)
        alpha = 0.4 + np.random.uniform(0, 0.4)
        frag = plt.Rectangle((fx, fy), fs, fs * 0.8,
                             facecolor=RED, edgecolor="#AA1122",
                             alpha=alpha, linewidth=0.5, angle=angle)
        ax1.add_patch(frag)

    failures = [
        "x  Thousands of floating fragments",
        "x  Diagonal-only connections allowed",
        "x  Marching cubes incompatible",
        "x  Cannot generate AM toolpath",
    ]
    for i, txt in enumerate(failures):
        ax1.text(5.0, 1.2 - i * 0.28, txt,
                 fontsize=8.5, color=RED, fontweight="bold", ha="center")

    # ── Right: 6-connectivity WORKS ──
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)

    title_box2 = FancyBboxPatch((0.3, 7.0), 9.4, 0.8,
                                boxstyle="round,pad=0.08",
                                facecolor=TEAL, edgecolor=TEAL, alpha=0.9)
    ax2.add_patch(title_box2)
    ax2.text(5.0, 7.4, "6-Connectivity — WORKS (Ours)",
             ha="center", va="center", fontsize=14, fontweight="bold",
             color=WHITE)

    cutaway = load_hq("xs_sasto_pa", max_size=700)
    if cutaway is not None:
        embed_image(ax2, cutaway, 5.0, 3.8, zoom=0.40)
    else:
        cutaway2 = load_hq("sasto_pa_cutaway", max_size=700)
        if cutaway2 is not None:
            embed_image(ax2, cutaway2, 5.0, 3.8, zoom=0.35)
        else:
            ax2.text(5.0, 4.0, "[Cutaway render showing\nclean topology]",
                     ha="center", va="center", fontsize=10, color=SPINE,
                     fontstyle="italic")

    successes = [
        "v  1 connected component",
        "v  Face-share adjacency only",
        "v  Watertight STL confirmed",
        "v  AM toolpath compatible",
    ]
    for i, txt in enumerate(successes):
        ax2.text(5.0, 1.2 - i * 0.28, txt,
                 fontsize=8.5, color=TEAL, fontweight="bold", ha="center")

    fig.text(0.5, 0.52, "vs.", ha="center", va="center",
             fontsize=22, fontweight="bold", color=RED)

    fig.text(0.5, 0.02,
             "Proposition: 1 six-connected foreground component -> 1 marching-cubes surface mesh  "
             "[Kong & Rosenfeld 1989]",
             ha="center", fontsize=9, fontstyle="italic", color=DARK,
             bbox=dict(boxstyle="round,pad=0.4", facecolor=LIGHT_BLUE,
                       edgecolor=TEAL, lw=1))

    save(fig, "fig06_connectivity")


def fig_dataset_pipeline():
    """Dataset pipeline with actual render thumbnails."""
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    fig.patch.set_facecolor(CARD)

    thumb_names = [
        "wireframe_panel",
        "original_solid",
        "sasto_pa_cutaway",
        "sasto_pa_solid",
    ]
    thumbnails = [load_hq(n, max_size=300) for n in thumb_names]

    stages = [
        ("Stage 1", "3DWire\nSkeleton", "14,293 buildings", TEAL),
        ("Stage 2", "Volumetric\nParts", "4-part labels", EXT_WALL),
        ("Stage 3", "FEA\nSimulation", "11,178 valid", RED),
        ("Stage 4", "128^3 Voxel\nGrid", "8,943 train", GOLD),
    ]

    box_w, box_h = 2.3, 3.8
    gap = 0.5
    start_x = 0.3

    for i, (label, title, sub, color) in enumerate(stages):
        x = start_x + i * (box_w + gap)
        y = 1.0

        rect = FancyBboxPatch((x, y), box_w, box_h,
                              boxstyle="round,pad=0.08", linewidth=1.5,
                              edgecolor=color, facecolor=WHITE)
        ax.add_patch(rect)

        banner = FancyBboxPatch((x, y + box_h - 0.5), box_w, 0.5,
                                boxstyle="round,pad=0.04", facecolor=color,
                                edgecolor=color)
        ax.add_patch(banner)
        ax.text(x + box_w/2, y + box_h - 0.25, label,
                ha="center", va="center", fontsize=10, fontweight="bold",
                color=WHITE)

        if thumbnails[i] is not None:
            embed_image(ax, thumbnails[i], x + box_w/2, y + 1.85, zoom=0.25)
        else:
            ph = FancyBboxPatch((x + 0.15, y + 0.8), box_w - 0.3, 2.0,
                                boxstyle="round,pad=0.04", linewidth=0.8,
                                edgecolor="#CCCCCC", facecolor=LIGHT_BLUE,
                                linestyle="--")
            ax.add_patch(ph)

        ax.text(x + box_w/2, y + 0.55, title,
                ha="center", va="center", fontsize=10, fontweight="bold",
                color=DARK)
        ax.text(x + box_w/2, y + 0.15, sub,
                ha="center", va="center", fontsize=8, fontstyle="italic",
                color=SPINE)

        if i < 3:
            ax.annotate("", xy=(x + box_w + gap * 0.15, y + box_h/2),
                        xytext=(x + box_w + 0.02, y + box_h/2),
                        arrowprops=dict(arrowstyle="-|>", color=GOLD,
                                        lw=2.5, mutation_scale=18))

    arrow_labels = ["Extrude +\nBoolean", "Gmsh +\nSfePy FEA", "Trimesh\nvoxelization"]
    for i, lbl in enumerate(arrow_labels):
        x = start_x + i * (box_w + gap) + box_w + gap/2
        ax.text(x, 1.0 + box_h/2 + 0.5, lbl,
                ha="center", va="bottom", fontsize=7, fontstyle="italic",
                color=DARK)

    stats = [("Train", "8,943"), ("Val", "1,121"), ("Test", "1,114")]
    for j, (label, n) in enumerate(stats):
        sx = 3.0 + j * 2.5
        rect = FancyBboxPatch((sx - 0.8, 0.15), 1.6, 0.5,
                              boxstyle="round,pad=0.05",
                              facecolor=TEAL if j < 2 else RED,
                              edgecolor=TEAL if j < 2 else RED)
        ax.add_patch(rect)
        ax.text(sx, 0.40, f"{label}: {n}",
                ha="center", va="center", fontsize=9, fontweight="bold",
                color=WHITE)

    save(fig, "fig03_dataset_pipeline")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Improved poster figures — HQ renders + reference-style VA")
    print("=" * 60)

    print("\n[1] Extracting renders from composite figures...")
    extract_renders()

    print("\n[2] Visual Abstract (reference-style with bracket)...")
    fig_visual_abstract()

    print("\n[3] Uniform vs Optimized (HQ renders)...")
    fig_uniform_vs_optimized()

    print("\n[4] Reference case comparison...")
    fig_ref_comparison()

    print("\n[5] Connectivity (with HQ cutaway)...")
    fig_connectivity()

    print("\n[6] Dataset Pipeline (with HQ thumbnails)...")
    fig_dataset_pipeline()

    print("\n" + "=" * 60)
    print("Done. Updated figures in poster_final/")
    print("=" * 60)


if __name__ == "__main__":
    main()
