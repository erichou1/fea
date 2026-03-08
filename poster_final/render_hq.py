#!/usr/bin/env python3
"""
Generate poster-specific renders using the better renderer (render_figures.py).
Produces individual house renders + thumbnails cropped for poster embedding.
Saves to poster_final/renders_hq/
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
from pathlib import Path

# Import the high-quality renderer
from render_figures import (
    load_ref, load_sample, build_colored_mesh, build_cutaway,
    compute_camera_poses, render_mesh, trim_whitespace,
    RENDER_W, RENDER_H, BG_COLOR, TARGET_FACES, BLUR_SIGMA
)

OUT = Path(__file__).parent / "renders_hq"
OUT.mkdir(exist_ok=True)


def save_render(img_array, name, max_w=1200, max_h=900):
    """Save a trimmed render, resized for poster embedding."""
    img = trim_whitespace(img_array)
    pil = Image.fromarray(img)
    # Resize to poster-friendly dimensions while keeping aspect ratio
    pil.thumbnail((max_w, max_h), Image.LANCZOS)
    out_path = OUT / f"{name}.png"
    pil.save(str(out_path), dpi=(300, 300))
    print(f"  -> {name}.png  ({pil.size[0]}x{pil.size[1]})")
    return out_path


def render_ref_case():
    """Render the reference case: original, SASTO-U, SASTO-PA in isometric + cutaway."""
    print("\n=== Reference Case Renders ===")
    part, occ_orig, occ_v11, occ_v12 = load_ref()

    # Build meshes
    orig = build_colored_mesh(occ_orig, part)
    pa = build_colored_mesh(occ_v11, part)
    u = build_colored_mesh(occ_v12, part)

    orig_cut = build_cutaway(occ_orig, part)
    pa_cut = build_cutaway(occ_v11, part)

    # Camera poses from original (consistent framing)
    poses, _ = compute_camera_poses(orig)

    # Isometric views
    cam = poses['isometric']
    print("  Original isometric...")
    save_render(render_mesh(orig, cam), "ref_original_iso")
    print("  SASTO-PA isometric...")
    save_render(render_mesh(pa, cam), "ref_sasto_pa_iso")
    print("  SASTO-U isometric...")
    save_render(render_mesh(u, cam), "ref_sasto_u_iso")

    # Front views
    cam_f = poses['front']
    print("  Original front...")
    save_render(render_mesh(orig, cam_f), "ref_original_front")
    print("  SASTO-PA front...")
    save_render(render_mesh(pa, cam_f), "ref_sasto_pa_front")

    # Cutaway views (front camera)
    cam_cut = poses['cutaway_front']
    print("  Original cutaway...")
    if orig_cut:
        save_render(render_mesh(orig_cut, cam_cut), "ref_original_cutaway")
    print("  SASTO-PA cutaway...")
    if pa_cut:
        save_render(render_mesh(pa_cut, cam_cut), "ref_sasto_pa_cutaway")


def render_thumbnails():
    """Small square-ish thumbnails for the Visual Abstract."""
    print("\n=== Thumbnail Renders ===")
    part, occ_orig, occ_v11, occ_v12 = load_ref()

    orig = build_colored_mesh(occ_orig, part)
    pa = build_colored_mesh(occ_v11, part)
    pa_cut = build_cutaway(occ_v11, part)

    poses, _ = compute_camera_poses(orig)

    # Smaller render for thumbnails
    cam = poses['isometric']
    cam_cut = poses['cutaway_front']

    print("  Thumbnail: original...")
    save_render(render_mesh(orig, cam, w=800, h=600), "thumb_original", max_w=600, max_h=450)
    print("  Thumbnail: optimized...")
    save_render(render_mesh(pa, cam, w=800, h=600), "thumb_optimized", max_w=600, max_h=450)
    print("  Thumbnail: cutaway...")
    if pa_cut:
        save_render(render_mesh(pa_cut, cam_cut, w=800, h=600), "thumb_cutaway", max_w=600, max_h=450)


def render_sample_designs():
    """Render a few batch-optimized sample designs (isometric only)."""
    print("\n=== Sample Design Renders ===")
    sample_ids = ["04203", "08018", "05728", "01440"]

    for sid in sample_ids:
        print(f"\n  Sample {sid}...")
        base_occ, opt_occ, part = load_sample(sid)
        if base_occ is None:
            print(f"    SKIP (no data)")
            continue

        orig_mesh = build_colored_mesh(base_occ, part)
        if orig_mesh is None:
            print(f"    SKIP (meshing failed)")
            continue

        poses, _ = compute_camera_poses(orig_mesh)
        cam = poses['isometric']

        print(f"    Original...")
        save_render(render_mesh(orig_mesh, cam), f"sample_{sid}_original")

        if opt_occ is not None:
            opt_mesh = build_colored_mesh(opt_occ, part)
            if opt_mesh:
                print(f"    Optimized...")
                save_render(render_mesh(opt_mesh, cam), f"sample_{sid}_optimized")

            opt_cut = build_cutaway(opt_occ, part)
            if opt_cut:
                print(f"    Cutaway...")
                save_render(render_mesh(opt_cut, cam), f"sample_{sid}_cutaway")


def main():
    print("=" * 60)
    print("Generating HQ poster renders (using render_figures.py)")
    print("=" * 60)

    render_ref_case()
    render_thumbnails()
    render_sample_designs()

    print("\n" + "=" * 60)
    n = len(list(OUT.glob("*.png")))
    print(f"Done. {n} HQ renders saved to poster_final/renders_hq/")
    print("=" * 60)


if __name__ == "__main__":
    main()
