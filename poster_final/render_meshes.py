#!/usr/bin/env python3
"""
Render 3D house meshes from PLY files to publication-quality PNGs.
Uses trimesh + pyrender for offscreen rendering.

Renders:
  - REF original (colored, cutaway)
  - REF optimized PA (colored, cutaway, transparent)
  - Several sample designs (original + optimized)
  - Composites: before/after side-by-side

Output directory: poster_final/renders/
"""

import os
import sys
import numpy as np

import trimesh
import pyrender
from PIL import Image, ImageDraw, ImageFont

# Paths
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCREENSHOT_STLS = os.path.join(BASE, "figures", "screenshot_stls")
COLORED_STLS = os.path.join(BASE, "figures", "stl_exports_colored")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "renders")
os.makedirs(OUT, exist_ok=True)

# Poster palette
BG_COLOR = np.array([0xF7, 0xF9, 0xFC, 0xFF]) / 255.0  # CARD fill
NAVY_BG = np.array([0x06, 0x2B, 0x7A, 0xFF]) / 255.0
WHITE_BG = np.array([1.0, 1.0, 1.0, 1.0])

# Camera configs for different views
VIEWS = {
    "iso": {  # Standard isometric
        "eye": np.array([1.5, 1.2, 1.5]),
        "center": np.array([0.0, 0.0, 0.0]),
        "up": np.array([0.0, 1.0, 0.0]),
    },
    "front": {
        "eye": np.array([0.0, 0.5, 2.5]),
        "center": np.array([0.0, 0.0, 0.0]),
        "up": np.array([0.0, 1.0, 0.0]),
    },
    "top": {
        "eye": np.array([0.0, 3.0, 0.1]),
        "center": np.array([0.0, 0.0, 0.0]),
        "up": np.array([0.0, 0.0, -1.0]),
    },
    "three_quarter": {
        "eye": np.array([1.8, 1.0, 1.2]),
        "center": np.array([0.0, 0.0, 0.0]),
        "up": np.array([0.0, 1.0, 0.0]),
    },
    "three_quarter_r": {
        "eye": np.array([-1.8, 1.0, 1.2]),
        "center": np.array([0.0, 0.0, 0.0]),
        "up": np.array([0.0, 1.0, 0.0]),
    },
}


def load_colored_mesh(ply_path):
    """Load a PLY mesh with vertex colors."""
    mesh = trimesh.load(ply_path)
    return mesh


def center_and_normalize(mesh, scale=1.0):
    """Center mesh at origin and normalize to unit scale."""
    centroid = mesh.bounds.mean(axis=0)
    mesh.vertices -= centroid
    extent = mesh.bounds[1] - mesh.bounds[0]
    max_extent = extent.max()
    if max_extent > 0:
        mesh.vertices *= scale / max_extent
    return mesh


def trimesh_to_pyrender(mesh, alpha=1.0):
    """Convert trimesh to pyrender mesh, preserving vertex colors."""
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        colors = mesh.visual.vertex_colors.astype(np.float32) / 255.0
        if alpha < 1.0:
            colors[:, 3] = alpha
        # Create pyrender primitive with vertex colors
        material = pyrender.MetallicRoughnessMaterial(
            alphaMode='BLEND' if alpha < 1.0 else 'OPAQUE',
            baseColorFactor=[1.0, 1.0, 1.0, alpha],
            metallicFactor=0.1,
            roughnessFactor=0.7,
        )
        pr_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
    else:
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.7, 0.7, 0.7, alpha],
            metallicFactor=0.1,
            roughnessFactor=0.7,
        )
        pr_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
    return pr_mesh


def render_mesh(mesh, view_name="iso", width=1200, height=900,
                bg_color=None, alpha=1.0):
    """Render a trimesh mesh to an RGBA numpy array."""
    if bg_color is None:
        bg_color = BG_COLOR

    scene = pyrender.Scene(bg_color=bg_color, ambient_light=[0.3, 0.3, 0.3])

    # Add mesh
    pr_mesh = trimesh_to_pyrender(mesh, alpha=alpha)
    scene.add(pr_mesh)

    # Camera
    view = VIEWS[view_name]
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.5)

    # Calculate camera pose
    eye = view["eye"]
    center = view["center"]
    up = view["up"]

    # Build look-at matrix
    z = eye - center
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)

    cam_pose = np.eye(4)
    cam_pose[:3, 0] = x
    cam_pose[:3, 1] = y
    cam_pose[:3, 2] = z
    cam_pose[:3, 3] = eye

    scene.add(cam, pose=cam_pose)

    # Lights
    key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    key_pose = np.eye(4)
    key_pose[:3, 3] = [2, 3, 2]
    scene.add(key_light, pose=cam_pose)

    fill_light = pyrender.DirectionalLight(color=np.array([0.8, 0.85, 1.0]),
                                           intensity=1.5)
    fill_pose = np.eye(4)
    fill_pose[:3, 0] = [-1, 0, 0]
    fill_pose[:3, 1] = [0, 1, 0]
    fill_pose[:3, 2] = [0, 0, -1]
    fill_pose[:3, 3] = [-2, 1, -2]
    scene.add(fill_light, pose=fill_pose)

    # Render  
    try:
        r = pyrender.OffscreenRenderer(width, height)
        color, depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        r.delete()
        return color
    except Exception as e:
        print(f"  Render error: {e}")
        # Fallback: create a solid color image
        img = np.ones((height, width, 4), dtype=np.uint8)
        img[:, :, :3] = (np.array(bg_color[:3]) * 255).astype(np.uint8)
        img[:, :, 3] = 255
        return img


def render_and_save(ply_path, name, view="iso", width=1200, height=900,
                    bg_color=None):
    """Load PLY, render, and save to PNG."""
    if not os.path.isfile(ply_path):
        print(f"  SKIP (not found): {ply_path}")
        return None

    print(f"  Rendering {name} ({view})...")
    mesh = load_colored_mesh(ply_path)
    mesh = center_and_normalize(mesh, scale=1.8)
    img = render_mesh(mesh, view_name=view, width=width, height=height,
                      bg_color=bg_color)
    img_pil = Image.fromarray(img)
    out_path = os.path.join(OUT, f"{name}.png")
    img_pil.save(out_path)
    print(f"  -> {out_path}")
    return out_path


def make_comparison(img_left_path, img_right_path, output_name,
                    label_left="Original", label_right="SASTO-PA Optimized",
                    badge_text=None, badge_color=(0xD7, 0x26, 0x3D)):
    """Create side-by-side comparison image with labels."""
    if not img_left_path or not img_right_path:
        return None
    if not os.path.isfile(img_left_path) or not os.path.isfile(img_right_path):
        return None

    left = Image.open(img_left_path)
    right = Image.open(img_right_path)

    # Resize to same height
    h = max(left.height, right.height)
    if left.height != h:
        left = left.resize((int(left.width * h / left.height), h), Image.LANCZOS)
    if right.height != h:
        right = right.resize((int(right.width * h / right.height), h), Image.LANCZOS)

    gap = 40
    label_h = 50
    total_w = left.width + right.width + gap
    total_h = h + label_h

    bg = (0xF7, 0xF9, 0xFC, 0xFF)
    comp = Image.new("RGBA", (total_w, total_h), bg)
    comp.paste(left, (0, label_h))
    comp.paste(right, (left.width + gap, label_h))

    draw = ImageDraw.Draw(comp)

    # Try to get a good font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
        font_badge = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except:
        try:
            font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 28)
            font_badge = ImageFont.truetype("/Library/Fonts/Arial.ttf", 22)
        except:
            font = ImageFont.load_default()
            font_badge = font

    dark = (0x0B, 0x17, 0x36)
    draw.text((left.width // 2, 10), label_left,
              fill=dark, font=font, anchor="mt")
    draw.text((left.width + gap + right.width // 2, 10), label_right,
              fill=dark, font=font, anchor="mt")

    # Arrow in gap
    arrow_y = label_h + h // 2
    arrow_x = left.width + gap // 2
    draw.text((arrow_x, arrow_y), "→", fill=(0xCF, 0xA5, 0x35), font=font, anchor="mm")

    # Badge
    if badge_text:
        bx = left.width + gap + right.width // 2
        by = label_h + h - 50
        # Badge background
        bbox = font_badge.getbbox(badge_text)
        bw = bbox[2] - bbox[0] + 30
        bh_val = bbox[3] - bbox[1] + 16
        draw.rounded_rectangle(
            [bx - bw//2, by - bh_val//2, bx + bw//2, by + bh_val//2],
            radius=12, fill=badge_color + (230,))
        draw.text((bx, by), badge_text, fill=(255, 255, 255), font=font_badge,
                  anchor="mm")

    out_path = os.path.join(OUT, f"{output_name}.png")
    comp.save(out_path)
    print(f"  -> {out_path}")
    return out_path


def main():
    print("=" * 60)
    print("SASTO Poster — 3D Mesh Rendering")
    print("=" * 60)

    # ── Reference Case Renders ──
    print("\n[1] Reference Case renders...")
    # Try both directories
    for prefix, directory in [
        ("REF_original", SCREENSHOT_STLS),
        ("REF_SASTO_PA", SCREENSHOT_STLS),
        ("ref_original", COLORED_STLS),
        ("ref_v11_pa", COLORED_STLS),
    ]:
        for variant in ["colored", "cutaway"]:
            ply = os.path.join(directory, f"{prefix}_{variant}.ply")
            if os.path.isfile(ply):
                for view in ["iso", "three_quarter"]:
                    render_and_save(ply, f"{prefix}_{variant}_{view}", view=view)

    # ── Sample designs ──
    print("\n[2] Sample design renders...")
    for design_id in ["04203", "08018", "05728", "01440"]:
        for variant in ["original_colored", "optimized_colored", "optimized_cutaway"]:
            for directory in [SCREENSHOT_STLS, COLORED_STLS]:
                # Check both naming conventions
                for name_pattern in [
                    f"{design_id}_{variant}",
                    f"{design_id}_{variant.replace('_colored', '')}",
                ]:
                    ply = os.path.join(directory, f"{name_pattern}.ply")
                    if os.path.isfile(ply):
                        render_and_save(ply, f"{design_id}_{variant}_iso", view="iso")
                        break

    # ── Comparison composites ──
    print("\n[3] Comparison composites...")

    # REF case: original vs SASTO-PA
    ref_orig = None
    ref_opt = None
    for d in [SCREENSHOT_STLS, COLORED_STLS]:
        for prefix in ["REF_original", "ref_original"]:
            p = os.path.join(OUT, f"{prefix}_colored_iso.png")
            if os.path.isfile(p):
                ref_orig = p
                break
        for prefix in ["REF_SASTO_PA", "ref_v11_pa"]:
            p = os.path.join(OUT, f"{prefix}_colored_iso.png")
            if os.path.isfile(p):
                ref_opt = p
                break

    if ref_orig and ref_opt:
        make_comparison(ref_orig, ref_opt, "ref_comparison_iso",
                        "Original", "SASTO-PA Optimized",
                        "-45.0% material")

    # 04203 comparison
    p_orig = os.path.join(OUT, "04203_original_colored_iso.png")
    p_opt = os.path.join(OUT, "04203_optimized_colored_iso.png")
    if os.path.isfile(p_orig) and os.path.isfile(p_opt):
        make_comparison(p_orig, p_opt, "04203_comparison_iso",
                        "Original", "SASTO-PA Optimized",
                        "-45.0% reduction")

    # ── Small thumbnails for Visual Abstract ──
    print("\n[4] Visual Abstract thumbnails...")
    # Small renders for pipeline steps
    for prefix, directory in [
        ("REF_original", SCREENSHOT_STLS),
        ("ref_original", COLORED_STLS),
    ]:
        ply = os.path.join(directory, f"{prefix}_colored.ply")
        if os.path.isfile(ply):
            render_and_save(ply, "thumb_original_house",
                            view="three_quarter", width=600, height=450)
            break

    for prefix, directory in [
        ("REF_SASTO_PA", SCREENSHOT_STLS),
        ("ref_v11_pa", COLORED_STLS),
    ]:
        ply = os.path.join(directory, f"{prefix}_colored.ply")
        if os.path.isfile(ply):
            render_and_save(ply, "thumb_optimized_house",
                            view="three_quarter", width=600, height=450)
            break

    for prefix, directory in [
        ("REF_SASTO_PA", SCREENSHOT_STLS),
        ("ref_v11_pa", COLORED_STLS),
    ]:
        ply = os.path.join(directory, f"{prefix}_cutaway.ply")
        if os.path.isfile(ply):
            render_and_save(ply, "thumb_cutaway_house",
                            view="three_quarter", width=600, height=450)
            break

    print("\n" + "=" * 60)
    rendered = [f for f in os.listdir(OUT) if f.endswith(".png")]
    print(f"Done. {len(rendered)} images in {OUT}/")
    for f in sorted(rendered):
        print(f"  {f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
