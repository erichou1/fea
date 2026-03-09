"""
Render introduction before/after images from existing GLBs.
Outputs:
  poster_images_extracted/intro_original_cutaway.png
  poster_images_extracted/intro_optimized_cutaway.png
"""
import os, sys
import numpy as np
import trimesh
import pyrender
import PIL.Image

os.environ["PYOPENGL_PLATFORM"] = "osmesa"   # headless

OUT = "poster_images_extracted"
W, H = 900, 720

CAMERA_POSE = np.array([
    [ 0.707, -0.354,  0.612, 1.8],
    [ 0.707,  0.354, -0.612, 1.8],
    [ 0.000,  0.866,  0.500, 1.2],
    [ 0.000,  0.000,  0.000, 1.0],
])

def render_glb(glb_path, out_png, bg=(245, 248, 252)):
    scene_mesh = trimesh.load(glb_path, force="scene")

    # centre + normalise scale
    bounds = scene_mesh.bounds
    if bounds is None:
        # might be a scene with multiple meshes
        all_bounds = np.concatenate([
            m.bounds for m in scene_mesh.geometry.values()
            if hasattr(m, "bounds") and m.bounds is not None
        ])
        bounds = np.array([all_bounds.min(axis=0), all_bounds.max(axis=0)])

    center = bounds.mean(axis=0)
    extent = (bounds[1] - bounds[0]).max()

    # Build pyrender scene
    pr_scene = pyrender.Scene(bg_color=[*bg, 255],
                               ambient_light=[0.35, 0.35, 0.35])

    for name, geom in scene_mesh.geometry.items():
        if not isinstance(geom, trimesh.Trimesh):
            continue
        geom = geom.copy()
        geom.apply_translation(-center)
        geom.apply_scale(2.0 / extent)

        if hasattr(geom.visual, "to_color"):
            visual = geom.visual.to_color()
            vertex_colors = visual.vertex_colors
        else:
            vertex_colors = None

        mesh = pyrender.Mesh.from_trimesh(
            geom,
            smooth=False,
        )
        pr_scene.add(mesh)

    # Camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=W/H)
    # Isometric-ish view from front-right-above
    dist = 3.2
    elev, azim = np.radians(25), np.radians(-40)
    cx = dist * np.cos(elev) * np.sin(azim)
    cy = dist * np.cos(elev) * np.cos(azim)
    cz = dist * np.sin(elev)
    eye    = np.array([cx, cy, cz])
    target = np.array([0, 0, 0])
    up     = np.array([0, 0, 1])

    z_ax = (eye - target); z_ax /= np.linalg.norm(z_ax)
    x_ax = np.cross(up, z_ax); x_ax /= np.linalg.norm(x_ax)
    y_ax = np.cross(z_ax, x_ax)

    cam_pose = np.eye(4)
    cam_pose[:3, 0] = x_ax
    cam_pose[:3, 1] = y_ax
    cam_pose[:3, 2] = z_ax
    cam_pose[:3, 3] = eye
    pr_scene.add(camera, pose=cam_pose)

    # Lights
    for pos in ([2, 2, 4], [-3, 1, 3], [0, -3, 2]):
        light = pyrender.DirectionalLight(color=[1,1,1], intensity=3.5)
        lp = np.eye(4); lp[:3, 3] = pos
        pr_scene.add(light, pose=lp)

    renderer = pyrender.OffscreenRenderer(W, H)
    color, _ = renderer.render(pr_scene)
    renderer.delete()

    PIL.Image.fromarray(color).save(out_png)
    print(f"  ✓ {out_png}")


print("Rendering original cutaway…")
render_glb("figures/stl_exports_colored/REF_original_cutaway.glb",
           f"{OUT}/intro_original_cutaway.png")

print("Rendering SASTO-PA cutaway…")
render_glb("figures/stl_exports_colored/REF_SASTO_PA_cutaway.glb",
           f"{OUT}/intro_optimized_cutaway.png")

print("Done.")
