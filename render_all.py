#!/usr/bin/env python
"""Render all figures one by one with progress tracking."""
import sys, time
sys.stdout.reconfigure(line_buffering=True)

import render_figures as rf

figures = [
    ("fig_model_comparison", rf.generate_fig_model_comparison),
    ("fig12_stl_comparison", rf.generate_fig12_stl_comparison),
    ("fig_wireframe_pipeline", rf.generate_fig_wireframe_pipeline),
    ("fig_optimized_gallery", lambda: rf.generate_gallery("fig_optimized_gallery.png")),
    ("fig_diverse_stl_gallery", lambda: rf.generate_gallery("fig_diverse_stl_gallery.png",
                                         gallery_ids=["00137", "11357", "06149", "00857"])),
    ("fig_type_comparison", rf.generate_fig_type_comparison),
    ("fig_cross_section_comparison", rf.generate_fig_cross_section_comparison),
    ("fig_failure_gallery", rf.generate_fig_failure_gallery),
]

print("=" * 60)
print("Rendering all figures")
print("=" * 60)

for name, func in figures:
    t0 = time.time()
    try:
        func()
        print(f"  >> {name} completed in {time.time()-t0:.0f}s")
    except Exception as e:
        import traceback
        print(f"  >> FAILED {name}: {e}")
        traceback.print_exc()

print("\n\nALL DONE")
