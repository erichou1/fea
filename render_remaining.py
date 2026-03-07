#!/usr/bin/env python
"""Render remaining figures one at a time."""
import time, sys
sys.stdout.reconfigure(line_buffering=True)
import render_figures as rf

figures = [
    ("fig_optimized_gallery.png", lambda: rf.generate_gallery("fig_optimized_gallery.png")),
    ("fig_diverse_stl_gallery.png", lambda: rf.generate_gallery("fig_diverse_stl_gallery.png", gallery_ids=["00137", "11357", "06149", "00857"])),
    ("fig_type_comparison.png", lambda: rf.generate_fig_type_comparison()),
    ("fig_cross_section_comparison.png", lambda: rf.generate_fig_cross_section_comparison()),
]

for name, func in figures:
    print(f"\n{'='*60}")
    print(f"Starting {name}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        func()
        print(f"  Completed {name} in {time.time()-t0:.0f}s")
    except Exception as e:
        print(f"  FAILED {name}: {e}")

print("\n\nALL DONE")
