# SASTO ISEF Poster — Layout Guide & Content Map

## Poster Dimensions
- **Size:** 48 × 36 inches (landscape, tri-fold)
- **Panels:** Left 12" | Center 24" | Right 12"
- **Title band:** 3.25" tall across full width

---

## Visual Layout Diagram

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                              TITLE BAND (3.25")                                │
│  [House Renders]  SURROGATE-ACCELERATED STRUCTURAL OPTIMIZATION   [Credits]    │
│                   Additive Manufacturing: Harnessing FEA...                     │
│                   Eric Hou                                          [Gold line] │
├──────────────┬─────────────────────────────────────┬──────────────┤
│  LEFT (12")  │         CENTER (24")                │  RIGHT (12") │
│              │                                     │              │
│ ┌──────────┐ │ ┌─────────────────────────────────┐ │ ┌──────────┐ │
│ │ L1: VIS  │ │ │      C1: ENGINEERING             │ │ │ R1: STAT │ │
│ │ ABSTRACT │ │ │      METHODOLOGY                 │ │ │ ANALYSIS │ │
│ │ (4.00")  │ │ │      (16.50")                    │ │ │ (13.80") │ │
│ │          │ │ │                                   │ │ │          │ │
│ │ Fig.1    │ │ │ ┌──────────┬──────────┐          │ │ │ Surrogate│ │
│ │ Pipeline │ │ │ │ Dataset  │ Deep     │          │ │ │ Table    │ │
│ └──────────┘ │ │ │ Pipeline │ Ensemble │ (7.80") │ │ │          │ │
│ ┌──────────┐ │ │ │ Fig.6    │ Fig.7    │          │ │ │ Converg. │ │
│ │ L2: INTRO│ │ │ ├──────────┼──────────┤          │ │ │ Fig.14   │ │
│ │ (5.80")  │ │ │ │ SASTO    │ 6-Conn   │          │ │ │          │ │
│ │          │ │ │ │ Algorithm│ Guarantee│ (bot)   │ │ │ k-Factor │ │
│ │ Text  |  │ │ │ │ Fig.8    │ Fig.9    │          │ │ │ Fig.15   │ │
│ │       |F2│ │ │ └──────────┴──────────┘          │ │ │          │ │
│ └──────────┘ │ └─────────────────────────────────┘ │ │ Conformal│ │
│ ┌──────────┐ │ ┌─────────────────────────────────┐ │ │ Fig.16   │ │
│ │L3: OBJEC │ │ │ C2: RESULTS & IN-SILICO VALID.  │ │ └──────────┘ │
│ │ (2.50")  │ │ │ (remaining)                      │ │ ┌──────────┐ │
│ │ 4 goals  │ │ │                                   │ │ │ R2: CONC │ │
│ │ Pipeline │ │ │ ┌────────┬────────┬────────┐     │ │ │ (5.50")  │ │
│ └──────────┘ │ │ │Ref Case│Multi-  │Speedup │     │ │ │ 5 points │ │
│ ┌──────────┐ │ │ │00472   │Geometry│& FEA   │     │ │ │ Impact   │ │
│ │L4: DESIGN│ │ │ │Table 1 │Fig.10  │Fig.12  │     │ │ └──────────┘ │
│ │ CRITERIA │ │ │ │Renders │Fig.11  │Fig.13  │     │ │ ┌──────────┐ │
│ │ (3.50")  │ │ │ └────────┴────────┴────────┘     │ │ │ R3: FUT  │ │
│ │ Table    │ │ │ ┌─────────────────────────────┐   │ │ │ (3.80")  │ │
│ │ Eqs      │ │ │ │    STATS BANNER              │   │ │ │ 3 items  │ │
│ └──────────┘ │ │ │23.5% | 23-92× | 0/1114 |50s│   │ │ └──────────┘ │
│ ┌──────────┐ │ │ └─────────────────────────────┘   │ │ ┌──────────┐ │
│ │L5: PROB  │ │ └─────────────────────────────────┘ │ │ │ R4: REFS │ │
│ │ FRAMING  │ │                                     │ │ │ (fill)   │ │
│ │ (fill)   │ │                                     │ │ │ 10 refs  │ │
│ │ Obj/Sens/│ │                                     │ │ └──────────┘ │
│ │ Part-awr │ │                                     │              │
│ └──────────┘ │                                     │              │
└──────────────┴─────────────────────────────────────┴──────────────┘
```

---

## Color Palette

| Name | Hex | Usage |
|------|-----|-------|
| bg-navy | `#062B7A` | Background fill |
| title-band | `#032061` | Title band background |
| section-bar | `#0A3D9A` | Section headers |
| card-fill | `#F7F9FC` | Content card background |
| card-border | `#B7C5E3` | Card outlines |
| text-dark | `#0B1736` | Body text |
| accent-teal | `#008C9E` | Primary accent, SPEED/PRINT badges |
| accent-red | `#D7263D` | Alert accent, SAFETY badges |
| accent-gold | `#CFA535` | Title rule, headline numbers |
| eq-pill | `#E8EEF8` | Equation backgrounds |

---

## Font System

| Element | Font | Size | Weight |
|---------|------|------|--------|
| Title | Arial Black | 44pt (PPTX) / 62pt (LaTeX) | Bold |
| Subtitle | Arial | 26pt / 32pt | Bold Italic |
| Author | Arial | 22pt / 26pt | Bold |
| Section headers | Arial Black | 28pt / 24pt | Bold, ALL CAPS, white on section-bar |
| Sub-headers | Arial | 12pt / 16pt | Bold |
| Body text | Arial | 10pt / 13.5pt | Regular |
| Captions | Arial | 8pt / 11.5pt | Italic |
| Table body | Arial | 10pt | Regular |
| Equation pills | Arial | 10pt | Regular, centered |

---

## Figure Map (20 figures in poster_final/)

| ID | Filename | Section | Description |
|----|----------|---------|-------------|
| 1 | `fig01_visual_abstract_pipeline.png` | L1 Visual Abstract | End-to-end SASTO pipeline: wireframe→parts→FEA→surrogate→optimize→STL |
| 2 | `fig02_uniform_vs_optimized.png` | L2 Introduction | Side-by-side: uniform wall vs. SASTO-optimized |
| 3 | `fig03_dataset_pipeline.png` | C1-A Dataset | 14,293 wireframes → 11,178 FEA → 128³ voxel pipeline |
| 4 | `fig04_architecture.png` | C1-B Deep Ensemble | CNN architecture: 7-ch 128³ → conv stages → dual pool → 3 outputs |
| 5 | `fig05_sasto_flowchart.png` | C1-C SASTO Algorithm | Three-phase erosion flowchart |
| 6 | `fig06_connectivity.png` | C1-D 6-Connectivity | 26-conn FAILS vs 6-conn WORKS comparison |
| 7 | `fig07_histogram.png` | C2-B Multi-Geometry | Volume reduction distribution across 1,114 designs |
| 8 | `fig08_per_part.png` | C2-B Multi-Geometry | Per-part material retention bar chart |
| 9 | `fig09_speedup.png` | C2-C Speedup | SIMP vs SASTO timing comparison |
| 10 | `fig10_fea_compliance.png` | C2-C FEA Validation | Compliance ratio scatter (0/1,114 violations) |
| 11 | `fig11_convergence.png` | R1 Statistics | Convergence curves: SASTO-PA vs SASTO-U |
| 12 | `fig12_k_factor.png` | R1 Statistics | k-factor Pareto frontier |
| 13 | `fig13_uncertainty.png` | R1 Statistics | Uncertainty band evolution |
| 14 | `fig14_part_aware_thickness.png` | L5 Problem Framing | Part-aware thickness schematic |
| 15 | `fig15_surrogate_table.png` | R1 Statistics | Surrogate performance metrics table |
| 16 | `fig16_optimization_objective.png` | L5 Problem Framing | Optimization objective equation card |
| 17 | `fig17_sensitivity_formula.png` | L5 Problem Framing | Backprop sensitivity formula card |
| 18 | `fig18_design_criteria_table.png` | L4 Design Criteria | Engineering constraint table |
| 19 | `fig19_reference_table.png` | C2-A Reference Case | Reference case results comparison table |
| 20 | `fig20_stats_banner.png` | C2 Bottom | 4-stat key results banner |

---

## Section-by-Section Content

### TITLE BAND (0–3.25")
- **Left:** 2 house render thumbnails (from poster_images_extracted/)
- **Center:** Title (44pt), subtitle (26pt), author (22pt)
- **Right:** Credit line, ISEF category, image attribution
- **Bottom:** Gold accent rule

### LEFT PANEL (12")

#### L1: VISUAL ABSTRACT (4.00")
- Fig. 1 pipeline diagram (2.70" tall)
- Caption + 3-sentence description

#### L2: INTRODUCTION (5.80")
- 4 sub-sections: CO₂ Crisis, Additive Manufacturing, Computational Bottleneck, SASTO Contribution
- Fig. 2 on right side (uniform vs. optimized)
- Dense paragraphs matching reference poster text density

#### L3: RESEARCH OBJECTIVES (2.50")
- 4 numbered goals with colored badges: SPEED (teal), PRINTABILITY (teal), EFFICIENCY (gold), SAFETY (red)
- Pipeline flow: Data Generation → Surrogate Training → SASTO Optimize → Validate & Certify

#### L4: ENGINEERING DESIGN CRITERIA (3.50")
- Constraint table (6 rows): stress, compliance, displacement, wall thickness (ext/int), mesh integrity
- 2 equation pills: allowable stress formula, conservative ensemble bound
- Material properties text block

#### L5: PROBLEM FRAMING (fills remaining ~6")
- Optimization objective (Fig. 16)
- Sensitivity formula (Fig. 17)
- Part-aware thickness (Fig. 14)

### CENTER PANEL (24")

#### C1: ENGINEERING METHODOLOGY (16.50")
2×2 sub-card grid:
- **Top-left (7.80"):** Dataset pipeline (Fig. 3) + split table + filtering description
- **Top-right (7.80"):** Deep ensemble architecture (Fig. 4) + hyperparameter table + training details
- **Bottom-left:** SASTO algorithm (Fig. 5) + 3 phase descriptions + post-processing
- **Bottom-right:** 6-connectivity (Fig. 6) + proposition box + digital topology definition

#### C2: RESULTS & IN-SILICO VALIDATION (fills remaining ~15")
3-column sub-layout:
- **Col A:** Reference case renders + Table 1 + badge (-45.0%)
- **Col B:** Histogram (Fig. 7) + per-part (Fig. 8)
- **Col C:** Speedup (Fig. 9) + FEA validation (Fig. 10) + badges
- **Bottom banner:** 4 key stats in gold on section-bar blue

### RIGHT PANEL (12")

#### R1: STATISTICAL ANALYSIS (13.80")
- Surrogate table (Fig. 15) + explanation box
- Convergence (Fig. 11) + comparison text
- k-factor Pareto (Fig. 12) + operating point badge
- Conformal/UQ (Fig. 13) + calibration statistics

#### R2: CONCLUSIONS (5.50")
- 5 numbered conclusions with teal circles
- Impact strip (gold border): CO₂ reduction potential

#### R3: FUTURE WORK (3.80")
- 3 items: Active learning, nonlinear FEA, physical testing
- Protocol pipeline: Optimized STL → 3D Print → Compression + DIC

#### R4: KEY REFERENCES (fills remaining)
- 10 references, 9pt text

---

## Files Delivered

| File | Purpose |
|------|---------|
| `poster_final/generate_all_poster_figures.py` | Generates all 20 matplotlib figures |
| `poster_final/build_poster_v5.py` | Builds 48×36" PPTX with all figures |
| `poster_final/poster.tex` | LaTeX beamerposter template |
| `poster_final/SASTO_ISEF_Poster_v5.pptx` | Ready-to-edit PPTX output |
| `poster_final/fig01–fig20_*.png` | 20 generated figure PNGs |
| `poster_final/LAYOUT_GUIDE.md` | This document |

---

## Manual Steps Remaining

1. **3D Renders:** Replace "[Thumbnail Image]" placeholders in Fig. 1, 3, 4, 6 with actual house renders from `figures/screenshot_stls/` or re-render using Blender/three.js
2. **Reference Case Renders:** Insert before/after 3D house renders in C2-A from `figures/screenshot_stls/REF_*.glb`
3. **LaTeX Compilation:** `lualatex poster.tex` (requires TeX Live with beamerposter, tcolorbox, fontspec)
4. **PPTX Polish:** Open in PowerPoint, adjust text overflow, fine-tune image positioning
5. **Print:** Export to PDF at 300 DPI for 48×36" print
