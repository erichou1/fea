# ISEF Poster / Trifold Summary

## Title
**Surrogate-Accelerated Topology Optimization of 3D-Printed Concrete Houses via Deep Ensemble Sensitivity Ranking and 6-Connectivity Preservation**

**Eric Hou | February 2026**

---

## LEFT PANEL: Background & Problem

### The Problem
- Residential construction → **11% of global CO₂ emissions**
- Concrete production alone → **8%** (IEA, 2021)
- Conventional walls are **uniform thickness** — structurally wasteful
- 3D-printed concrete enables arbitrary geometries **at no extra cost**

### The Gap
Topology optimization (finding where to place material) is **too slow** for buildings:
- Classical method (SIMP): **5–30 hours** per house
- Requires hundreds of finite element simulations

### Additional Problem
- Standard voxel topology checks (26-connectivity) produce **floating mesh fragments**
- Makes optimized designs **unprintable** ❌

---

## CENTER PANEL: Method & Results

### SASTO: Surrogate-Accelerated Sensitivity Topology Optimization

**Key Idea:** Replace expensive FEA simulations with a trained neural network surrogate

1. **Train** a 5-member deep ensemble CNN (43.8M params) on 11,178 house simulations
2. **Optimize** by ranking voxels with backpropagation gradients and removing the least important ones
3. **Guarantee** single-component mesh via 6-connectivity topology preservation

### Three Contributions
| # | Contribution | Impact |
|---|-------------|--------|
| 1 | SASTO algorithm | **100–700× speedup** over SIMP |
| 2 | 6-connectivity preservation | **Zero floating fragments** (vs thousands) |
| 3 | Part-aware thickness | **+10.7 pp** additional reduction |

### Key Results (Simulated)
| Metric | Value |
|--------|-------|
| Material reduction | **45.0%** |
| Runtime | **159.5 seconds** (consumer GPU) |
| Mesh components | **1** (single, printable) |
| Constraints violated | **0** |

### Where Material Was Removed
| Part | Material Retained |
|------|------------------|
| Exterior walls | 91% ← load-bearing |
| Interior walls | 13% ← mostly removed |
| Roof | 93% |
| Floor | 96% |

> The algorithm **automatically discovers** that interior partition walls are structurally redundant — without being told.

### Figures
- **Figure 4:** Convergence plot → `figures/fig4_convergence.png`
- **Figure 5:** Per-part breakdown → `figures/fig5_per_part.png`
- **Figure 6:** Efficiency comparison → `figures/fig6_efficiency.png`
- **Figure 9:** Ablation summary → `figures/fig9_ablation.png`
- **Figure 11:** Speedup comparison → `figures/fig11_speedup.png`

---

## RIGHT PANEL: Conclusion & Future Work

### What Was Validated ✓
- [x] 45% material reduction with all constraints satisfied
- [x] Single-component printable mesh guaranteed
- [x] 6-connectivity eliminates floating fragments (ablation proved)
- [x] Part-aware thickness adds 10.7 pp over uniform

### What Remains ✗
- [ ] Ground-truth FEA re-analysis of optimized designs
- [ ] Physical 3D-print test of scaled model
- [ ] Formal surrogate accuracy metrics (MAE, RMSE, R²)
- [ ] Multi-geometry generalization (≥ 10 floor plans)
- [ ] Seismic loading scenarios

### Novelty Claims
1. **First** surrogate-accelerated topology optimization at building scale
2. **First** identification of 26-connectivity → floating fragments failure mode in voxel TO
3. **New** Efficiency-Integrity Index (I_EI) for comparing optimization variants
4. **New** formal proof that 6-connectivity is sufficient for marching cubes mesh connectivity

### Impact
If validated through FEA re-analysis and physical testing, SASTO could enable:
- **Real-time design exploration** for 3D-printed houses
- **~45% concrete reduction** → proportional CO₂ reduction
- **Automated structural optimization** accessible on consumer hardware

### Code & Data
- GitHub: https://github.com/erichou1/fea.git
- Trained model: `checkpoints/final_model.pth`
- All optimization data: `fea_ml/runs/v3/optimization_128/`

---

*All results tagged [Simulated]. Physical validation is future work.*
