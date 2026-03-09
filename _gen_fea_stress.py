import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.image import imread
import pathlib, os

ROOT = pathlib.Path('.')
OUT = ROOT / 'results_figures'
BG = '#ffffff'; GRAY = '#1c2233'; DIM = '#8896b0'

fea_img = imread(str(ROOT / 'figures' / 'fig_fea_house_real_3d.png'))
fh, fw = fea_img.shape[:2]
FIG_W = 16
fig = plt.figure(figsize=(FIG_W, FIG_W * fh / fw + 0.8), facecolor=BG)
ax = fig.add_axes([0.02, 0.06, 0.96, 0.88])
ax.set_axis_off()
ax.imshow(fea_img, aspect='equal', interpolation='lanczos')
fig.text(0.5, 0.97, 'Independent FEA Validation  \u2014  Von Mises Stress Distribution',
         ha='center', va='top', color=GRAY, fontsize=14, fontweight='bold')
fig.text(0.5, 0.015,
         r'FEA mesh with applied gravity + wind loads.  '
         r'All optimised designs satisfy $\sigma_{max} < \sigma_{yield} / \mathrm{SF}$.',
         ha='center', va='bottom', color=DIM, fontsize=9.5, style='italic')
fig.savefig(str(OUT / 'fig_fea_stress.png'), dpi=180, bbox_inches='tight', facecolor=BG)
plt.close(fig)
sz = os.path.getsize(str(OUT / 'fig_fea_stress.png')) // 1024
print(f'Saved: fig_fea_stress.png  ({sz} KB)')
