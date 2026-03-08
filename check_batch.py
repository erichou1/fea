import json, os
d = 'fea_ml/runs/v3/batch_results_all'
data = json.load(open(os.path.join(d, '00000', 'optimization_summary.json')))
print('Keys:', list(data.keys()))
for k,v in data.items():
    if k != 'history':
        print(f'  {k}: {v}')
print(f'history length: {len(data.get("history",[]))}')

count = 0
vol_reds = []
for folder in sorted(os.listdir(d)):
    try:
        s = json.load(open(os.path.join(d, folder, 'optimization_summary.json')))
        if s.get('success'):
            count += 1
            vol_reds.append(s['volume_reduction']*100)
    except:
        pass
import numpy as np
arr = np.array(vol_reds)
print(f'Total successful: {count}')
print(f'Mean: {arr.mean():.1f}%, Std: {arr.std():.1f}%, Max: {arr.max():.1f}%, Min: {arr.min():.1f}%')
