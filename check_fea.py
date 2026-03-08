import json
fea = json.load(open('fea_ml/runs/v3/fea_validation_full.json'))
print(f'Type: {type(fea).__name__}, Length: {len(fea)}')
if isinstance(fea, list) and fea:
    print('First item keys:', list(fea[0].keys()))
    print('First item:', {k: fea[0][k] for k in list(fea[0].keys())[:10]})
    # Check compliance ratios
    comp_ratios = [x.get('compliance_ratio', x.get('comp_ratio')) for x in fea if x.get('compliance_ratio') or x.get('comp_ratio')]
    print(f'Compliance ratios found: {len(comp_ratios)}')
    if comp_ratios:
        import numpy as np
        arr = np.array(comp_ratios)
        print(f'  Mean: {arr.mean():.3f}, Max: {arr.max():.3f}, Min: {arr.min():.3f}')
