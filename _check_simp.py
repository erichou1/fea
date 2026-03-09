import json
d = json.load(open('fea_ml/runs/v3/simp_benchmark.json'))
for x in d:
    print(f'{x["sample_id"]} {x["group"]:16} simp_red={x["volume_reduction_pct"]:.1f} sasto_red={x["sasto_reduction_pct"]:.1f} simp_t={x["total_time_s"]:.0f}s comp_r={x["comp_ratio"]:.3f} sasto_comp_r={x["sasto_comp_ratio"]:.3f}')
