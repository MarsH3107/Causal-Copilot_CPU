import json
import numpy as np
import pandas as pd

# 读取边列表JSON (最方便)
with open('original_edge_list.json', 'r') as f:
    data = json.load(f)
    edges = data['edges']
    
# 筛选指向CPI的所有边
cpi_causes = [e for e in edges if e['target'] == 'CPI' and e['type'] == 'directed (->)']
print(f"影响CPI的参数: {[e['source'] for e in cpi_causes]}")

# 读取NumPy格式 (最快)
graph = np.load('original_adjacency_matrix.npy')

# 读取CSV格式
df = pd.read_csv('specific_causal_relationships.csv')
# 对比original和revised的差异
diff = df[df['version'] == 'revised'].merge(
    df[df['version'] == 'original'], 
    on=['source', 'target'], 
    how='outer', 
    suffixes=('_revised', '_original')
)