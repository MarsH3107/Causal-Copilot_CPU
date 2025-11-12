import sys
sys.path.insert(0, '/home/marsh/Documents/Project/Casual_Copilot/Causal-Copilot-main/externals/causal-learn')
import pandas as pd
from causal_discovery.hierarchical_constraints import detect_hierarchical_structure

# 读取数据
data = pd.read_csv("/home/marsh/Documents/Project/Casual_Copilot/DSE_Copilot/DAC_dataset/Arc_LHD_1000_caus_caption.csv")

# 检测层次结构
is_hierarchical, tier0, tier1, tier2 = detect_hierarchical_structure(data.columns)

print(f"是否为层次结构: {is_hierarchical}")
print(f"Tier 0 变量: {tier0}")
print(f"numFetchBufferEntries 在 Tier 0: {'numFetchBufferEntries' in tier0}")
print(f"fetchWidth 在 Tier 0: {'fetchWidth' in tier0}")