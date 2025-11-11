"""
Hierarchical Constraints for CPU Design Space Exploration
自动检测并应用三层因果约束
"""

from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from utils.logger import logger


def detect_hierarchical_structure(columns):
    """
    自动检测数据是否包含三层结构
    
    返回: (is_hierarchical, tier0, tier1, tier2)
    """
    columns_lower = [col.lower() for col in columns]
    
    # 定义三层变量（小写，用于匹配）
    tier0_keywords = [
        'branchpredictor', 'fetchwidth', 'numfetchbufferentries', 'numrasentries',
        'maxbrcount', 'decodewidth', 'numrobentries', 'numintphysregisters',
        'memissuewidth', 'intissuewidth', 'numldqentries', 'enableprefetching',
        'enablesfbopt', 'numrxqentries', 'numrcqentries', 'nl2tlbentries',
        'nl2tlbways', 'nicacheways', 'nicachetlbways', 'ndcacheways',
        'ndcachemshrss', 'ndcachetlbways'
    ]
    
    tier1_keywords = [
        'power_leakage', 'power_internal', 'power_switching', 'power_memory',
        'power_register', 'power_logic', 'power_clock', 'power_pad',
        'area_cell_count', 'area_cell_area', 'area_net_area'
    ]
    
    tier2_keywords = ['cpi', 'power', 'area', 'time']
    
    # 匹配实际列名
    tier0 = []
    tier1 = []
    tier2 = []
    
    for col in columns:
        col_lower = col.lower()
        
        # 检查Tier 0（需要精确或模糊匹配）
        for keyword in tier0_keywords:
            if col_lower == keyword or col_lower.replace('_', '') == keyword.replace('_', ''):
                tier0.append(col)
                break
        
        # 检查Tier 1（power_*和area_*）
        if any(col_lower.startswith(prefix) for prefix in ['power_', 'area_']):
            if col_lower != 'power' and col_lower != 'area':  # 排除聚合变量
                tier1.append(col)
                continue
        
        # 检查Tier 2
        if col_lower in tier2_keywords:
            tier2.append(col)
    
    # 判断是否是三层结构（至少有一些tier0和tier2变量）
    is_hierarchical = len(tier0) >= 5 and len(tier2) >= 2
    
    if is_hierarchical:
        logger.detail(f"Detected hierarchical structure:")
        logger.detail(f"  Tier 0 (Parameters): {len(tier0)} variables")
        logger.detail(f"  Tier 1 (Components): {len(tier1)} variables")
        logger.detail(f"  Tier 2 (Metrics): {len(tier2)} variables")
    
    return is_hierarchical, tier0, tier1, tier2


def create_hierarchical_constraints(tier0, tier1, tier2):
    """
    创建三层约束的BackgroundKnowledge对象
    
    参数:
        tier0: Tier 0变量列表（设计参数）
        tier1: Tier 1变量列表（中间组件）
        tier2: Tier 2变量列表（性能指标）
    
    返回:
        BackgroundKnowledge对象
    """
    bk = BackgroundKnowledge()
    
    logger.process("Creating hierarchical causal constraints")
    
    # ========== 1. 禁止Tier 0内部的边 ==========
    logger.detail("Constraint 1: Forbidding edges within Tier 0 (parameters)")
    count = 0
    for i in tier0:
        for j in tier0:
            if i != j:
                bk.add_forbidden_by_pattern(i, j)
                count += 1
    logger.detail(f"  Added {count} forbidden edges within Tier 0")
    
    # ========== 2. 禁止Tier 1内部的边 ==========
    logger.detail("Constraint 2: Forbidding edges within Tier 1 (components)")
    count = 0
    for i in tier1:
        for j in tier1:
            if i != j:
                bk.add_forbidden_by_pattern(i, j)
                count += 1
    logger.detail(f"  Added {count} forbidden edges within Tier 1")
    
    # ========== 3. 禁止Tier 2内部的边 ==========
    logger.detail("Constraint 3: Forbidding edges within Tier 2 (metrics)")
    count = 0
    for i in tier2:
        for j in tier2:
            if i != j:
                bk.add_forbidden_by_pattern(i, j)
                count += 1
    logger.detail(f"  Added {count} forbidden edges within Tier 2")
    
    # ========== 4. 禁止反向因果（Tier 1 → Tier 0）==========
    logger.detail("Constraint 4: Forbidding reverse causality (Tier 1 → Tier 0)")
    count = 0
    for t1 in tier1:
        for t0 in tier0:
            bk.add_forbidden_by_pattern(t1, t0)
            count += 1
    logger.detail(f"  Added {count} forbidden reverse edges")
    
    # ========== 5. 禁止反向因果（Tier 2 → Tier 0）==========
    logger.detail("Constraint 5: Forbidding reverse causality (Tier 2 → Tier 0)")
    count = 0
    for t2 in tier2:
        for t0 in tier0:
            bk.add_forbidden_by_pattern(t2, t0)
            count += 1
    logger.detail(f"  Added {count} forbidden reverse edges")
    
    # ========== 6. 禁止反向因果（Tier 2 → Tier 1）==========
    logger.detail("Constraint 6: Forbidding reverse causality (Tier 2 → Tier 1)")
    count = 0
    for t2 in tier2:
        for t1 in tier1:
            bk.add_forbidden_by_pattern(t2, t1)
            count += 1
    logger.detail(f"  Added {count} forbidden reverse edges")
    
    # ========== 7. 强制聚合关系（Tier 1 → Tier 2）==========
    logger.detail("Constraint 7: Requiring aggregation (Tier 1 → Tier 2)")
    count = 0
    
    # power_* → power
    power_components = [v for v in tier1 if v.lower().startswith('power_')]
    if 'power' in [v.lower() for v in tier2]:
        power_var = next(v for v in tier2 if v.lower() == 'power')
        for p in power_components:
            bk.add_required_by_pattern(p, power_var)
            count += 1
        logger.detail(f"  Required {len(power_components)} power components → power")
    
    # area_* → area (排除area_cell_count)
    area_components = [v for v in tier1 if v.lower().startswith('area_') and 
                      not v.lower() == 'area_cell_count']
    if 'area' in [v.lower() for v in tier2]:
        area_var = next(v for v in tier2 if v.lower() == 'area')
        for a in area_components:
            bk.add_required_by_pattern(a, area_var)
            count += 1
        logger.detail(f"  Required {len(area_components)} area components → area")
    
    logger.success(f"✅ Created hierarchical constraints:")
    logger.detail(f"   Total forbidden edges: {len(tier0)*(len(tier0)-1) + len(tier1)*(len(tier1)-1) + len(tier2)*(len(tier2)-1) + len(tier1)*len(tier0) + len(tier2)*len(tier0) + len(tier2)*len(tier1)}")
    logger.detail(f"   Total required edges: {count}")
    
    return bk


def apply_hierarchical_constraints_to_data(data):
    """
    检测数据并创建约束（如果适用）
    
    参数:
        data: pandas DataFrame
    
    返回:
        BackgroundKnowledge对象或None
    """
    is_hierarchical, tier0, tier1, tier2 = detect_hierarchical_structure(data.columns)
    
    if is_hierarchical:
        logger.info("📊 Hierarchical structure detected - applying causal constraints")
        bk = create_hierarchical_constraints(tier0, tier1, tier2)
        return bk
    else:
        logger.info("No hierarchical structure detected - running without constraints")
        return None