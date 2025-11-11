import json
import numpy as np
import pandas as pd
import os

def export_causal_graphs(global_state, output_dir=None):
    """
    导出原始(converted)和优化后(revised)的因果图为JSON和CSV格式
    
    参数:
        global_state: 全局状态对象
        output_dir: 输出目录,默认使用 global_state.user_data.output_graph_dir
    """
    if output_dir is None:
        output_dir = global_state.user_data.output_graph_dir
    
    # 获取变量名
    columns = global_state.user_data.processed_data.columns.tolist()
    
    # 1. 导出原始因果图 (converted_graph)
    converted_graph = global_state.results.converted_graph
    export_single_graph(converted_graph, columns, output_dir, prefix="original")
    
    # 2. 导出优化后的因果图 (revised_graph) - 如果存在
    if hasattr(global_state.results, 'revised_graph') and global_state.results.revised_graph is not None:
        revised_graph = global_state.results.revised_graph
        export_single_graph(revised_graph, columns, output_dir, prefix="revised")
        print(f"✅ Revised graph exported")
    
    # 3. 导出Bootstrap概率矩阵 - 如果存在
    if hasattr(global_state.results, 'bootstrap_probability') and global_state.results.bootstrap_probability is not None:
        bootstrap_prob = global_state.results.bootstrap_probability
        np.save(os.path.join(output_dir, 'bootstrap_probability.npy'), bootstrap_prob)
        print(f"✅ Bootstrap probability saved to: bootstrap_probability.npy")
    
    print(f"\n📁 All files saved to: {output_dir}")


def export_single_graph(graph, columns, output_dir, prefix="original"):
    """
    导出单个因果图为多种格式
    
    参数:
        graph: 邻接矩阵 (np.ndarray)
        columns: 变量名列表
        output_dir: 输出目录
        prefix: 文件名前缀 (original/revised)
    """
    edge_types = {
        0: 'none',
        1: 'directed (->)',
        2: 'undirected (-)',
        3: 'bidirected (<->)',
        4: 'half_directed (o->)',
        5: 'half_undirected (o-)',
        6: 'no_edge (o-o)',
        7: 'correlated (---)'
    }
    
    # === 格式1: 完整邻接矩阵 JSON ===
    adjacency_dict = {
        'variables': columns,
        'adjacency_matrix': graph.tolist(),
        'edge_types': edge_types,
        'description': 'Matrix[i,j] represents edge from j to i'
    }
    json_path = os.path.join(output_dir, f'{prefix}_adjacency_matrix.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(adjacency_dict, f, indent=2, ensure_ascii=False)
    print(f"✅ {prefix} adjacency matrix saved to: {prefix}_adjacency_matrix.json")
    
    # === 格式2: NumPy格式 (方便Python读取) ===
    npy_path = os.path.join(output_dir, f'{prefix}_adjacency_matrix.npy')
    np.save(npy_path, graph)
    print(f"✅ {prefix} numpy array saved to: {prefix}_adjacency_matrix.npy")
    
    # === 格式3: 边列表 JSON (更直观) ===
    edge_list = []
    for i in range(len(columns)):
        for j in range(len(columns)):
            if graph[i, j] != 0:  # 存在边
                edge_list.append({
                    'source': columns[j],      # j是源
                    'target': columns[i],      # i是目标
                    'type': edge_types[graph[i, j]],
                    'code': int(graph[i, j])
                })
    
    edge_list_dict = {
        'edges': edge_list,
        'edge_count': len(edge_list),
        'node_count': len(columns),
        'edge_types': edge_types
    }
    edge_json_path = os.path.join(output_dir, f'{prefix}_edge_list.json')
    with open(edge_json_path, 'w', encoding='utf-8') as f:
        json.dump(edge_list_dict, f, indent=2, ensure_ascii=False)
    print(f"✅ {prefix} edge list saved to: {prefix}_edge_list.json")
    
    # === 格式4: CSV格式 (Excel可直接打开) ===
    # 4.1 完整邻接矩阵CSV
    df_adj = pd.DataFrame(graph, index=columns, columns=columns)
    csv_path = os.path.join(output_dir, f'{prefix}_adjacency_matrix.csv')
    df_adj.to_csv(csv_path)
    print(f"✅ {prefix} adjacency CSV saved to: {prefix}_adjacency_matrix.csv")
    
    # 4.2 边列表CSV
    if edge_list:
        df_edges = pd.DataFrame(edge_list)
        edge_csv_path = os.path.join(output_dir, f'{prefix}_edge_list.csv')
        df_edges.to_csv(edge_csv_path, index=False)
        print(f"✅ {prefix} edge list CSV saved to: {prefix}_edge_list.csv")


def export_specific_relationships(global_state, source_vars, target_vars, output_dir=None):
    """
    导出特定变量之间的因果关系 (例如: 架构参数 -> 性能指标)
    
    参数:
        global_state: 全局状态
        source_vars: 源变量列表 (例如前22个架构参数)
        target_vars: 目标变量列表 (例如后4个性能指标)
        output_dir: 输出目录
    """
    if output_dir is None:
        output_dir = global_state.user_data.output_graph_dir
    
    columns = global_state.user_data.processed_data.columns
    converted_graph = global_state.results.converted_graph
    
    # 获取索引
    source_indices = [columns.get_loc(var) for var in source_vars if var in columns]
    target_indices = [columns.get_loc(var) for var in target_vars if var in columns]
    
    # 提取子图
    relationships = []
    for src_idx in source_indices:
        for tgt_idx in target_indices:
            if converted_graph[tgt_idx, src_idx] != 0:  # src -> tgt
                relationships.append({
                    'source': columns[src_idx],
                    'target': columns[tgt_idx],
                    'type': get_edge_type_name(converted_graph[tgt_idx, src_idx]),
                    'code': int(converted_graph[tgt_idx, src_idx]),
                    'version': 'original'
                })
    
    # 如果有revised版本,也加入
    if hasattr(global_state.results, 'revised_graph') and global_state.results.revised_graph is not None:
        revised_graph = global_state.results.revised_graph
        for src_idx in source_indices:
            for tgt_idx in target_indices:
                if revised_graph[tgt_idx, src_idx] != 0:
                    relationships.append({
                        'source': columns[src_idx],
                        'target': columns[tgt_idx],
                        'type': get_edge_type_name(revised_graph[tgt_idx, src_idx]),
                        'code': int(revised_graph[tgt_idx, src_idx]),
                        'version': 'revised'
                    })
    
    # 保存为CSV
    df = pd.DataFrame(relationships)
    csv_path = os.path.join(output_dir, 'specific_causal_relationships.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Specific relationships saved to: specific_causal_relationships.csv")
    print(f"📊 Found {len(relationships)} causal edges")
    
    # 保存为JSON
    json_path = os.path.join(output_dir, 'specific_causal_relationships.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'relationships': relationships}, f, indent=2, ensure_ascii=False)
    
    return df


def get_edge_type_name(code):
    """获取边类型的名称"""
    edge_types = {
        0: 'none',
        1: 'directed',
        2: 'undirected',
        3: 'bidirected',
        4: 'half_directed',
        5: 'half_undirected',
        6: 'no_edge',
        7: 'correlated'
    }
    return edge_types.get(code, 'unknown')


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 在 main.py 的最后添加这些调用:
    
    # 1. 导出所有因果图
    export_causal_graphs(global_state)
    
    # 2. 导出特定关系 (架构参数 -> 性能指标)
    # 方式1: 按列位置划分 (推荐-灵活)
    n_arch_params = 22  # 前22列是架构参数
    arch_params = global_state.user_data.processed_data.columns[:n_arch_params].tolist()
    metrics = global_state.user_data.processed_data.columns[n_arch_params:].tolist()
    
    # 方式2: 手动指定列名 (如果你确定列名)
    # metrics = ['CPI', 'flush', 'ICacheMiss', 'DCacheMiss']
    
    # 方式3: 根据关键字自动识别
    # metrics = [col for col in columns if any(keyword in col.lower() 
    #            for keyword in ['cpi', 'miss', 'flush', 'cache'])]
    
    export_specific_relationships(
        global_state, 
        source_vars=arch_params,
        target_vars=metrics
    )
    
    print(f"\n✅ Exported relationships:")
    print(f"   - Source variables (arch params): {len(arch_params)}")
    print(f"   - Target variables (metrics): {metrics}")