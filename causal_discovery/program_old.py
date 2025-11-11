import json
import causal_discovery.wrappers as wrappers
from causal_discovery.wrappers.utils.tab_utils import remove_highly_correlated_features, add_correlated_nodes_to_graph, restore_original_node_indices
from causal_discovery.hierarchical_constraints import apply_hierarchical_constraints_to_data
from utils.logger import logger


class Programming(object):
    def __init__(self, args):
        self.args = args

    def forward(self, global_state):
        # ========== ★ 新增：自动检测并创建背景知识约束 ★ ==========
        background_knowledge = apply_hierarchical_constraints_to_data(
            global_state.user_data.processed_data
        )
        # ===========================================================
        
        # Check if we should automatically find and handle correlated features
        if global_state.algorithm.handle_correlated_features:
            threshold = getattr(global_state.algorithm, 'correlation_threshold', 0.99)
            # Automatically find and remove highly correlated features
            reduced_data, adjusted_mapping, original_indices = remove_highly_correlated_features(
                global_state.user_data.processed_data, 
                threshold=threshold
            )
                        
            # Only proceed with reduced dataset if we found correlated features
            if len(original_indices) < global_state.user_data.processed_data.shape[1]:
                # Run algorithm on reduced dataset
                algo_func = getattr(wrappers, global_state.algorithm.selected_algorithm)
                
                # ★ 修改1：传入background_knowledge ★
                try:
                    graph, info, raw_result = algo_func(global_state.algorithm.algorithm_arguments).fit(
                        reduced_data, 
                        background_knowledge=background_knowledge
                    )
                except TypeError:
                    # 算法不支持background_knowledge参数，回退到无约束版本
                    logger.warning(f"{global_state.algorithm.selected_algorithm} does not support background_knowledge parameter")
                    graph, info, raw_result = algo_func(global_state.algorithm.algorithm_arguments).fit(reduced_data)
                # ==========================================
                
                # Restore original indices in the mapping if needed
                restored_graph, restored_mapping = restore_original_node_indices(
                    graph, original_indices, adjusted_mapping
                )

                # Add back the highly correlated features to the graph
                final_graph = add_correlated_nodes_to_graph(
                    restored_graph, 
                    data=global_state.user_data.processed_data,
                    threshold=threshold,
                    original_indices=original_indices
                )
                
                # Store original and expanded results
                global_state.results.raw_result = raw_result
                global_state.results.converted_graph = final_graph
                info['original_graph'] = graph  # Store the original graph before adding correlated nodes
                info['high_corr_features_removed'] = original_indices
            else:
                # No correlated features found, run algorithm on the full dataset
                algo_func = getattr(wrappers, global_state.algorithm.selected_algorithm)
                
                # ★ 修改2：传入background_knowledge ★
                try:
                    graph, info, raw_result = algo_func(global_state.algorithm.algorithm_arguments).fit(
                        global_state.user_data.processed_data,
                        background_knowledge=background_knowledge
                    )
                except TypeError:
                    logger.warning(f"{global_state.algorithm.selected_algorithm} does not support background_knowledge parameter")
                    graph, info, raw_result = algo_func(global_state.algorithm.algorithm_arguments).fit(
                        global_state.user_data.processed_data
                    )
                # ==========================================
                
                global_state.results.raw_result = raw_result
                global_state.results.converted_graph = graph
        else:
            # Run algorithm on the full dataset
            algo_func = getattr(wrappers, global_state.algorithm.selected_algorithm)
            
            # ★ 修改3：传入background_knowledge ★
            try:
                graph, info, raw_result = algo_func(global_state.algorithm.algorithm_arguments).fit(
                    global_state.user_data.processed_data,
                    background_knowledge=background_knowledge
                )
            except TypeError:
                logger.warning(f"{global_state.algorithm.selected_algorithm} does not support background_knowledge parameter")
                graph, info, raw_result = algo_func(global_state.algorithm.algorithm_arguments).fit(
                    global_state.user_data.processed_data
                )
            # ==========================================
            
            global_state.results.raw_result = raw_result
            global_state.results.converted_graph = graph
            
        # Handle time-series specific data
        if global_state.statistics.time_series:
            if 'lag_matrix' in info:
                # Store the original lag matrix
                original_lag_matrix = info['lag_matrix']
                global_state.results.lagged_graph = original_lag_matrix
                
                # If we have correlated features, add them to the lag graph as well
                if global_state.algorithm.handle_correlated_features:
                    threshold = getattr(global_state.algorithm, 'correlation_threshold', 0.99)
                    # Add correlated nodes to the lag graph
                    enhanced_lag_matrix = add_correlated_nodes_to_graph(
                        original_lag_matrix,
                        data=global_state.user_data.processed_data,
                        threshold=threshold,
                        original_indices=original_indices
                    )
                    global_state.results.lagged_graph = enhanced_lag_matrix
                    info['lag_matrix'] = enhanced_lag_matrix
            else:
                global_state.results.lagged_graph = None

        global_state.results.raw_info = info
        
        # ========== ★ 新增：记录约束应用的统计信息 ★ ==========
        if background_knowledge is not None:
            self._log_constraint_statistics(global_state)
        # =====================================================
       
        return global_state
    
    def _log_constraint_statistics(self, global_state):
        """记录约束应用后的统计信息"""
        from causal_discovery.hierarchical_constraints import detect_hierarchical_structure
        
        data = global_state.user_data.processed_data
        graph = global_state.results.converted_graph
        
        is_hierarchical, tier0, tier1, tier2 = detect_hierarchical_structure(data.columns)
        
        if not is_hierarchical:
            return
        
        # 创建索引映射
        name_to_idx = {name: idx for idx, name in enumerate(data.columns)}
        tier0_idx = [name_to_idx[v] for v in tier0 if v in name_to_idx]
        tier1_idx = [name_to_idx[v] for v in tier1 if v in name_to_idx]
        tier2_idx = [name_to_idx[v] for v in tier2 if v in name_to_idx]
        
        # 统计边数
        import numpy as np
        edges_t0_t1 = sum(graph[i, j] != 0 for i in tier1_idx for j in tier0_idx)
        edges_t1_t2 = sum(graph[i, j] != 0 for i in tier2_idx for j in tier1_idx)
        edges_t0_t2 = sum(graph[i, j] != 0 for i in tier2_idx for j in tier0_idx)
        
        # 检查是否有违反约束的边（理论上应该为0）
        violations_t0 = sum(graph[i, j] != 0 for i in tier0_idx for j in tier0_idx)
        violations_t1 = sum(graph[i, j] != 0 for i in tier1_idx for j in tier1_idx)
        violations_t2 = sum(graph[i, j] != 0 for i in tier2_idx for j in tier2_idx)
        violations_reverse = sum(
            graph[j, i] != 0 
            for i in tier0_idx 
            for j in tier1_idx + tier2_idx
        )
        
        logger.success("✅ Causal structure discovered with constraints:")
        logger.detail(f"   Tier 0 (Parameters): {len(tier0)} nodes")
        logger.detail(f"   Tier 1 (Components): {len(tier1)} nodes")
        logger.detail(f"   Tier 2 (Metrics): {len(tier2)} nodes")
        logger.detail(f"")
        logger.detail(f"   Edges Tier 0 → Tier 1: {edges_t0_t1}")
        logger.detail(f"   Edges Tier 1 → Tier 2: {edges_t1_t2}")
        logger.detail(f"   Edges Tier 0 → Tier 2: {edges_t0_t2} (direct)")
        
        # 检查约束违规（应该全为0）
        total_violations = violations_t0 + violations_t1 + violations_t2 + violations_reverse
        if total_violations > 0:
            logger.warning(f"⚠️  Found {total_violations} constraint violations:")
            if violations_t0 > 0:
                logger.warning(f"   - Tier 0 internal edges: {violations_t0}")
            if violations_t1 > 0:
                logger.warning(f"   - Tier 1 internal edges: {violations_t1}")
            if violations_t2 > 0:
                logger.warning(f"   - Tier 2 internal edges: {violations_t2}")
            if violations_reverse > 0:
                logger.warning(f"   - Reverse causality edges: {violations_reverse}")
        else:
            logger.success("   ✓ All constraints satisfied!")