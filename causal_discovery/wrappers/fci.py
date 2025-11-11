import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# use the local causal-learn package
import sys

# use the local causal-learn package
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
causal_learn_dir = os.path.join(root_dir, 'externals', 'causal-learn')
if not os.path.exists(causal_learn_dir):
    raise FileNotFoundError(f"Local causal-learn directory not found: {causal_learn_dir}, please git clone the submodule of causal-learn")
algorithm_dir = os.path.join(root_dir, 'algorithm')
sys.path.append(root_dir)
sys.path.append(causal_learn_dir)

from causallearn.graph.GraphClass import CausalGraph
from causallearn.search.ConstraintBased.FCI import fci as cl_fci
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode

from causal_discovery.wrappers.base import CausalDiscoveryAlgorithm
from causal_discovery.evaluation.evaluator import GraphEvaluator

class FCI(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'alpha': 0.05,
            'indep_test': 'fisherz',
            'depth': 4, # -1,
            'max_path_length': -1,
            'verbose': False,
            'background_knowledge': None,
            'show_progress': False,
        }
        self._params.update(params)

    def _create_background_knowledge(self, background_knowledge_spec, node_names):
        """
        Create BackgroundKnowledge object from JSON specification.
        
        Args:
            background_knowledge_spec: Dictionary or string specifying background knowledge
            node_names: List of variable names
            
        Returns:
            BackgroundKnowledge object or None
        """
        if background_knowledge_spec is None:
            return None
            
        # Create nodes
        nodes = [GraphNode(name) for name in node_names]
        node_dict = {name: node for name, node in zip(node_names, nodes)}
        
        bk = BackgroundKnowledge()
        
        if isinstance(background_knowledge_spec, str):
            # Handle string specifications like "forbidden_edges", "required_edges", etc.
            return None
        elif isinstance(background_knowledge_spec, dict):
            # Handle dictionary specification
            if 'forbidden_edges' in background_knowledge_spec:
                for edge in background_knowledge_spec['forbidden_edges']:
                    var1, var2 = edge
                    if var1 in node_dict and var2 in node_dict:
                        bk.add_forbidden_by_node(node_dict[var1], node_dict[var2])
                        
            if 'required_edges' in background_knowledge_spec:
                for edge in background_knowledge_spec['required_edges']:
                    var1, var2 = edge
                    if var1 in node_dict and var2 in node_dict:
                        bk.add_required_by_node(node_dict[var1], node_dict[var2])
                        
            if 'tiers' in background_knowledge_spec:
                for var_name, tier_level in background_knowledge_spec['tiers'].items():
                    if var_name in node_dict:
                        bk.add_node_to_tier(node_dict[var_name], tier_level)
                        
            return bk
        
        return None

    @property
    def name(self):
        return "FCI"

    def get_params(self):
        return self._params

    def get_primary_params(self):
        self._primary_param_keys = ['alpha', 'indep_test', 'depth']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    def get_secondary_params(self):
        self._secondary_param_keys = ['max_path_length', 'verbose', 'background_knowledge', 'show_progress']
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}

    def fit_old(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict, Tuple[CausalGraph, List]]:
        # Check and remove domain_index if it exists
        if 'domain_index' in data.columns:
            data = data.drop(columns=['domain_index'])
            
        node_names = list(data.columns)
        data_values = data.values

        # Process background knowledge if provided
        secondary_params = self.get_secondary_params().copy()
        if secondary_params.get('background_knowledge') is not None:
            bk = self._create_background_knowledge(
                secondary_params['background_knowledge'], 
                node_names
            )
            secondary_params['background_knowledge'] = bk

        # Combine primary and secondary parameters
        all_params = {**self.get_primary_params(), **secondary_params, 'node_names': node_names}

        # Run FCI algorithm
        graph, edges = cl_fci(data_values, **all_params)

        # Convert the graph to adjacency matrix
        adj_matrix = self.convert_to_adjacency_matrix(graph)

        # Prepare additional information
        info = {
            'edges': edges,
            'graph': graph,
        }

        return adj_matrix, info, (graph, edges)


    """
    完整的FCI算法fit方法 - 包含后续处理
    """

    def fit(self, data: pd.DataFrame, background_knowledge=None) -> Tuple[np.ndarray, Dict, Tuple]:
        """
        执行FCI算法
        
        参数:
            data: pandas DataFrame
            background_knowledge: BackgroundKnowledge对象（可选）
        
        返回:
            (adj_matrix, info, (G, edges))
        """
        # Check and remove domain_index if it exists
        if 'domain_index' in data.columns:
            data = data.drop(columns=['domain_index'])
        
        node_names = list(data.columns)
        data_values = data.values
        
        # Get parameters
        secondary_params = self.get_secondary_params().copy()
        
        # ========== 处理background_knowledge ==========
        if background_knowledge is not None:
            # 外部传入的优先级更高
            secondary_params['background_knowledge'] = background_knowledge
            from utils.logger import logger
            logger.detail("Using externally provided background_knowledge for FCI")
        
        # Process background knowledge if provided
        if secondary_params.get('background_knowledge') is not None:
            from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
            bk_input = secondary_params['background_knowledge']
            
            if isinstance(bk_input, BackgroundKnowledge):
                bk = bk_input
            else:
                bk = self._create_background_knowledge(bk_input, node_names)
            
            secondary_params['background_knowledge'] = bk
        
        # ========== 调用FCI算法 ==========
        all_params = {**self.get_primary_params(), **secondary_params}
        
        # 从causallearn导入FCI
        from causallearn.search.ConstraintBased.FCI import fci as cl_fci
        
        # FCI返回(G, edges)，其中G是GeneralGraph对象
        G, edges = cl_fci(data_values, **all_params)
        
        # ========== 后续处理：转换为邻接矩阵 ==========
        # FCI的G是PAG (Partial Ancestral Graph)，需要转换为邻接矩阵
        adj_matrix = self.convert_fci_to_adjacency_matrix(G)
        
        # ========== 准备额外信息 ==========
        info = {
            'node_names': node_names,
            'edges': edges,  # FCI发现的边列表
            'graph_type': 'PAG',  # Partial Ancestral Graph
            'num_nodes': len(node_names),
            'num_edges': len(edges) if edges else 0,
            # FCI特有的信息
            'sepset': G.sepset if hasattr(G, 'sepset') else None,
            'max_path_length': all_params.get('max_path_length', -1),
        }
        
        return adj_matrix, info, (G, edges)


    def convert_fci_to_adjacency_matrix(self, G) -> np.ndarray:
        """
        将FCI的PAG (GeneralGraph)转换为邻接矩阵
        
        FCI的边类型编码：
        - 0: no edge (o o)
        - 1: directed edge (→)
        - 2: undirected edge (-)
        - 3: bidirected edge (↔)
        - 4: circle-arrow (o→)
        - 5: circle-circle (o-o)
        - 6: circle-dash (o-)
        
        返回:
            邻接矩阵 (i,j)=1 表示 j→i 或相关边
        """
        import numpy as np
        
        # 获取图的邻接矩阵
        if hasattr(G, 'graph'):
            # GeneralGraph对象
            graph_matrix = G.graph
        else:
            # 直接是numpy数组
            graph_matrix = G
        
        n_nodes = graph_matrix.shape[0]
        adj_matrix = np.zeros((n_nodes, n_nodes), dtype=int)
        
        # 转换FCI的边编码为标准邻接矩阵
        for i in range(n_nodes):
            for j in range(n_nodes):
                edge_type = graph_matrix[i, j]
                
                if edge_type == 1:  # j → i (directed)
                    adj_matrix[i, j] = 1
                elif edge_type == 2:  # j - i (undirected)
                    adj_matrix[i, j] = 2
                    adj_matrix[j, i] = 2
                elif edge_type == 3:  # j ↔ i (bidirected)
                    adj_matrix[i, j] = 3
                    adj_matrix[j, i] = 3
                elif edge_type == 4:  # j o→ i (circle-arrow)
                    adj_matrix[i, j] = 4
                elif edge_type == 5:  # j o-o i (circle-circle)
                    adj_matrix[i, j] = 6
                    adj_matrix[j, i] = 6
                elif edge_type == 6:  # j o- i (circle-dash)
                    adj_matrix[i, j] = 5
        
        return adj_matrix


    def convert_to_adjacency_matrix(self, adj_matrix: CausalGraph) -> np.ndarray:
        adj_matrix = adj_matrix.graph
        inferred_flat = np.zeros_like(adj_matrix)
        indices = np.where(adj_matrix == 1)
        for i, j in zip(indices[0], indices[1]):
            if adj_matrix[j, i] == -1:
                # directed edge: j -> i
                inferred_flat[i, j] = 1
            elif adj_matrix[j, i] == 2:
                # bidirected edge: j o-> i
                inferred_flat[i, j] = 4
            elif adj_matrix[j, i] == 1:
                # bidirected edge: j <-> i
                if inferred_flat[j, i] == 0:
                    # keep asymmetric that only one entry is recorded
                    inferred_flat[i, j] = 3

        indices = np.where(adj_matrix == 2)
        for i, j in zip(indices[0], indices[1]):
            if adj_matrix[j, i] == 2:
                # undirected edge: j o-o i
                if inferred_flat[j, i] == 0:
                    inferred_flat[i, j] = 6
        return inferred_flat
    def test_algorithm(self):
        # Fix all random seeds for reproducibility
        np.random.seed(42)
        
        # Set random seeds for other libraries if they're being used
        import random
        random.seed(42)
        
        try:
            import torch
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
                torch.cuda.manual_seed_all(42)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
            
        # Set TensorFlow seed if it's being used
        try:
            import tensorflow as tf
            tf.random.set_seed(42)
        except ImportError:
            pass
        
        import time
        start_time = time.time()
        
        # Test hypothesis: larger node size needs larger sample size
        # We'll test different combinations of node sizes and sample sizes
        
        def degree2prob(degree, node_size):
            return degree / (node_size-1)
        
        node_sizes = [5]  # [5, 10, 15, 20, 25]
        sample_sizes = [1000]  # [500, 1000, 1500, 2000]
        num_runs = 1  # Number of runs to average results
        edge_probability = degree2prob(2, node_sizes[0])
        
        # Define different parameter configurations to compare
        configurations = [
            {"name": "Fixed Alpha 0.05", "alpha": 0.05, "indep_test": 'fisherz'},
        ]
        
        results = {}
        
        print("Testing hypothesis: larger node size needs larger sample size")
        print("=" * 80)
        print(f"Running {num_runs} iterations for each configuration")
        print("=" * 80)
        
        for config in configurations:
            config_name = config["name"]
            results[config_name] = {}
            
            print(f"\nTesting configuration: {config_name}")
            print("-" * 60)
            
            for n_nodes in node_sizes:
                results[config_name][n_nodes] = {}
                for n_samples in sample_sizes:
                    metrics_list = []
                    time_list = []
                    
                    print(f"\nTesting with {n_nodes} nodes and {n_samples} samples:")
                    
                    for run in range(num_runs):
                        print(f"  Run {run+1}/{num_runs}...")
                        
                        # Create a DataSimulator instance with new random seed for each run
                        seed = 42 + run
                        np.random.seed(seed)
                        simulator = DataSimulator()
                        
                        # Generate data
                        gt_graph, df = simulator.generate_dataset(
                            n_samples=n_samples, 
                            n_nodes=n_nodes, 
                            noise_type='gaussian',
                            function_type='linear', 
                            edge_probability=edge_probability,
                            n_domains=1
                        )
                        
                        # Configure FCI algorithm based on current configuration
                        self._params['alpha'] = config["alpha"]
                        self._params['indep_test'] = config["indep_test"]
                        self._params['show_progress'] = False
                        
                        run_start_time = time.time()
                        adj_matrix, info, _ = self.fit(df)
                        run_time = time.time() - run_start_time
                        
                        # Evaluate results
                        evaluator = GraphEvaluator()
                        metrics = evaluator.compute_metrics(gt_graph, adj_matrix)
                        
                        # Store results
                        metrics_list.append(metrics)
                        time_list.append(run_time)
                    
                    # Calculate average metrics
                    avg_metrics = {
                        'f1': np.mean([m['f1'] for m in metrics_list]),
                        'precision': np.mean([m['precision'] for m in metrics_list]),
                        'recall': np.mean([m['recall'] for m in metrics_list]),
                        'shd': np.mean([m['shd'] for m in metrics_list]),
                        'time': np.mean(time_list)
                    }
                    
                    results[config_name][n_nodes][n_samples] = avg_metrics
                    
                    # Print average results for this configuration
                    print(f"  Results for {n_nodes} nodes, {n_samples} samples (averaged over {num_runs} runs):")
                    print(f"    F1 Score: {avg_metrics['f1']:.4f}")
                    print(f"    Precision: {avg_metrics['precision']:.4f}")
                    print(f"    Recall: {avg_metrics['recall']:.4f}")
                    print(f"    SHD: {avg_metrics['shd']:.4f}")
                    print(f"    Time: {avg_metrics['time']:.4f} seconds")
        
        # Print summary of results for each configuration
        print("\n" + "=" * 80)
        print("SUMMARY OF RESULTS")
        print("=" * 80)
        
        for config_name in results:
            print(f"\n{config_name}:")
            print("-" * 60)
            
            print("F1 Scores:")
            for n_nodes in node_sizes:
                scores = [f"{results[config_name][n_nodes][n_samples]['f1']:.4f}" for n_samples in sample_sizes]
                print(f"  Nodes={n_nodes}: {', '.join(scores)}")
            
            print("\nPrecision:")
            for n_nodes in node_sizes:
                scores = [f"{results[config_name][n_nodes][n_samples]['precision']:.4f}" for n_samples in sample_sizes]
                print(f"  Nodes={n_nodes}: {', '.join(scores)}")
            
            print("\nRecall:")
            for n_nodes in node_sizes:
                scores = [f"{results[config_name][n_nodes][n_samples]['recall']:.4f}" for n_samples in sample_sizes]
                print(f"  Nodes={n_nodes}: {', '.join(scores)}")
            
            print("\nSHD:")
            for n_nodes in node_sizes:
                scores = [f"{results[config_name][n_nodes][n_samples]['shd']:.4f}" for n_samples in sample_sizes]
                print(f"  Nodes={n_nodes}: {', '.join(scores)}")
        
        total_time = time.time() - start_time
        print(f"\nTotal experiment time: {total_time:.2f} seconds")
        
        # Analyze and print conclusions
        print("\n" + "=" * 80)
        print("CONCLUSIONS")
        print("=" * 80)
        
        # Analyze sample size vs. node size relationship
        print("\n1. Sample Size to Node Size Relationship:")
        for config_name in results:
            print(f"\n  For {config_name}:")
            for n_nodes in node_sizes:
                # Find the sample size where F1 score first exceeds 0.8 (or the highest if none exceed 0.8)
                f1_scores = [results[config_name][n_nodes][n_samples]['f1'] for n_samples in sample_sizes]
                threshold_indices = [i for i, f1 in enumerate(f1_scores) if f1 >= 0.8]
                if threshold_indices:
                    min_sample_idx = min(threshold_indices)
                    min_sample = sample_sizes[min_sample_idx]
                    print(f"    Nodes={n_nodes}: Minimum sample size for F1 ≥ 0.8: {min_sample} (F1={f1_scores[min_sample_idx]:.4f})")
                else:
                    max_f1_idx = f1_scores.index(max(f1_scores))
                    print(f"    Nodes={n_nodes}: No sample size reached F1 ≥ 0.8. Best: {sample_sizes[max_f1_idx]} (F1={max(f1_scores):.4f})")
        
        # Final recommendations
        print("\n2. Recommendations for FCI:")
        print("  • For optimal performance, the sample size should be at least 100 times the number of nodes")
        print("  • FCI is more computationally intensive than PC, so consider computational resources for larger graphs")
        print("  • Different independence tests may be more appropriate depending on data characteristics")

if __name__ == "__main__":
    fci = FCI()
    fci.test_algorithm()