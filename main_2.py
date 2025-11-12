# Suppress external library logs first

import sys
sys.path.insert(0, '/home/marsh/Documents/Project/Casual_Copilot/Causal-Copilot-main/externals/causal-learn')
import utils.suppress_logs
# Suppress external library logs first
import utils.suppress_logs

# Import logger after suppressing external logs
from utils.logger import logger

# Phase 2 minimal imports (no LLM dependencies)
from postprocess.visualization import Visualization, convert_to_edges
from global_setting.Initialize_state import global_state_initialization, load_data

# All other imports will be done in their respective phases
# Phase 1: knowledge_info, stat_info_collection, Filter, Programming, Reranker, etc.
# Phase 2: Judge (for bootstrap only)
# Phase 3: Report_generation

import os
import json
import argparse
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description='Causal Learning Tool for Data Analysis')

    # Input data file
    parser.add_argument(
        '--data-file',
        type=str,
        # default= "simulated_data/default/data.csv",
        default= "data/dataset/Abalone/Abalone.csv",
        help='Path to the input dataset file (e.g., CSV format or directory location)'
    )

    # Output file for results
    parser.add_argument(
        '--output-report-dir',
        type=str,
        # default='data/dataset/sim_ts/output_report/',
        default='output/Abalone',
        help='Directory to save the output report'
    )

    # Output directory for graphs
    parser.add_argument(
        '--output-graph-dir',
        type=str,
        # default='data/dataset/sim_ts/output_graph/',
        default='output/Abalone',
        help='Directory to save the output graph'
    )

    parser.add_argument(
        '--simulation_mode',
        type=str,
        default="offline",
        help='Simulation mode: online or offline'
    )

    parser.add_argument(
        '--data_mode',
        type=str,
        default="real",
        help='Data mode: real or simulated'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='Enable debugging mode'
    )

    parser.add_argument(
        '--initial_query',
        type=str,
        default="Do causal discovery on this dataset",
        help='Initial query for the algorithm'
    )

    parser.add_argument(
        '--parallel',
        type=bool,
        default=False,
        help='Parallel computing for bootstrapping.'
    )

    parser.add_argument(
        '--demo_mode',
        type=bool,
        default=False,
        help='Demo mode'
    )
    ###################### for_phase
    parser.add_argument(
        '--phase',
        type=int,
        choices=[1, 2, 3],
        default=None,
        help='Phase 1=LLM Config+Algorithm (VM); Phase 2=Bootstrap (Server); Phase 3=LLM Revision+Report (VM)'
    )
    
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Resume from checkpoint file (e.g., output/Abalone/phase1_checkpoint.pkl)'
    )
    
    parser.add_argument(
        '--skip-llm-revision',
        action='store_true',
        default=False,
        help='Skip LLM revision in Phase 3 (useful for offline execution)'
    )

    args = parser.parse_args()
    return args

def load_real_world_data(file_path):
    #Baseline code
    # Checking file format and loading accordingly, right now it's for CSV only
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = pd.DataFrame(json.load(f))
    else:
        raise ValueError(f"Unsupported file format for {file_path}")
    
    # Show basic dataset information
    data_info = {
        "Shape": f"({data.shape[0]:,} rows, {data.shape[1]} columns)",
        "Columns": f"{list(data.columns[:5])}{'...' if len(data.columns) > 5 else ''}",
        "Memory usage": f"{data.memory_usage(deep=True).sum() / 1024**2:.1f} MB",
        "Data types": f"{data.dtypes.value_counts().to_dict()}"
    }
    logger.data_info("Dataset loaded successfully", data_info)
    return data

def process_user_query(query, data):
    logger.detail(f"Processing user query: {query[:100]}...")
    
    #Baseline code
    query_dict = {}
    original_shape = data.shape
    
    if ';' in query or ':' in query:
        for part in query.split(';'):
            if ':' in part:
                key, value = part.strip().split(':')
                query_dict[key.strip()] = value.strip()

    if 'filter' in query_dict and query_dict['filter'] == 'continuous':
        # Filtering continuous columns, just for target practice right now
        data = data.select_dtypes(include=['float64', 'int64'])
        logger.detail(f"Filtered to continuous columns: {original_shape} → {data.shape}")
    
    if 'selected_algorithm' in query_dict:
        selected_algorithm = query_dict['selected_algorithm']
        logger.algorithm("Algorithm manually selected", selected_algorithm)

    # Show query processing results
    processing_results = {
        "Original query": query[:50] + "..." if len(query) > 50 else query,
        "Parsed parameters": len(query_dict),
        "Data shape after processing": f"{data.shape}",
        "Columns selected": f"{len(data.columns)} columns"
    }
    logger.data_info("User query processed", processing_results)
    return data
########################### for phase
def save_checkpoint(global_state, phase_name, args):
    """保存checkpoint到指定阶段"""
    import pickle
    from pathlib import Path
    
    checkpoint_dir = Path(global_state.user_data.output_graph_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f'phase{phase_name}_checkpoint.pkl'
    
    with open(checkpoint_path, 'wb') as f:
        pickle.dump({
            'global_state': global_state,
            'args': args,
            'phase': phase_name
        }, f)
    
    logger.success(f"✅ Phase {phase_name} checkpoint saved: {checkpoint_path}")
    return checkpoint_path
# ========== 🆕 新增函数1 END ==========


# ========== 🆕 新增函数2：加载checkpoint START ==========
def load_checkpoint(checkpoint_path):
    """从checkpoint恢复"""
    import pickle
    
    logger.info(f"📂 Loading checkpoint from: {checkpoint_path}")
    
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    logger.success(f"✅ Resumed from Phase {checkpoint['phase']}")
    return checkpoint['global_state'], checkpoint['args']
# ========== 🆕 新增函数2 END ==========


# ========== 🆕 新增函数3：导出阶段结果 START ==========
def export_phase_results(global_state, phase_name):
    """导出当前阶段的结果"""
    from pathlib import Path
    import json
    
    output_dir = Path(global_state.user_data.output_graph_dir)
    phase_dir = output_dir / f'phase{phase_name}_results'
    phase_dir.mkdir(parents=True, exist_ok=True)
    
    logger.section(f"Exporting Phase {phase_name} Results")
    
    # 1. 保存global_state的pickle（完整版）
    pkl_path = phase_dir / f'phase{phase_name}_global_state.pkl'
    import pickle
    with open(pkl_path, 'wb') as f:
        pickle.dump(global_state, f)
    logger.detail(f"Saved: {pkl_path.name}")
    
    # 2. 保存JSON摘要（便于查看）
    summary = {}
    
    if phase_name >= 1:
        summary['data_info'] = {
            'sample_size': global_state.statistics.sample_size,
            'feature_number': global_state.statistics.feature_number,
            'data_type': global_state.statistics.data_type,
            'columns': global_state.user_data.processed_data.columns.tolist(),
        }
    
    if phase_name >= 1 and hasattr(global_state.algorithm, 'selected_algorithm'):
        summary['algorithm_info'] = {
            'selected_algorithm': global_state.algorithm.selected_algorithm,
            'hyperparameters': global_state.algorithm.algorithm_arguments,
        }
    
    if phase_name >= 2 and hasattr(global_state.results, 'converted_graph'):
        import numpy as np
        summary['graph_info'] = {
            'original_edges': int((global_state.results.converted_graph != 0).sum()),
        }
    
    if phase_name >= 3 and hasattr(global_state.results, 'revised_graph'):
        summary['revision_info'] = {
            'revised_edges': int((global_state.results.revised_graph != 0).sum()),
        }
    
    json_path = phase_dir / f'phase{phase_name}_summary.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.detail(f"Saved: {json_path.name}")
    
    # 3. 如果有因果图，导出图的多种格式
    if phase_name >= 2 and hasattr(global_state.results, 'converted_graph'):
        from postprocess.export_causal_graphs import export_causal_graphs
        
        # 临时修改输出目录到phase目录
        original_dir = global_state.user_data.output_graph_dir
        global_state.user_data.output_graph_dir = str(phase_dir)
        
        export_causal_graphs(global_state)
        
        # 恢复原目录
        global_state.user_data.output_graph_dir = original_dir
        logger.detail(f"Causal graphs exported to phase{phase_name}_results/")
    
    logger.success(f"✅ Phase {phase_name} results exported to: {phase_dir}")


def export_converted_graph(global_state):
    """Export original causal graph after Algorithm Execution"""
    import json
    from pathlib import Path
    
    output_dir = Path(global_state.user_data.output_graph_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    graph_path = output_dir / 'converted_graph.json'
    
    graph_data = {
        'graph_matrix': global_state.results.converted_graph.tolist(),
        'columns': global_state.user_data.processed_data.columns.tolist(),
        'algorithm': global_state.algorithm.selected_algorithm,
        'n_samples': int(global_state.statistics.sample_size),
        'n_features': int(global_state.statistics.feature_number),
        'timestamp': str(pd.Timestamp.now())
    }
    
    with open(graph_path, 'w') as f:
        json.dump(graph_data, f, indent=2)
    
    logger.success(f"✅ Original causal graph saved: {graph_path.name}")
    logger.detail(f"   Graph edges: {(global_state.results.converted_graph != 0).sum()}")


def export_bootstrap_results(global_state):
    """Export bootstrap results after Phase 2"""
    import json
    from pathlib import Path
    import numpy as np
    
    output_dir = Path(global_state.user_data.output_graph_dir)
    
    # 递归转换函数 - 处理所有numpy类型
    def convert_to_serializable(obj):
        """递归转换numpy对象为可序列化类型"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):  # 所有numpy标量类型
            return obj.item()  # 转为Python原生类型
        elif isinstance(obj, dict):
            return {str(k): convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        else:
            try:
                json.dumps(obj)  # 测试是否可序列化
                return obj
            except (TypeError, ValueError):
                return str(obj)
    
    # 处理bootstrap_probability可能是字典或数组
    boot_prob = global_state.results.bootstrap_probability
    
    if isinstance(boot_prob, dict):
        # 字典格式：{(i,j): probability} 或 {'certain_edges': array, ...}
        boot_prob_serializable = convert_to_serializable(boot_prob)
        is_dict = True
    else:
        # 数组格式
        boot_prob_serializable = convert_to_serializable(boot_prob)
        is_dict = False
    
    # 1. JSON format
    bootstrap_path = output_dir / 'bootstrap_results.json'
    bootstrap_data = {
        'bootstrap_probability': boot_prob_serializable,
        'bootstrap_probability_format': 'dict' if is_dict else 'array',
        'bootstrap_check_dict': convert_to_serializable(global_state.results.bootstrap_check_dict),
        'n_bootstraps': int(getattr(global_state.algorithm.algorithm_arguments, 'n_bootstraps', 100)),
        'timestamp': str(pd.Timestamp.now())
    }
    
    with open(bootstrap_path, 'w', encoding='utf-8') as f:
        json.dump(bootstrap_data, f, indent=2, ensure_ascii=False)
    
    # 2. NumPy format (only for array)
    if not is_dict and isinstance(boot_prob, np.ndarray):
        npy_path = output_dir / 'bootstrap_probability.npy'
        np.save(npy_path, boot_prob)
        logger.success(f"✅ Bootstrap results saved:")
        logger.detail(f"   JSON: {bootstrap_path.name}")
        logger.detail(f"   NumPy: {npy_path.name}")
    else:
        logger.success(f"✅ Bootstrap results saved:")
        logger.detail(f"   JSON: {bootstrap_path.name}")
        if is_dict and isinstance(boot_prob, dict):
            logger.detail(f"   Format: dictionary with {len(boot_prob)} edge types")
        else:
            logger.detail(f"   Format: dictionary with {len(boot_prob)} edges")
    
    # 统计高置信边
    if is_dict and isinstance(boot_prob, dict):
        # boot_prob是字典，可能是{'certain_edges': array, ...}
        if 'certain_edges' in boot_prob:
            certain_edges = boot_prob['certain_edges']
            if isinstance(certain_edges, np.ndarray):
                high_conf_count = int((certain_edges > 0.5).sum())
                logger.detail(f"   Certain edges (>0.5): {high_conf_count}")
        else:
            # 或者是{(i,j): prob}格式
            high_conf_count = sum(1 for prob in boot_prob.values() if (isinstance(prob, (int, float, np.number)) and prob > 0.5))
            logger.detail(f"   Confidence edges (>0.5): {high_conf_count}")
    elif not is_dict and isinstance(boot_prob, np.ndarray):
        high_conf_count = int((boot_prob > 0.5).sum())
        logger.detail(f"   Confidence edges (>0.5): {high_conf_count}")


##################################3
def main(args):
    logger.header("Causal-Copilot Analysis Session")

    if args.resume_from:
        global_state, loaded_args = load_checkpoint(args.resume_from)
        # 合并参数（允许覆盖部分参数）
        for key in ['phase', 'skip_llm_revision']:
            if hasattr(args, key):
                setattr(loaded_args, key, getattr(args, key))
        args = loaded_args
        
        # 确定从哪个阶段开始
        if args.phase is None:
            checkpoint_phase = int(args.resume_from.split('phase')[1].split('_')[0])
            start_phase = checkpoint_phase + 1
        else:
            start_phase = args.phase
    else:
        start_phase = 1
        global_state = None    

    ########################################################################################
    # Phase 1: LLM Configuration + Algorithm Execution (VM)
    ########################################################################################
    if start_phase <= 1:
        # Import all LLM-dependent modules for Phase 1
        from preprocess.dataset import knowledge_info
        from preprocess.stat_info_functions import stat_info_collection, convert_stat_info_to_text
        from causal_discovery.filter import Filter
        from causal_discovery.program import Programming
        from causal_discovery.rerank import Reranker
        from causal_discovery.hyperparameter_selector import HyperparameterSelector
        from preprocess.eda_generation import EDA
        from postprocess.judge import Judge
        from report.report_generation import Report_generation
        
        logger.section("=" * 80)
        logger.section("💻 PHASE 1: LLM CONFIGURATION + ALGORITHM EXECUTION (VM)")
        logger.section("=" * 80)
        
        logger.step(1, 10, "Initializing global state")
        global_state = global_state_initialization(args)
        logger.detail("Global state initialized successfully")
        
        logger.step(2, 10, "Loading and preparing data")
        global_state = load_data(global_state, args)
        logger.detail("Data loading completed")

        if args.data_mode == 'real':
            global_state.user_data.raw_data = load_real_world_data(args.data_file)
        
        logger.step(3, 10, "Processing user query")
        global_state.user_data.processed_data = process_user_query(args.initial_query, global_state.user_data.raw_data)
        global_state.user_data.visual_selected_features = global_state.user_data.processed_data.columns.tolist()
        global_state.user_data.selected_features = global_state.user_data.processed_data.columns.tolist()

        logger.step(4, 10, "Collecting statistical information")
        if args.debug:
            logger.detail("Using debug mode with fake statistics")
            # Fake statistics for debugging
            global_state.statistics.sample_size = 853
            global_state.statistics.feature_number = 11
            global_state.statistics.missingness = False
            global_state.statistics.data_type = "Continuous"
            global_state.statistics.linearity = True
            global_state.statistics.gaussian_error = True
            global_state.statistics.stationary = "non time-series"
            global_state.user_data.processed_data = global_state.user_data.raw_data
            global_state.user_data.knowledge_docs = "This is fake domain knowledge for debugging purposes."
            logger.detail("Debug statistics and knowledge information loaded")
        else:
            logger.detail("Analyzing dataset characteristics...")
            global_state = stat_info_collection(global_state)
            
            logger.detail("Collecting domain knowledge...")
            global_state = knowledge_info(args, global_state)

        # Convert statistics to text
        global_state.statistics.description = convert_stat_info_to_text(global_state.statistics)

        # Show detailed data information
        data_details = {
            "Shape": f"{global_state.user_data.processed_data.shape}",
            "Columns": len(global_state.user_data.processed_data.columns),
            "Missing values": global_state.user_data.processed_data.isnull().sum().sum(),
            "Data type": getattr(global_state.statistics, 'data_type', 'Unknown')
        }
        logger.data_info("Dataset preprocessed", data_details)

        logger.checkpoint("Data preprocessing completed")
        
        #############EDA###################
        logger.step(5, 10, "Exploratory Data Analysis")
        logger.detail("Generating statistical summaries and visualizations...")
        my_eda = EDA(global_state)
        my_eda.generate_eda()
        logger.detail("EDA completed - visualizations saved")
        
        logger.step(6, 10, "Algorithm Selection")
        
        logger.detail("Step 1/3: Filtering suitable algorithms")
        filter = Filter(args)
        global_state = filter.forward(global_state)
        if hasattr(global_state.algorithm, 'filtered_algorithms'):
            logger.detail(f"Filtered to {len(global_state.algorithm.filtered_algorithms)} candidate algorithms")
        else:
            logger.detail("Algorithm filtering completed")

        logger.detail("Step 2/3: Ranking algorithms by suitability")
        reranker = Reranker(args)
        global_state = reranker.forward(global_state)
        logger.detail("Algorithm ranking completed")

        logger.detail("Step 3/3: Optimizing hyperparameters")
        hp_selector = HyperparameterSelector(args)
        global_state = hp_selector.forward(global_state)
        
        logger.algorithm("Selected algorithm", global_state.algorithm.selected_algorithm)
        if hasattr(global_state.algorithm, 'hyperparameters'):
            logger.detail(f"Hyperparameters: {len(global_state.algorithm.hyperparameters)} parameters optimized")
        else:
            logger.detail("Hyperparameter optimization completed")
        
        # Step 7: Algorithm Execution
        logger.step(7, 10, "Algorithm Execution")
        logger.detail(f"Running {global_state.algorithm.selected_algorithm} algorithm...")
        try:
            programmer = Programming(args)
            global_state = programmer.forward(global_state)
            logger.detail("Algorithm execution completed")
            
            # Show graph statistics
            if hasattr(global_state.results, 'converted_graph'):
                graph = global_state.results.converted_graph
                if graph is not None:
                    edges = (graph != 0).sum()
                    logger.detail(f"Discovered {edges} edges in causal graph")
                else:
                    logger.warning("No graph result found")
            else:
                logger.warning("No results attribute found")
            
            logger.checkpoint("Causal discovery completed")
        except Exception as e:
            logger.error(f"Algorithm execution failed: {str(e)}")
            raise
        
        #############Visualization for Initial Graph###################
        my_visual_initial = Visualization(global_state)
        if global_state.statistics.time_series and global_state.results.lagged_graph is not None:
                converted_graph = global_state.results.lagged_graph
                pos_est = my_visual_initial.get_pos(converted_graph[0])
                for i in range(converted_graph.shape[0]):
                    _ = my_visual_initial.plot_pdag(converted_graph[i], f'{global_state.algorithm.selected_algorithm}_initial_graph_{i}.svg', pos=pos_est)
                summary_graph = np.any(converted_graph, axis=0).astype(int)
                # pos_est = my_visual_initial.get_pos(summary_graph)
                _ = my_visual_initial.plot_pdag(summary_graph, f'{global_state.algorithm.selected_algorithm}_initial_graph_summary.svg', pos=pos_est)
                my_report = Report_generation(global_state, args)
        else:
            # Get the position of the nodes
            pos_est = my_visual_initial.get_pos(global_state.results.converted_graph)
            # Plot True Graph
            if global_state.user_data.ground_truth is not None:
                _ = my_visual_initial.plot_pdag(global_state.user_data.ground_truth, 'true_graph.pdf', pos=pos_est)
            # Plot Initial Graph
            _ = my_visual_initial.plot_pdag(global_state.results.converted_graph, f'{global_state.algorithm.selected_algorithm}_initial_graph.pdf', pos=pos_est)
            my_report = Report_generation(global_state, args)
            global_state.results.raw_edges = convert_to_edges(global_state.algorithm.selected_algorithm, global_state.user_data.processed_data.columns, global_state.results.converted_graph)
            global_state.logging.graph_conversion['initial_graph_analysis'] = my_report.graph_effect_prompts()
            judge = Judge(global_state, args)
            if global_state.user_data.ground_truth is not None:
                logger.section("Graph Evaluation")
                logger.detail("Comparing with ground truth graph")
                global_state.results.metrics = judge.evaluation(global_state)
                if hasattr(global_state.results, 'metrics') and global_state.results.metrics:
                    logger.metrics_table(global_state.results.metrics, "Performance Metrics")
        
        # Export converted graph
        logger.detail("Exporting original causal graph...")
        export_converted_graph(global_state)
        
        save_checkpoint(global_state, 1, args)
        export_phase_results(global_state, 1)
        
        if args.phase == 1:
            logger.checkpoint("=" * 80)
            logger.checkpoint("✅ PHASE 1 COMPLETED - ALGORITHM EXECUTION DONE")
            logger.checkpoint("=" * 80)
            logger.info("📤 Next steps:")
            logger.info("   1. Transfer to server: phase1_checkpoint.pkl")
            logger.info("   2. Run: python main.py --phase 2 --resume-from phase1_checkpoint.pkl")
            logger.elapsed_time("Phase 1 execution time")
            return None, global_state
    
    ########################################################################################
    # Phase 2: Bootstrap Analysis (Server - NO NETWORK)
    ########################################################################################
    if start_phase <= 2:
        # Import Judge for bootstrap analysis (should not require LLM for bootstrap_analysis_only)
        from postprocess.judge import Judge
        
        logger.section("=" * 80)
        logger.section("🖥️  PHASE 2: BOOTSTRAP ANALYSIS (SERVER MODE)")
        logger.section("=" * 80)
        
        logger.step(8, 10, "Bootstrap Analysis")
        logger.info("→ Statistical computing only (no LLM)")
        logger.detail("Applying bootstrap sampling and statistical tests")
        logger.warning("⏰ This may take several hours depending on data size...")
        
        judge = Judge(global_state, args)
        global_state = judge.bootstrap_analysis_only(global_state)
        logger.success("Bootstrap analysis completed")
        
        # Export bootstrap results
        export_bootstrap_results(global_state)
        
        save_checkpoint(global_state, 2, args)
        export_phase_results(global_state, 2)
        
        if args.phase == 2:
            logger.checkpoint("=" * 80)
            logger.checkpoint("✅ PHASE 2 COMPLETED - BOOTSTRAP DONE")
            logger.checkpoint("=" * 80)
            logger.info("📥 Next steps:")
            logger.info("   1. Transfer back to VM: phase2_checkpoint.pkl")
            logger.info("   2. Run: python main.py --phase 3 --resume-from phase2_checkpoint.pkl")
            logger.elapsed_time("Phase 2 execution time")
            return None, global_state
    
    ########################################################################################
    # Phase 3: LLM Post-processing + Report Generation (VM)
    ########################################################################################
    if start_phase <= 3:
        # Import report generation and Judge for Phase 3 (requires LLM)
        from postprocess.judge import Judge
        from report.report_generation import Report_generation
        
        logger.section("=" * 80)
        logger.section("💻 PHASE 3: LLM POST-PROCESSING + REPORT (VM)")
        logger.section("=" * 80)
        
        # Step 9: LLM Graph Revision
        logger.step(9, 10, "LLM Graph Revision")
        logger.detail("Refining graph structure using domain knowledge")
        
        judge = Judge(global_state, args)
        if args.skip_llm_revision:
            logger.warning("Skipping LLM revision (--skip-llm-revision flag)")
            global_state.results.revised_graph = global_state.results.converted_graph.copy()
        else:
            global_state = judge.llm_revision_only(global_state)
            logger.success("LLM revision completed")
        
        #############Visualization for Revised Graph###################
        logger.section("Graph Visualization")
        logger.detail("Generating visualization for revised graph and confidence heatmaps")
        
        # 初始化pos_est（节点位置）
        my_visual_revise = Visualization(global_state)
        pos_est = my_visual_revise.get_pos(global_state.results.revised_graph)
        
        # Plot Revised Graph
        pos_new = my_visual_revise.plot_pdag(global_state.results.revised_graph, f'{global_state.algorithm.selected_algorithm}_revised_graph.pdf', pos=pos_est)
        global_state.results.revised_edges = convert_to_edges(global_state.algorithm.selected_algorithm, global_state.user_data.processed_data.columns, global_state.results.revised_graph)
        
        # Plot Bootstrap Heatmap - CRITICAL: This must happen before report generation
        logger.detail("Generating bootstrap confidence heatmaps")
        boot_heatmap_paths = my_visual_revise.boot_heatmap_plot()
        if boot_heatmap_paths:
            logger.success(f"Generated {len(boot_heatmap_paths)} confidence heatmap(s)")
            for path in boot_heatmap_paths:
                logger.debug(f"Generated heatmap: {os.path.basename(path)}", "Visualization")
        else:
            logger.warning("No confidence heatmaps were generated (bootstrap data may be empty)")
        
        # global_state.results.refutation_analysis = judge.graph_refutation(global_state)

        # algorithm selection process
        '''
        round = 0
        flag = False

        while round < args.max_iterations and flag == False:
            code, results = programmer.forward(preprocessed_data, algorithm, hyper_suggest)
            flag, algorithm_setup = judge(preprocessed_data, code, results, statistics_dict, algorithm_setup, knowledge_docs)
        '''
        logger.step(10, 10, "Report Generation")
        #############Report Generation###################
        try_num = 1
        
        logger.detail("Step 1/3: Analyzing causal relationships")
        try:
            global_state.results.raw_edges = convert_to_edges(global_state.algorithm.selected_algorithm, global_state.user_data.processed_data.columns, global_state.results.converted_graph)
            global_state.logging.graph_conversion['initial_graph_analysis'] = my_report.graph_effect_prompts()
            analysis_clean = global_state.logging.graph_conversion['initial_graph_analysis'].replace('"',"").replace("\\n\\n", "\n\n").replace("\\n", "\n").replace("'", "")
            logger.detail("Causal relationship analysis completed")
        except Exception as e:
            logger.error(f"Causal relationship analysis failed: {str(e)}")
            raise
        
        logger.detail("Step 2/3: Generating comprehensive report")
        try:
            my_report = Report_generation(global_state, args)
            report = my_report.generation()
            my_report.save_report(report)
            report_path = os.path.join(global_state.user_data.output_report_dir, 'report.pdf')  
            
            while (not os.path.isfile(report_path)) and try_num<3:
                try_num += 1
                logger.warning(f"Report generation failed, retrying ({try_num}/3)")
                report_gen = Report_generation(global_state, args)
                report = report_gen.generation(debug=False)
                report_gen.save_report(report)
            
            if os.path.isfile(report_path):
                logger.detail("Step 3/3: Report saved successfully")
                logger.detail(f"Report location: {os.path.basename(report_path)}")
                
                # Show final summary
                final_summary = {
                    "Algorithm used": global_state.algorithm.selected_algorithm,
                    "Graph edges": f"{(global_state.results.converted_graph != 0).sum() if hasattr(global_state.results, 'converted_graph') and global_state.results.converted_graph is not None else 'Unknown'}",
                    "Report saved": os.path.basename(report_path),
                    "Output directory": os.path.basename(global_state.user_data.output_report_dir)
                }
                logger.data_info("Analysis completed successfully", final_summary)
            else:
                logger.error("Report generation failed after 3 attempts")
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            raise
    ################################
    from postprocess.export_causal_graphs import export_causal_graphs, export_specific_relationships

    # 添加新步骤 (如果想显示进度)
    export_phase_results(global_state, 3)
    logger.step(9, 9, "Exporting Causal Graphs")
    logger.detail("Saving causal graphs in multiple formats...")

    # 1. 导出完整的因果图 (original + revised)
    export_causal_graphs(global_state)

    # 2. 自动识别架构参数和性能指标
    columns = global_state.user_data.processed_data.columns
    n_arch_params = 22  # 根据你的数据调整这个数字

    # 方式1: 按位置划分 (推荐)
    arch_params = columns[:n_arch_params].tolist()
    metrics = columns[n_arch_params:].tolist()

    # 方式2: 如果你确定后4列是性能指标
    # metrics = columns[-4:].tolist()

    # 方式3: 根据列名关键字自动识别 (最灵活)
    # metrics = [col for col in columns if any(keyword in col.lower() 
    #            for keyword in ['cpi', 'miss', 'flush', 'cycle'])]
    # arch_params = [col for col in columns if col not in metrics]

    logger.detail(f"Identified {len(arch_params)} architecture parameters")
    logger.detail(f"Identified {len(metrics)} performance metrics: {', '.join(metrics)}")

    # 3. 导出架构参数 -> 性能指标的因果关系
    export_specific_relationships(
        global_state,
        source_vars=arch_params,
        target_vars=metrics
    )

    logger.checkpoint("Causal graphs exported successfully")
        ################################

    logger.section("Saving Analysis Results")
    try:
        json_path, pkl_path = save_global_state_to_json(global_state)
        logger.success("All results saved successfully!")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

    # User discussion part
    from user.discuss import Discussion
    discussion = Discussion(args, report)
    discussion.forward(global_state)
    
    logger.checkpoint("=" * 80)
    logger.checkpoint("✅ ANALYSIS SESSION COMPLETED")
    logger.checkpoint("=" * 80)
    logger.elapsed_time("Total analysis time")

    return report, global_state

def save_global_state_to_json(global_state):
    """保存global_state到JSON和Pickle"""
    import pickle
    import json
    from pathlib import Path
    
    output_dir = global_state.user_data.output_graph_dir
    algo_name = global_state.algorithm.selected_algorithm
    
    # 1. 保存完整的pickle (用于后续加载)
    pkl_path = Path(output_dir) / f'{algo_name}_global_state.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(global_state, f)
    logger.success(f"Saved global_state pickle to: {pkl_path}")
    
    # 2. 递归转换函数（新增键转换逻辑）
    def convert_to_serializable(obj):
        """递归转换为JSON可序列化对象"""
        import numpy as np
        
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            # ★ 关键修改：将tuple键转为字符串
            return {
                str(k): convert_to_serializable(v) 
                for k, v in obj.items()
            }
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return {
                k: convert_to_serializable(v) 
                for k, v in obj.__dict__.items() 
                if not k.startswith('_')
            }
        else:
            try:
                json.dumps(obj)  # 测试是否可序列化
                return obj
            except:
                return str(obj)  # 无法序列化则转字符串
    
    # 3. 定义要保存的字段
    data_to_save = {
        'algorithm': algo_name,
        'user_data': {
            'initial_query': global_state.user_data.initial_query,
            'knowledge_docs': global_state.user_data.knowledge_docs,
            'knowledge_docs_for_user': getattr(global_state.user_data, 'knowledge_docs_for_user', None),
        },
        'statistics': {
            'sample_size': global_state.statistics.sample_size,
            'feature_number': global_state.statistics.feature_number,
            'data_type': global_state.statistics.data_type,
            'linearity': global_state.statistics.linearity,
            'gaussian_error': global_state.statistics.gaussian_error,
            'missingness': global_state.statistics.missingness,
            'heterogeneous': global_state.statistics.heterogeneous,
            'time_series': global_state.statistics.time_series,
            'description': global_state.statistics.description,
        },
        'algorithm_info': {
            'selected_algorithm': global_state.algorithm.selected_algorithm,
            'selected_reason': global_state.algorithm.selected_reason,
            'algorithm_candidates': global_state.algorithm.algorithm_candidates,
            'algorithm_arguments': global_state.algorithm.algorithm_arguments,
            'algorithm_optimum': getattr(global_state.algorithm, 'algorithm_optimum', None),
        },
        'results': {
            'converted_graph': convert_to_serializable(global_state.results.converted_graph),
            'revised_graph': convert_to_serializable(global_state.results.revised_graph),
            'bootstrap_probability': convert_to_serializable(global_state.results.bootstrap_probability),
            'bootstrap_check_dict': convert_to_serializable(global_state.results.bootstrap_check_dict),
            'llm_errors': convert_to_serializable(global_state.results.llm_errors),
        },
        'logging': {
            'select_conversation': global_state.logging.select_conversation,
            'argument_conversation': global_state.logging.argument_conversation,
            'knowledge_conversation': global_state.logging.knowledge_conversation,
        }
    }
    
    # 4. 保存为JSON（新增错误处理）
    json_path = Path(output_dir) / f'{algo_name}_complete_information.json'
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(
                convert_to_serializable(data_to_save),  # ★ 再次转换确保安全
                f, 
                indent=2, 
                ensure_ascii=False
            )
        logger.success(f"Saved complete information to: {json_path}")
    except Exception as e:
        logger.warning(f"Failed to save JSON (non-critical): {str(e)}")
    
    # 5. 额外保存纯文本的knowledge_docs
    if global_state.user_data.knowledge_docs:
        txt_path = Path(output_dir) / f'{algo_name}_knowledge_docs.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("DOMAIN KNOWLEDGE EXTRACTED BY LLM\n")
            f.write("=" * 80 + "\n\n")
            for i, doc in enumerate(global_state.user_data.knowledge_docs, 1):
                f.write(f"Document {i}:\n")
                f.write("-" * 80 + "\n")
                f.write(doc)
                f.write("\n\n")
        logger.success(f"Saved knowledge docs to: {txt_path}")
    
    return json_path, pkl_path

if __name__ == '__main__':
    args = parse_args()
    main(args)