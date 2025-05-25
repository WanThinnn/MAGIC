import torch
import dgl
from utils.loaddata import load_batch_level_dataset, load_entity_level_dataset, load_metadata
import datetime
import argparse

def analyze_dataset_to_file(dataset_name, output_file):
    """Phân tích dataset và xuất kết quả ra file txt"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Header
        f.write(f"=== MAGIC Dataset Analysis Report ===\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Analysis Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n\n")
        
        try:
            if dataset_name in ['streamspot', 'wget']:
                analyze_batch_level_dataset(dataset_name, f)
            else:
                analyze_entity_level_dataset(dataset_name, f)
        except Exception as e:
            f.write(f"ERROR: Failed to analyze dataset: {str(e)}\n")
            print(f"Error analyzing {dataset_name}: {e}")

def analyze_batch_level_dataset(dataset_name, file_handle):
    """Phân tích batch-level dataset (StreamSpot, Wget)"""
    f = file_handle
    
    f.write("DATASET TYPE: Batch-level\n\n")
    
    # Load dataset
    dataset = load_batch_level_dataset(dataset_name)
    graphs = dataset['dataset']
    
    f.write("BASIC INFORMATION:\n")
    f.write(f"  Total graphs: {len(graphs)}\n")
    f.write(f"  Node features: {dataset['n_feat']} dimensions\n")
    f.write(f"  Edge features: {dataset['e_feat']} dimensions\n")
    
    # Kiểm tra cấu trúc dữ liệu
    f.write(f"  Data structure: {type(graphs[0])}\n")
    
    # Trích xuất graphs
    if isinstance(graphs[0], tuple):
        actual_graphs = [g[0] for g in graphs]
        f.write(f"  Note: Extracted graphs from tuples\n")
    else:
        actual_graphs = graphs
    
    f.write("\n")
    
    # Thống kê chi tiết
    analyze_graph_statistics(actual_graphs, f)
    
    # Train/test split info
    if 'train_index' in dataset:
        train_graphs = dataset['train_index']
        f.write(f"TRAIN/TEST SPLIT:\n")
        f.write(f"  Training graphs: {len(train_graphs)}\n")
        f.write(f"  Test graphs: {len(graphs) - len(train_graphs)}\n")
        f.write(f"  Train ratio: {len(train_graphs)/len(graphs)*100:.1f}%\n\n")

def analyze_entity_level_dataset(dataset_name, file_handle):
    """Phân tích entity-level dataset (DARPA)"""
    f = file_handle
    
    f.write("DATASET TYPE: Entity-level\n\n")
    
    # Load metadata
    metadata = load_metadata(dataset_name)
    
    f.write("BASIC INFORMATION:\n")
    f.write(f"  Training graphs: {metadata['n_train']}\n")
    f.write(f"  Test graphs: {metadata['n_test']}\n")
    f.write(f"  Node features: {metadata['node_feature_dim']} dimensions\n")
    f.write(f"  Edge features: {metadata['edge_feature_dim']} dimensions\n\n")
    
    # Malicious nodes info
    if 'malicious' in metadata:
        malicious_nodes, _ = metadata['malicious']
        f.write(f"GROUND TRUTH:\n")
        f.write(f"  Malicious nodes: {len(malicious_nodes)}\n")
        f.write(f"  Malicious node IDs: {malicious_nodes[:10]}...\n")
        f.write(f"  (showing first 10 out of {len(malicious_nodes)})\n\n")
    
    # Sample một vài graphs để phân tích
    f.write("SAMPLE GRAPH ANALYSIS:\n")
    sample_graphs = []
    
    # Phân tích 5 training graphs đầu tiên
    for i in range(min(5, metadata['n_train'])):
        try:
            graph = load_entity_level_dataset(dataset_name, 'train', i)
            sample_graphs.append(graph)
            f.write(f"  Training Graph {i}: {graph.num_nodes()} nodes, {graph.num_edges()} edges\n")
        except Exception as e:
            f.write(f"  Training Graph {i}: ERROR - {str(e)}\n")
    
    # Phân tích 3 test graphs đầu tiên
    for i in range(min(3, metadata['n_test'])):
        try:
            graph = load_entity_level_dataset(dataset_name, 'test', i)
            sample_graphs.append(graph)
            f.write(f"  Test Graph {i}: {graph.num_nodes()} nodes, {graph.num_edges()} edges\n")
        except Exception as e:
            f.write(f"  Test Graph {i}: ERROR - {str(e)}\n")
    
    f.write("\n")
    
    if sample_graphs:
        analyze_graph_statistics(sample_graphs, f, prefix="SAMPLE ")

def analyze_graph_statistics(graphs, file_handle, prefix=""):
    """Phân tích thống kê chi tiết của graphs"""
    f = file_handle
    
    # Tính toán thống kê
    nodes_count = [g.num_nodes() for g in graphs]
    edges_count = [g.num_edges() for g in graphs]
    
    f.write(f"{prefix}GRAPH STATISTICS:\n")
    f.write(f"  Total graphs analyzed: {len(graphs)}\n")
    f.write(f"  \n")
    f.write(f"  NODES:\n")
    f.write(f"    Min: {min(nodes_count)}\n")
    f.write(f"    Max: {max(nodes_count)}\n")
    f.write(f"    Average: {sum(nodes_count)/len(nodes_count):.1f}\n")
    f.write(f"    Median: {sorted(nodes_count)[len(nodes_count)//2]}\n")
    f.write(f"  \n")
    f.write(f"  EDGES:\n")
    f.write(f"    Min: {min(edges_count)}\n")
    f.write(f"    Max: {max(edges_count)}\n")
    f.write(f"    Average: {sum(edges_count)/len(edges_count):.1f}\n")
    f.write(f"    Median: {sorted(edges_count)[len(edges_count)//2]}\n")
    f.write(f"  \n")
    
    # Phân loại graphs theo kích thước
    small_graphs = [i for i, n in enumerate(nodes_count) if n <= 100]
    medium_graphs = [i for i, n in enumerate(nodes_count) if 100 < n <= 1000]
    large_graphs = [i for i, n in enumerate(nodes_count) if n > 1000]
    
    f.write(f"  SIZE DISTRIBUTION:\n")
    f.write(f"    Small graphs (≤100 nodes): {len(small_graphs)}\n")
    f.write(f"    Medium graphs (101-1000 nodes): {len(medium_graphs)}\n")
    f.write(f"    Large graphs (>1000 nodes): {len(large_graphs)}\n")
    f.write(f"  \n")
    
    # Density analysis
    densities = []
    for g in graphs:
        n_nodes = g.num_nodes()
        n_edges = g.num_edges()
        max_edges = n_nodes * (n_nodes - 1)
        density = n_edges / max_edges if max_edges > 0 else 0
        densities.append(density)
    
    f.write(f"  GRAPH DENSITY:\n")
    f.write(f"    Min density: {min(densities):.6f}\n")
    f.write(f"    Max density: {max(densities):.6f}\n")
    f.write(f"    Average density: {sum(densities)/len(densities):.6f}\n")
    f.write(f"  \n")

def analyze_all_datasets(output_dir="./"):
    """Phân tích tất cả datasets"""
    datasets = ['streamspot', 'wget', 'trace', 'theia', 'cadets']
    
    for dataset in datasets:
        print(f"Analyzing {dataset}...")
        output_file = f"{output_dir}analysis_{dataset}.txt"
        try:
            analyze_dataset_to_file(dataset, output_file)
            print(f"Analysis saved to {output_file}")
        except Exception as e:
            print(f"Failed to analyze {dataset}: {e}")
        print("-" * 40)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze MAGIC datasets')
    
    parser.add_argument('--dataset', type=str, default='streamspot',
                        choices=['streamspot', 'wget', 'trace', 'theia', 'cadets', 'fivedirections', 'all'],
                        help='Dataset to analyze (default: streamspot)')
    
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: analysis_{dataset}.txt)')
    
    parser.add_argument('--output_dir', type=str, default='./',
                        help='Output directory for analysis files (default: ./)')
    
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress information')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    if args.verbose:
        print(f"Starting analysis with arguments: {args}")
    
    if args.dataset == 'all':
        # Phân tích tất cả datasets
        print("Analyzing all datasets...")
        analyze_all_datasets(args.output_dir)
        print("All analyses completed!")
    else:
        # Phân tích dataset cụ thể
        dataset_name = args.dataset
        
        if args.output:
            output_file = args.output
        else:
            output_file = f'{args.output_dir}analysis_{dataset_name}.txt'
        
        if args.verbose:
            print(f"Dataset: {dataset_name}")
            print(f"Output file: {output_file}")
        
        print(f"Analyzing dataset: {dataset_name}")
        analyze_dataset_to_file(dataset_name, output_file)
        print(f"Analysis completed. Results saved to: {output_file}")