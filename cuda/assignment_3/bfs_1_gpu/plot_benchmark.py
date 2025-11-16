#!/usr/bin/env python3
"""
BFS CUDA Benchmark Visualization
Plots runtime comparisons for baseline, stream, and CUDA graph implementations
across different graph sizes and degrees.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Benchmark results extracted from benchmark_results.txt
# Format: (nodes, degree, baseline_ms, stream_ms, graph_ms)
benchmark_data = [
    (1000, 8, 50.156, 51.829, 44.522),
    (5000, 8, 51.207, 46.68, 48.635),
    (10000, 8, 45.659, 42.429, 43.736),
    (50000, 8, 44.327, 47.977, 47.534),
    (100000, 8, 51.429, 57.703, 52.664),
    (10000, 16, 44.378, 38.525, 41.653),
    (50000, 16, 44.825, 48.746, 50.876),
    (100000, 16, 62.107, 61.542, 55.219),
    (10000, 32, 41.942, 39.648, 49.6),
    (50000, 32, 43.657, 50.334, 48.755),
]

# Separate data by degree
degree_8 = [(n, b, s, g) for n, d, b, s, g in benchmark_data if d == 8]
degree_16 = [(n, b, s, g) for n, d, b, s, g in benchmark_data if d == 16]
degree_32 = [(n, b, s, g) for n, d, b, s, g in benchmark_data if d == 16]

# Separate data by fixed node sizes
nodes_10k = [(d, b, s, g) for n, d, b, s, g in benchmark_data if n == 10000]
nodes_50k = [(d, b, s, g) for n, d, b, s, g in benchmark_data if n == 50000]
nodes_100k = [(d, b, s, g) for n, d, b, s, g in benchmark_data if n == 100000]


def plot_by_degree():
    """Plot runtime vs number of nodes, separated by degree"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('BFS Runtime vs Graph Size (by Degree)', fontsize=16, fontweight='bold')
    
    datasets = [
        (degree_8, 8, axes[0]),
        (degree_16, 16, axes[1]),
        (degree_32, 32, axes[2])
    ]
    
    for data, degree, ax in datasets:
        if not data:
            continue
            
        nodes = [d[0] for d in data]
        baseline = [d[1] for d in data]
        stream = [d[2] for d in data]
        graph = [d[3] for d in data]
        
        ax.plot(nodes, baseline, 'o-', linewidth=2, markersize=8, label='Baseline', color='#e74c3c')
        ax.plot(nodes, stream, 's-', linewidth=2, markersize=8, label='Stream', color='#3498db')
        ax.plot(nodes, graph, '^-', linewidth=2, markersize=8, label='CUDA Graph', color='#2ecc71')
        
        ax.set_xlabel('Number of Nodes', fontsize=12, fontweight='bold')
        ax.set_ylabel('Runtime (ms)', fontsize=12, fontweight='bold')
        ax.set_title(f'Degree = {degree}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10)
        ax.set_xscale('log')
        
        # Add value labels
        for i, n in enumerate(nodes):
            ax.annotate(f'{baseline[i]:.1f}', (n, baseline[i]), 
                       textcoords="offset points", xytext=(0,5), ha='center', fontsize=8, color='#e74c3c')
    
    plt.tight_layout()
    plt.savefig('benchmark_by_degree.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: benchmark_by_degree.png")
    plt.close()


def plot_by_nodes():
    """Plot runtime vs degree, separated by number of nodes"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('BFS Runtime vs Degree (by Graph Size)', fontsize=16, fontweight='bold')
    
    datasets = [
        (nodes_10k, '10K', axes[0]),
        (nodes_50k, '50K', axes[1]),
        (nodes_100k, '100K', axes[2])
    ]
    
    for data, label, ax in datasets:
        if not data:
            continue
            
        degrees = [d[0] for d in data]
        baseline = [d[1] for d in data]
        stream = [d[2] for d in data]
        graph = [d[3] for d in data]
        
        x = np.arange(len(degrees))
        width = 0.25
        
        bars1 = ax.bar(x - width, baseline, width, label='Baseline', color='#e74c3c', alpha=0.8)
        bars2 = ax.bar(x, stream, width, label='Stream', color='#3498db', alpha=0.8)
        bars3 = ax.bar(x + width, graph, width, label='CUDA Graph', color='#2ecc71', alpha=0.8)
        
        ax.set_xlabel('Degree (Neighbors per Node)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Runtime (ms)', fontsize=12, fontweight='bold')
        ax.set_title(f'{label} Nodes', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(degrees)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('benchmark_by_nodes.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: benchmark_by_nodes.png")
    plt.close()


def plot_speedup_heatmap():
    """Plot speedup heatmap for CUDA Graph vs Baseline"""
    # Organize data into a matrix
    nodes_list = sorted(set(n for n, _, _, _, _ in benchmark_data))
    degrees_list = sorted(set(d for _, d, _, _, _ in benchmark_data))
    
    # Create speedup matrix
    speedup_matrix = np.zeros((len(degrees_list), len(nodes_list)))
    
    for i, degree in enumerate(degrees_list):
        for j, nodes in enumerate(nodes_list):
            # Find matching data point
            for n, d, baseline, stream, graph in benchmark_data:
                if n == nodes and d == degree:
                    speedup_matrix[i, j] = baseline / graph
                    break
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(speedup_matrix, cmap='RdYlGn', aspect='auto', vmin=0.8, vmax=1.2)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(nodes_list)))
    ax.set_yticks(np.arange(len(degrees_list)))
    ax.set_xticklabels([f'{n//1000}K' for n in nodes_list])
    ax.set_yticklabels(degrees_list)
    
    ax.set_xlabel('Number of Nodes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Degree', fontsize=12, fontweight='bold')
    ax.set_title('CUDA Graph Speedup vs Baseline\n(Green = Faster, Red = Slower)', 
                 fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Speedup (×)', rotation=270, labelpad=20, fontsize=11)
    
    # Annotate cells with values
    for i in range(len(degrees_list)):
        for j in range(len(nodes_list)):
            if speedup_matrix[i, j] > 0:
                text = ax.text(j, i, f'{speedup_matrix[i, j]:.2f}×',
                             ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('speedup_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: speedup_heatmap.png")
    plt.close()


def plot_combined_comparison():
    """Create a comprehensive comparison plot"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Top left: All data points
    ax1 = fig.add_subplot(gs[0, 0])
    nodes = [n for n, _, _, _, _ in benchmark_data]
    baseline = [b for _, _, b, _, _ in benchmark_data]
    stream = [s for _, _, _, s, _ in benchmark_data]
    graph = [g for _, _, _, _, g in benchmark_data]
    
    x = np.arange(len(benchmark_data))
    width = 0.25
    
    ax1.bar(x - width, baseline, width, label='Baseline', color='#e74c3c', alpha=0.8)
    ax1.bar(x, stream, width, label='Stream', color='#3498db', alpha=0.8)
    ax1.bar(x + width, graph, width, label='CUDA Graph', color='#2ecc71', alpha=0.8)
    
    labels = [f'{n//1000}K\nD{d}' for n, d, _, _, _ in benchmark_data]
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel('Runtime (ms)', fontsize=11, fontweight='bold')
    ax1.set_title('All Test Configurations', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Top right: Degree 8 scaling
    ax2 = fig.add_subplot(gs[0, 1])
    d8_data = degree_8
    nodes_d8 = [d[0] for d in d8_data]
    baseline_d8 = [d[1] for d in d8_data]
    stream_d8 = [d[2] for d in d8_data]
    graph_d8 = [d[3] for d in d8_data]
    
    ax2.plot(nodes_d8, baseline_d8, 'o-', linewidth=2.5, markersize=10, label='Baseline', color='#e74c3c')
    ax2.plot(nodes_d8, stream_d8, 's-', linewidth=2.5, markersize=10, label='Stream', color='#3498db')
    ax2.plot(nodes_d8, graph_d8, '^-', linewidth=2.5, markersize=10, label='CUDA Graph', color='#2ecc71')
    
    ax2.set_xlabel('Number of Nodes', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Runtime (ms)', fontsize=11, fontweight='bold')
    ax2.set_title('Scaling with Degree = 8', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xscale('log')
    
    # Bottom left: Speedup comparison
    ax3 = fig.add_subplot(gs[1, 0])
    stream_speedup = [baseline[i] / stream[i] for i in range(len(baseline))]
    graph_speedup = [baseline[i] / graph[i] for i in range(len(baseline))]
    
    x = np.arange(len(benchmark_data))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, stream_speedup, width, label='Stream vs Baseline', 
                    color='#3498db', alpha=0.8)
    bars2 = ax3.bar(x + width/2, graph_speedup, width, label='CUDA Graph vs Baseline', 
                    color='#2ecc71', alpha=0.8)
    
    ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Baseline (1.0×)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=8)
    ax3.set_ylabel('Speedup (×)', fontsize=11, fontweight='bold')
    ax3.set_title('Speedup Comparison', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Bottom right: Summary statistics
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    avg_baseline = np.mean(baseline)
    avg_stream = np.mean(stream)
    avg_graph = np.mean(graph)
    
    avg_stream_speedup = np.mean(stream_speedup)
    avg_graph_speedup = np.mean(graph_speedup)
    
    max_stream_speedup = max(stream_speedup)
    max_graph_speedup = max(graph_speedup)
    
    max_stream_idx = stream_speedup.index(max_stream_speedup)
    max_graph_idx = graph_speedup.index(max_graph_speedup)
    
    summary_text = f"""
    PERFORMANCE SUMMARY
    {'='*50}
    
    Average Runtimes:
      • Baseline:      {avg_baseline:.2f} ms
      • Stream:        {avg_stream:.2f} ms
      • CUDA Graph:    {avg_graph:.2f} ms
    
    Average Speedups:
      • Stream:        {avg_stream_speedup:.2f}×
      • CUDA Graph:    {avg_graph_speedup:.2f}×
    
    Best Speedups:
      • Stream:        {max_stream_speedup:.2f}× at {nodes[max_stream_idx]//1000}K nodes
      • CUDA Graph:    {max_graph_speedup:.2f}× at {nodes[max_graph_idx]//1000}K nodes
    
    Configuration Count: {len(benchmark_data)} test cases
    Node Range: {min(nodes):,} - {max(nodes):,}
    Degree Range: {min(d for _, d, _, _, _ in benchmark_data)} - {max(d for _, d, _, _, _ in benchmark_data)}
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    fig.suptitle('BFS CUDA Implementation Benchmark Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('benchmark_comprehensive.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: benchmark_comprehensive.png")
    plt.close()


def print_summary_table():
    """Print formatted summary table"""
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    print(f"{'Nodes':<10} {'Degree':<8} {'Baseline':<12} {'Stream':<12} {'Graph':<12} {'Best':<10}")
    print("-"*80)
    
    for nodes, degree, baseline, stream, graph in benchmark_data:
        times = {'Baseline': baseline, 'Stream': stream, 'Graph': graph}
        best = min(times.keys(), key=lambda k: times[k])
        
        print(f"{nodes:<10,} {degree:<8} {baseline:<12.2f} {stream:<12.2f} {graph:<12.2f} {best:<10}")
    
    print("-"*80)
    
    # Calculate averages
    avg_baseline = np.mean([b for _, _, b, _, _ in benchmark_data])
    avg_stream = np.mean([s for _, _, _, s, _ in benchmark_data])
    avg_graph = np.mean([g for _, _, _, _, g in benchmark_data])
    
    print(f"{'AVERAGE':<10} {'':<8} {avg_baseline:<12.2f} {avg_stream:<12.2f} {avg_graph:<12.2f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    print("\n🚀 Generating BFS CUDA Benchmark Visualizations...\n")
    
    print_summary_table()
    
    print("Creating plots...")
    plot_by_degree()
    plot_by_nodes()
    plot_speedup_heatmap()
    plot_combined_comparison()
    
    print("\n✅ All visualizations generated successfully!")
    print("\nGenerated files:")
    print("  • benchmark_by_degree.png - Runtime vs nodes, separated by degree")
    print("  • benchmark_by_nodes.png - Runtime vs degree, separated by node count")
    print("  • speedup_heatmap.png - CUDA Graph speedup heatmap")
    print("  • benchmark_comprehensive.png - Comprehensive analysis dashboard")
    print()
