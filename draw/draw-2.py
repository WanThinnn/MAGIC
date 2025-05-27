import matplotlib.pyplot as plt
import numpy as np

def plot_all_bar_charts():
    """Tất cả biểu đồ thẳng trong 1 hình"""
    
    # Dữ liệu
    datasets = ['StreamSpot', 'Wget', 'Trace', 'Theia', 'Cadets']
    auc = [0.9995, 0.9739, 0.9998, 0.9987, 0.9977]
    f1 = [0.9954, 0.9436, 0.9957, 0.9911, 0.9701]
    precision = [0.9920, 0.9139, 0.9917, 0.9823, 0.9441]
    recall = [0.9990, 0.9776, 0.9998, 1.0000, 0.9977]
    
    dataset_types = ['Batch-level', 'Batch-level', 'Entity-level', 'Entity-level', 'Entity-level']
    
    # Setup plot
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 11, 'font.weight': 'bold'})
    
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('MAGIC Performance Analysis - Bar Charts', fontsize=20, fontweight='bold', y=0.95)
    
    # 1. Main Performance Metrics (Top)
    ax1 = plt.subplot(2, 3, (1, 2))  # Span 2 columns
    x = np.arange(len(datasets))
    width = 0.2
    
    bars1 = ax1.bar(x - 1.5*width, auc, width, label='AUC', color='#2E86AB', alpha=0.8)
    bars2 = ax1.bar(x - 0.5*width, f1, width, label='F1-Score', color='#A23B72', alpha=0.8)
    bars3 = ax1.bar(x + 0.5*width, precision, width, label='Precision', color='#F18F01', alpha=0.8)
    bars4 = ax1.bar(x + 1.5*width, recall, width, label='Recall', color='#C73E1D', alpha=0.8)
    
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.003,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, fontsize=12)
    ax1.set_ylabel('Performance Score', fontsize=12, fontweight='bold')
    ax1.set_ylim(0.9, 1.05)
    ax1.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add dataset type annotations
    for i, dtype in enumerate(dataset_types):
        color = '#FF6B6B' if dtype == 'Batch-level' else '#4ECDC4'
        ax1.text(i, 0.915, dtype, ha='center', va='center', 
               fontsize=9, style='italic', color=color,
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # 2. AUC Comparison (Top Right)
    ax2 = plt.subplot(2, 3, 3)
    bars_auc = ax2.bar(datasets, auc, color=['#2E86AB', '#2E86AB', '#17A2B8', '#17A2B8', '#17A2B8'], alpha=0.8)
    
    for bar, value in zip(bars_auc, auc):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax2.set_ylabel('AUC Score', fontweight='bold')
    ax2.set_title('AUC Performance', fontweight='bold')
    ax2.set_ylim(0.97, 1.0)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Performance by Dataset Type (Bottom Left)
    ax3 = plt.subplot(2, 3, 4)
    batch_indices = [0, 1]
    entity_indices = [2, 3, 4]
    
    batch_avg_auc = np.mean([auc[i] for i in batch_indices])
    entity_avg_auc = np.mean([auc[i] for i in entity_indices])
    batch_avg_f1 = np.mean([f1[i] for i in batch_indices])
    entity_avg_f1 = np.mean([f1[i] for i in entity_indices])
    
    types = ['Batch-level', 'Entity-level']
    auc_avgs = [batch_avg_auc, entity_avg_auc]
    f1_avgs = [batch_avg_f1, entity_avg_f1]
    
    x_type = np.arange(len(types))
    width_type = 0.35
    
    bars_auc_avg = ax3.bar(x_type - width_type/2, auc_avgs, width_type, label='AUC', color='#2E86AB', alpha=0.8)
    bars_f1_avg = ax3.bar(x_type + width_type/2, f1_avgs, width_type, label='F1', color='#A23B72', alpha=0.8)
    
    for bars in [bars_auc_avg, bars_f1_avg]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax3.set_xticks(x_type)
    ax3.set_xticklabels(types)
    ax3.set_ylabel('Average Score', fontweight='bold')
    ax3.set_title('Performance by Type', fontweight='bold')
    ax3.legend()
    ax3.set_ylim(0.94, 1.0)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Error Analysis (Bottom Center)
    ax4 = plt.subplot(2, 3, 5)
    # Simulated error data
    total_fp = 1784  # Sum of false positives
    total_fn = 201   # Sum of false negatives
    error_types = ['False\nPositives', 'False\nNegatives']
    error_values = [total_fp, total_fn]
    colors_error = ['orange', 'red']
    
    bars_error = ax4.bar(error_types, error_values, color=colors_error, alpha=0.8)
    ax4.set_ylabel('Total Errors', fontweight='bold')
    ax4.set_title('Error Distribution', fontweight='bold')
    ax4.set_yscale('log')
    ax4.grid(axis='y', alpha=0.3)
    
    for bar, value in zip(bars_error, error_values):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.1,
                f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 5. Precision vs Recall (Bottom Right)
    ax5 = plt.subplot(2, 3, 6)
    colors_scatter = ['#FF6B6B', '#FF6B6B', '#4ECDC4', '#4ECDC4', '#4ECDC4']
    scatter = ax5.scatter(recall, precision, s=120, alpha=0.8, c=colors_scatter, 
                         edgecolors='black', linewidth=1)
    
    for i, dataset in enumerate(datasets):
        ax5.annotate(dataset, (recall[i], precision[i]), 
                    xytext=(8, 8), textcoords='offset points', fontsize=9, fontweight='bold')
    
    ax5.set_xlabel('Recall', fontweight='bold')
    ax5.set_ylabel('Precision', fontweight='bold')
    ax5.set_title('Precision vs Recall', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0.9, 1.01)
    ax5.set_ylim(0.9, 1.01)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('magic_all_bar_charts.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_all_circular_charts():
    """Tất cả biểu đồ tròn trong 1 hình"""
    
    # Dữ liệu
    datasets = ['StreamSpot', 'Wget', 'Trace', 'Theia', 'Cadets']
    auc = [0.9995, 0.9739, 0.9998, 0.9987, 0.9977]
    f1 = [0.9954, 0.9436, 0.9957, 0.9911, 0.9701]
    precision = [0.9920, 0.9139, 0.9917, 0.9823, 0.9441]
    recall = [0.9990, 0.9776, 0.9998, 1.0000, 0.9977]
    
    # Setup plot
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 11, 'font.weight': 'bold'})
    
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('MAGIC Performance Analysis - Circular Charts', fontsize=20, fontweight='bold', y=0.95)
    
    # 1. Main Radar Chart (Top Left)
    ax1 = plt.subplot(2, 3, 1, projection='polar')
    angles = np.linspace(0, 2 * np.pi, len(datasets), endpoint=False).tolist()
    
    auc_radar = auc + [auc[0]]
    f1_radar = f1 + [f1[0]]
    precision_radar = precision + [precision[0]]
    recall_radar = recall + [recall[0]]
    angles += angles[:1]
    
    ax1.plot(angles, auc_radar, 'o-', linewidth=3, color='#2E86AB', 
            markersize=6, label='AUC', alpha=0.8)
    ax1.fill(angles, auc_radar, alpha=0.15, color='#2E86AB')
    
    ax1.plot(angles, f1_radar, 's-', linewidth=3, color='#A23B72', 
            markersize=6, label='F1-Score', alpha=0.8)
    ax1.fill(angles, f1_radar, alpha=0.15, color='#A23B72')
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(datasets, fontsize=10)
    ax1.set_ylim(0.9, 1.0)
    ax1.set_title('Performance Radar\n(AUC & F1)', fontweight='bold', pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=9)
    ax1.grid(True, alpha=0.4)
    
    # 2. AUC Only Radar (Top Center)
    ax2 = plt.subplot(2, 3, 2, projection='polar')
    ax2.plot(angles, auc_radar, 'o-', linewidth=4, color='#2E86AB', 
            markersize=8, alpha=0.9)
    ax2.fill(angles, auc_radar, alpha=0.3, color='#2E86AB')
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(datasets, fontsize=10)
    ax2.set_ylim(0.97, 1.0)
    ax2.set_title('AUC Performance\nRadar', fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.4)
    
    # Add AUC values
    for i, (angle, dataset) in enumerate(zip(angles[:-1], datasets)):
        ax2.text(angle, auc[i] + 0.003, f'{auc[i]:.3f}', 
               ha='center', va='center', fontsize=8, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8))
    
    # 3. Dataset Type Distribution (Top Right)
    ax3 = plt.subplot(2, 3, 3)
    type_counts = {'Batch-level': 2, 'Entity-level': 3}
    colors = ['#FF6B6B', '#4ECDC4']
    wedges, texts, autotexts = ax3.pie(type_counts.values(), labels=type_counts.keys(), 
                                       colors=colors, autopct='%1.0f%%', startangle=90,
                                       textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax3.set_title('Dataset Types\nDistribution', fontweight='bold')
    
    # 4. Performance Distribution (Bottom Left)
    ax4 = plt.subplot(2, 3, 4)
    # Performance level distribution
    excellent = sum(1 for x in auc if x >= 0.995)  # 3 datasets
    very_good = sum(1 for x in auc if 0.99 <= x < 0.995)  # 1 dataset
    good = sum(1 for x in auc if x < 0.99)  # 1 dataset
    
    perf_labels = ['Excellent\n(≥99.5%)', 'Very Good\n(99-99.5%)', 'Good\n(<99%)']
    perf_values = [excellent, very_good, good]
    colors_perf = ['#2E8B57', '#32CD32', '#FFD700']
    
    wedges, texts, autotexts = ax4.pie(perf_values, labels=perf_labels, 
                                       colors=colors_perf, autopct='%1.0f%%', startangle=90,
                                       textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax4.set_title('AUC Performance\nLevels', fontweight='bold')
    
    # 5. Precision vs Recall Radar (Bottom Center)
    ax5 = plt.subplot(2, 3, 5, projection='polar')
    ax5.plot(angles, precision_radar, '^-', linewidth=3, color='#F18F01', 
            markersize=6, label='Precision', alpha=0.8)
    ax5.fill(angles, precision_radar, alpha=0.15, color='#F18F01')
    
    ax5.plot(angles, recall_radar, 'D-', linewidth=3, color='#C73E1D', 
            markersize=6, label='Recall', alpha=0.8)
    ax5.fill(angles, recall_radar, alpha=0.15, color='#C73E1D')
    
    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(datasets, fontsize=10)
    ax5.set_ylim(0.9, 1.0)
    ax5.set_title('Precision vs Recall\nRadar', fontweight='bold', pad=20)
    ax5.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=9)
    ax5.grid(True, alpha=0.4)
    
    # 6. Metric Comparison Donut (Bottom Right)
    ax6 = plt.subplot(2, 3, 6)
    avg_metrics = [np.mean(auc), np.mean(f1), np.mean(precision), np.mean(recall)]
    metric_labels = ['AUC', 'F1-Score', 'Precision', 'Recall']
    colors_metrics = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # Create donut chart
    wedges, texts, autotexts = ax6.pie(avg_metrics, labels=metric_labels, 
                                       colors=colors_metrics, autopct='%1.3f', startangle=90,
                                       textprops={'fontsize': 10, 'fontweight': 'bold'},
                                       pctdistance=0.85)
    
    # Add center circle for donut effect
    centre_circle = plt.Circle((0,0), 0.60, fc='white')
    ax6.add_artist(centre_circle)
    ax6.text(0, 0, 'Average\nMetrics', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    ax6.set_title('Average Performance\nMetrics', fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('magic_all_circular_charts.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Tạo cả 2 hình dashboard"""
    print("Generating MAGIC performance dashboards...")
    
    # Tạo dashboard biểu đồ thẳng
    print("1. Creating all bar charts dashboard...")
    plot_all_bar_charts()
    
    # Tạo dashboard biểu đồ tròn
    print("2. Creating all circular charts dashboard...")
    plot_all_circular_charts()
    
    print("\nDashboards generated successfully!")
    print("Files saved:")
    print("- magic_all_bar_charts.png")
    print("- magic_all_circular_charts.png")
    
    # Print summary
    datasets = ['StreamSpot', 'Wget', 'Trace', 'Theia', 'Cadets']
    auc = [0.9995, 0.9739, 0.9998, 0.9987, 0.9977]
    
    print(f"\nPerformance Summary:")
    print("=" * 50)
    for i, dataset in enumerate(datasets):
        print(f"{dataset:12} | AUC: {auc[i]:.4f}")
    print("=" * 50)
    print(f"Average AUC: {np.mean(auc):.4f}")

if __name__ == "__main__":
    main()