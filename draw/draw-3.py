import matplotlib.pyplot as plt
import numpy as np

def plot_fivedirections_bar_charts():
    """Tất cả biểu đồ thẳng cho FiveDirections dataset"""
    
    # Dữ liệu từ analysis_fivedirections.txt
    datasets = ['FiveDirections']
    auc = [0.7539]
    f1 = [0.0]
    precision = [1.0]
    recall = [0.0]
    
    # Confusion matrix data cho FiveDirections
    tn = 166962
    fp = 1
    fn = 202
    tp = 0
    
    # Setup plot
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 11, 'font.weight': 'bold'})
    
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('FiveDirections Dataset Performance Analysis - Bar Charts', fontsize=20, fontweight='bold', y=0.95)
    
    # 1. Main Performance Metrics (Top Left)
    ax1 = plt.subplot(2, 3, 1)
    metrics = ['AUC', 'F1-Score', 'Precision', 'Recall']
    values = [auc[0], f1[0], precision[0], recall[0]]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    bars = ax1.bar(metrics, values, color=colors, alpha=0.8)
    
    for bar, value in zip(bars, values):
        if value > 0.05:
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        else:
            ax1.text(bar.get_x() + bar.get_width()/2., 0.05,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_title('Performance Metrics', fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add reference lines
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random')
    ax1.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='Excellent')
    ax1.legend(fontsize=9)
    
    # 2. AUC Focus (Top Center)
    ax2 = plt.subplot(2, 3, 2)
    auc_breakdown = ['AUC Score', 'Remaining']
    auc_values = [auc[0], 1 - auc[0]]
    auc_colors = ['#2E86AB', '#E0E0E0']
    
    bars_auc = ax2.bar(auc_breakdown, auc_values, color=auc_colors, alpha=0.8)
    
    for bar, value in zip(bars_auc, auc_values):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax2.set_ylabel('Score', fontweight='bold')
    ax2.set_title('AUC Breakdown', fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Confusion Matrix Visualization (Top Right)
    ax3 = plt.subplot(2, 3, 3)
    confusion_data = [tn, fp, fn, tp]
    confusion_labels = ['True\nNegative', 'False\nPositive', 'False\nNegative', 'True\nPositive']
    colors_conf = ['green', 'orange', 'red', 'blue']
    
    bars_conf = ax3.bar(confusion_labels, confusion_data, color=colors_conf, alpha=0.8)
    ax3.set_ylabel('Count', fontweight='bold')
    ax3.set_title('Confusion Matrix', fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, value in zip(bars_conf, confusion_data):
        if value > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.1,
                    f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 4. Detection Analysis (Bottom Left)
    ax4 = plt.subplot(2, 3, 4)
    detection_metrics = ['Detection\nRate', 'False Alarm\nRate', 'Accuracy', 'Miss Rate']
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    miss_rate = fn / (tp + fn) if (tp + fn) > 0 else 1
    
    detection_values = [detection_rate, false_alarm_rate, accuracy, miss_rate]
    colors_det = ['blue', 'orange', 'green', 'red']
    
    bars_det = ax4.bar(detection_metrics, detection_values, color=colors_det, alpha=0.8)
    
    for bar, value in zip(bars_det, detection_values):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax4.set_ylabel('Rate', fontweight='bold')
    ax4.set_title('Detection Analysis', fontweight='bold')
    ax4.set_ylim(0, 1.1)
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Precision vs Recall Analysis (Bottom Center)
    ax5 = plt.subplot(2, 3, 5)
    pr_metrics = ['Precision', 'Recall']
    pr_values = [precision[0], recall[0]]
    pr_colors = ['#F18F01', '#C73E1D']
    
    bars_pr = ax5.bar(pr_metrics, pr_values, color=pr_colors, alpha=0.8, width=0.6)
    
    for bar, value in zip(bars_pr, pr_values):
        if value > 0.05:
            ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        else:
            ax5.text(bar.get_x() + bar.get_width()/2., 0.05,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax5.set_ylabel('Score', fontweight='bold')
    ax5.set_title('Precision vs Recall', fontweight='bold')
    ax5.set_ylim(0, 1.1)
    ax5.grid(axis='y', alpha=0.3)
    
    # Add trade-off indicator
    ax5.text(0.5, 0.8, 'High Precision\nZero Recall\n(Conservative Model)', 
             ha='center', va='center', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    # 6. Error Types Distribution (Bottom Right)
    ax6 = plt.subplot(2, 3, 6)
    error_types = ['False\nPositives', 'False\nNegatives', 'Correct\nPredictions']
    error_values = [fp, fn, tn + tp]
    error_colors = ['orange', 'red', 'green']
    
    bars_error = ax6.bar(error_types, error_values, color=error_colors, alpha=0.8)
    ax6.set_ylabel('Count', fontweight='bold')
    ax6.set_title('Prediction Distribution', fontweight='bold')
    ax6.set_yscale('log')
    ax6.grid(axis='y', alpha=0.3)
    
    for bar, value in zip(bars_error, error_values):
        if value > 0:
            ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.1,
                    f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('fivedirections_bar_charts.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_fivedirections_circular_charts():
    """Tất cả biểu đồ tròn cho FiveDirections dataset"""
    
    # Dữ liệu
    datasets = ['FiveDirections']
    auc = [0.7539]
    f1 = [0.0]
    precision = [1.0]
    recall = [0.0]
    
    # Confusion matrix
    tn, fp, fn, tp = 166962, 1, 202, 0
    
    # Setup plot
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 11, 'font.weight': 'bold'})
    
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('FiveDirections Dataset Performance Analysis - Circular Charts', fontsize=20, fontweight='bold', y=0.95)
    
    # 1. Performance Radar Chart (Top Left)
    ax1 = plt.subplot(2, 3, 1, projection='polar')
    
    # Create radar chart for metrics
    metrics = ['AUC', 'F1-Score', 'Precision', 'Recall']
    values = [auc[0], f1[0], precision[0], recall[0]]
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    values_radar = values + [values[0]]
    angles += angles[:1]
    
    ax1.plot(angles, values_radar, 'o-', linewidth=3, color='#E74C3C', 
            markersize=8, alpha=0.8)
    ax1.fill(angles, values_radar, alpha=0.25, color='#E74C3C')
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(metrics, fontsize=11)
    ax1.set_ylim(0, 1.0)
    ax1.set_title('Performance Radar', fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.4)
    
    # Add values
    for i, (angle, metric, value) in enumerate(zip(angles[:-1], metrics, values)):
        ax1.text(angle, value + 0.05, f'{value:.3f}', 
               ha='center', va='center', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8))
    
    # 2. Confusion Matrix Pie Chart (Top Center)
    ax2 = plt.subplot(2, 3, 2)
    conf_labels = ['True Negative', 'False Positive', 'False Negative']
    conf_values = [tn, fp, fn]
    conf_colors = ['green', 'orange', 'red']
    
    wedges, texts, autotexts = ax2.pie(conf_values, labels=conf_labels, 
                                       colors=conf_colors, autopct='%1.1f%%', startangle=90,
                                       textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax2.set_title('Confusion Matrix\nDistribution', fontweight='bold')
    
    # 3. Performance Level Gauge (Top Right)
    ax3 = plt.subplot(2, 3, 3, projection='polar')
    
    # Create gauge chart for AUC
    theta = np.linspace(0, np.pi, 100)
    auc_val = auc[0]
    
    # Background arc
    ax3.plot(theta, [1]*100, color='lightgray', linewidth=10, alpha=0.3)
    
    # Performance arc
    auc_theta = np.linspace(0, np.pi * auc_val, int(100 * auc_val))
    if auc_val >= 0.9:
        color = 'green'
    elif auc_val >= 0.7:
        color = 'orange'
    else:
        color = 'red'
    
    ax3.plot(auc_theta, [1]*len(auc_theta), color=color, linewidth=10, alpha=0.8)
    
    ax3.set_ylim(0, 1.2)
    ax3.set_theta_zero_location('W')
    ax3.set_theta_direction(1)
    ax3.set_thetagrids([0, 45, 90, 135, 180], ['1.0', '0.75', '0.5', '0.25', '0.0'])
    ax3.set_title('AUC Gauge', fontweight='bold', pad=20)
    ax3.grid(True, alpha=0.4)
    
    # Add AUC value
    ax3.text(np.pi/2, 0.5, f'{auc_val:.3f}', ha='center', va='center', 
            fontsize=16, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
    
    # 4. Detection Effectiveness Pie (Bottom Left)
    ax4 = plt.subplot(2, 3, 4)
    
    det_labels = ['Missed Detections (100%)', 'Successful Detections (0%)']
    det_values = [100, 0.001]  # Add tiny value to show in pie
    det_colors = ['red', 'green']
    
    wedges, texts, autotexts = ax4.pie([1], labels=['100% Missed\nDetections'], colors=['red'], 
                                      autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax4.set_title('Detection\nEffectiveness', fontweight='bold')
    
    # 5. Accuracy Breakdown (Bottom Center)
    ax5 = plt.subplot(2, 3, 5)
    
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total
    error_rate = (fp + fn) / total
    
    acc_labels = ['Correct Predictions', 'Incorrect Predictions']
    acc_values = [accuracy, error_rate]
    acc_colors = ['green', 'red']
    
    wedges, texts, autotexts = ax5.pie(acc_values, labels=acc_labels, 
                                       colors=acc_colors, autopct='%1.3f%%', startangle=90,
                                       textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax5.set_title('Overall\nAccuracy', fontweight='bold')
    
    # 6. Metric Balance Donut (Bottom Right)
    ax6 = plt.subplot(2, 3, 6)
    
    # Create donut showing metric balance
    balance_labels = ['AUC (Good)', 'Precision (Perfect)', 'Recall (Zero)']
    balance_values = [auc[0], precision[0] * 0.3, 0.001]  # Scale for visualization
    balance_colors = ['#2E86AB', '#27AE60', '#E74C3C']
    
    wedges, texts, autotexts = ax6.pie(balance_values, labels=balance_labels, 
                                       colors=balance_colors, autopct='%1.1f%%', startangle=90,
                                       textprops={'fontsize': 10, 'fontweight': 'bold'},
                                       pctdistance=0.85)
    
    # Add center circle for donut effect
    centre_circle = plt.Circle((0,0), 0.60, fc='white')
    ax6.add_artist(centre_circle)
    ax6.text(0, 0, 'Model\nCharacteristics', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    ax6.set_title('Performance\nBalance', fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('fivedirections_circular_charts.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Tạo cả 2 dashboard cho FiveDirections"""
    print("Generating FiveDirections performance dashboards...")
    
    # Tạo dashboard biểu đồ thẳng
    print("1. Creating FiveDirections bar charts dashboard...")
    plot_fivedirections_bar_charts()
    
    # Tạo dashboard biểu đồ tròn
    print("2. Creating FiveDirections circular charts dashboard...")
    plot_fivedirections_circular_charts()
    
    print("\nDashboards generated successfully!")
    print("Files saved:")
    print("- fivedirections_bar_charts.png")
    print("- fivedirections_circular_charts.png")
    
    # Print summary
    print(f"\nFiveDirections Performance Summary:")
    print("=" * 50)
    print(f"AUC:       0.7539 (75.39%)")
    print(f"F1-Score:  0.000  (0.00%)")
    print(f"Precision: 1.000  (100.00%)")
    print(f"Recall:    0.000  (0.00%)")
    print("=" * 50)
    print("Analysis: High precision but zero recall - very conservative model")

if __name__ == "__main__":
    main()