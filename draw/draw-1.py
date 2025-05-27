import matplotlib.pyplot as plt
import numpy as np

# Set style for better poster presentation
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'font.weight': 'bold'})

# Biểu đồ 1: Datasets của tác giả gốc
def plot_author_datasets():
    datasets = ['StreamSpot', 'Wget', 'Trace', 'Theia', 'Cadets']
    auc = [0.9995, 0.9739, 0.9998, 0.9987, 0.9977]
    f1 = [0.9954, 0.9436, 0.9957, 0.9911, 0.9701]
    precision = [0.9920, 0.9139, 0.9917, 0.9823, 0.9441]
    recall = [0.9990, 0.9776, 0.9998, 0.9996, 0.9977]

    x = np.arange(len(datasets))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars1 = ax.bar(x - 1.5*width, auc, width, label='AUC', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x - 0.5*width, f1, width, label='F1-Score', color='#A23B72', alpha=0.8)
    bars3 = ax.bar(x + 0.5*width, precision, width, label='Precision', color='#F18F01', alpha=0.8)
    bars4 = ax.bar(x + 1.5*width, recall, width, label='Recall', color='#C73E1D', alpha=0.8)

    # Thêm giá trị trên cột
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance Score', fontsize=14, fontweight='bold')
    ax.set_ylim(0.9, 1.05)
    ax.set_title('MAGIC Performance on Standard Datasets', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('magic_author_datasets.png', dpi=300, bbox_inches='tight')
    plt.show()

# Biểu đồ 2: Dataset của bạn (FiveDirections)
def plot_custom_dataset():
    datasets = ['FiveDirections']
    auc = [0.7539]  # Từ analysis_fivedirections.txt
    
    # Nếu có thêm metrics cho FiveDirections, thêm vào đây
    # f1 = [0.xxxx]
    # precision = [0.xxxx] 
    # recall = [0.xxxx]

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Chỉ vẽ AUC nếu không có metrics khác
    bars = ax.bar(datasets, auc, color='#E74C3C', alpha=0.8, width=0.5)
    
    # Thêm giá trị
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.4f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('AUC Score', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_title('MAGIC Performance on Custom Dataset', fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)

    # Thêm benchmark line
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random Classifier')
    ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='Excellent Threshold')
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('magic_custom_dataset.png', dpi=300, bbox_inches='tight')
    plt.show()

# Biểu đồ so sánh tổng quan (Alternative)
def plot_comparison_summary():
    fig, (ax1) = plt.subplots(1, figsize=(25, 6))
    
    # Subplot 1: Author datasets với 4 metrics
    datasets_author = ['StreamSpot', 'Wget', 'Trace', 'Theia', 'Cadets']
    auc_author = [0.9995, 0.9739, 0.9998, 0.9987, 0.9977]
    f1_author = [0.9954, 0.9436, 0.9957, 0.9911, 0.9701]
    precision_author = [0.9920, 0.9139, 0.9917, 0.9823, 0.9441]
    recall_author = [0.9990, 0.9776, 0.9998, 0.9996, 0.9977]
    
    x = np.arange(len(datasets_author))
    width = 0.2
    
    bars1 = ax1.bar(x - 1.5*width, auc_author, width, label='AUC', color='#2E86AB', alpha=0.8)
    bars2 = ax1.bar(x - 0.5*width, f1_author, width, label='F1-Score', color='#A23B72', alpha=0.8)
    bars3 = ax1.bar(x + 0.5*width, precision_author, width, label='Precision', color='#F18F01', alpha=0.8)
    bars4 = ax1.bar(x + 1.5*width, recall_author, width, label='Recall', color='#C73E1D', alpha=0.8)
    
    # Thêm giá trị trên cột cho subplot 1
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets_author, fontsize=12, fontweight='bold')
    ax1.set_title('Standard Datasets Performance', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Performance Score', fontsize=12, fontweight='bold')
    ax1.set_ylim(0.9, 1.05)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # # Subplot 2: Custom dataset với metrics có sẵn
    # datasets_custom = ['FiveDirections']
    # auc_custom = [0.7539]
    # f1_custom = [0.0]        # Từ analysis_fivedirections.txt
    # precision_custom = [1.0]  # Từ analysis_fivedirections.txt  
    # recall_custom = [0.0]     # Từ analysis_fivedirections.txt
    
    # x_custom = np.arange(len(datasets_custom))
    
    # bars1_custom = ax2.bar(x_custom - 1.5*width, auc_custom, width, label='AUC', color='#2E86AB', alpha=0.8)
    # bars2_custom = ax2.bar(x_custom - 0.5*width, f1_custom, width, label='F1-Score', color='#A23B72', alpha=0.8)
    # bars3_custom = ax2.bar(x_custom + 0.5*width, precision_custom, width, label='Precision', color='#F18F01', alpha=0.8)
    # bars4_custom = ax2.bar(x_custom + 1.5*width, recall_custom, width, label='Recall', color='#C73E1D', alpha=0.8)
    
    # # Thêm giá trị cho subplot 2
    # values_custom = [auc_custom, f1_custom, precision_custom, recall_custom]
    # for bars, values in zip([bars1_custom, bars2_custom, bars3_custom, bars4_custom], values_custom):
    #     for bar, value in zip(bars, values):
    #         if value > 0.05:  # Chỉ hiển thị text nếu cột đủ cao
    #             ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
    #                     f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    #         elif value > 0:  # Cho các giá trị nhỏ, hiển thị bên cạnh
    #             ax2.text(bar.get_x() + bar.get_width() + 0.05, bar.get_height()/2,
    #                     f'{value:.3f}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    # ax2.set_xticks(x_custom)
    # ax2.set_xticklabels(datasets_custom, fontsize=12, fontweight='bold')
    # ax2.set_title('Custom Dataset Performance', fontsize=14, fontweight='bold')
    # ax2.set_ylabel('Performance Score', fontsize=12, fontweight='bold')
    # ax2.set_ylim(0, 1.1)
    # ax2.legend(loc='upper right', fontsize=10)
    # ax2.grid(axis='y', alpha=0.3)
    
    # # Thêm reference lines cho custom dataset
    # ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    # ax2.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, linewidth=1)
    
    # # Thêm text cho reference lines
    # ax2.text(0.02, 0.52, 'Random', fontsize=9, alpha=0.7)
    # ax2.text(0.02, 0.92, 'Excellent', fontsize=9, alpha=0.7)
    
    plt.suptitle('MAGIC Model Performance: Standard vs Custom Datasets', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('magic_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# Chạy các hàm
if __name__ == "__main__":
    # print("Generating performance charts for poster...")
    
    # plot_author_datasets()
    # plot_custom_dataset() 
    plot_comparison_summary()
    
    # print("✅ All charts generated successfully!")
    # print("📊 Files saved:")
    # print("   - magic_author_datasets.png")
    # print("   - magic_custom_dataset.png") 
    # print("   - magic_performance_comparison.png")