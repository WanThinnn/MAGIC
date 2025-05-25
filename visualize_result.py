import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # hoặc 'Qt5Agg'
import seaborn as sns
import numpy as np
import requests
import json
from datetime import datetime
import os
import subprocess
API_BASE_URL = "http://localhost:5000"

def get_evaluation_result():
    """Lấy kết quả evaluation từ API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/eval/status")
        if response.status_code == 200:
            data = response.json()
            if data['success'] and data['data']['result']:
                return data['data']['result']
        return None
    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu từ API: {e}")
        return None

def parse_metrics(result_data):
    """Parse metrics từ API response"""
    try:
        metrics = {}
        
        # Parse các giá trị số
        if 'auc' in result_data:
            metrics['auc'] = float(result_data['auc'])
        if 'f1' in result_data:
            metrics['f1'] = float(result_data['f1'])
        if 'precision' in result_data:
            metrics['precision'] = float(result_data['precision'])
        if 'recall' in result_data:
            metrics['recall'] = float(result_data['recall'])
        
        # Parse confusion matrix values
        if 'tn' in result_data:
            metrics['tn'] = int(result_data['tn'])
        if 'fp' in result_data:
            metrics['fp'] = int(result_data['fp'])
        if 'fn' in result_data:
            metrics['fn'] = int(result_data['fn'])
        if 'tp' in result_data:
            metrics['tp'] = int(result_data['tp'])
        
        # Parse test AUC
        if 'test_auc' in result_data:
            metrics['test_auc'] = result_data['test_auc']
        
        return metrics
    except Exception as e:
        print(f"Lỗi parse metrics: {e}")
        return None

def visualize_magic_results(metrics=None, dataset_name="Unknown"):
    """Tạo dashboard visualization cho kết quả MAGIC"""
    
    # Nếu không có metrics, dùng dữ liệu mẫu
    if metrics is None:
        print("Không có dữ liệu từ API, sử dụng dữ liệu mẫu (Cadets dataset)")
        metrics = {
            'auc': 0.9977,
            'f1': 0.9701,
            'precision': 0.9441,
            'recall': 0.9977,
            'tn': 343568,
            'fp': 759,
            'fn': 30,
            'tp': 12816,
            'test_auc': '#Test_AUC: 0.9977±0.0000'
        }
        dataset_name = "Cadets (Sample)"
    
    # Extract values
    TN = metrics.get('tn', 0)
    FP = metrics.get('fp', 0)
    FN = metrics.get('fn', 0)
    TP = metrics.get('tp', 0)
    
    performance_metrics = {
        'AUC': metrics.get('auc', 0),
        'F1-Score': metrics.get('f1', 0),
        'Precision': metrics.get('precision', 0),
        'Recall': metrics.get('recall', 0)
    }
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'MAGIC Model Performance Analysis - {dataset_name} Dataset', 
                 fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix
    plt.subplot(3, 3, 1)
    cm = np.array([[TN, FP], [FN, TP]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Malicious'],
                yticklabels=['Normal', 'Malicious'])
    plt.title('Confusion Matrix')
    
    # 2. Performance Metrics Bar Chart
    plt.subplot(3, 3, 2)
    bars = plt.bar(performance_metrics.keys(), performance_metrics.values(), 
                   color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
    for bar, value in zip(bars, performance_metrics.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    plt.ylim(0, 1.1)
    plt.title('Performance Metrics')
    plt.xticks(rotation=45)
    
    # 3. Error Analysis Pie Chart
    plt.subplot(3, 3, 3)
    if FP + FN > 0:
        error_data = [FP, FN]
        error_labels = ['False Positive', 'False Negative']
        colors = ['orange', 'red']
        plt.pie(error_data, labels=error_labels, autopct='%1.1f%%', 
                colors=colors, startangle=90)
    else:
        plt.text(0.5, 0.5, 'No Errors!', ha='center', va='center', 
                transform=plt.gca().transAxes, fontsize=12, fontweight='bold')
    plt.title('Error Distribution')
    
    # 4. Prediction Distribution (Log Scale)
    plt.subplot(3, 3, 4)
    categories = ['TN', 'FP', 'FN', 'TP']
    values = [TN, FP, FN, TP]
    colors = ['lightgreen', 'orange', 'red', 'darkgreen']
    bars = plt.bar(categories, values, color=colors)
    plt.yscale('log')
    plt.title('Prediction Counts (Log Scale)')
    for bar, value in zip(bars, values):
        if value > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{value:,}', ha='center', va='bottom', fontsize=8)
    
    # 5. Precision vs Recall Scatter
    plt.subplot(3, 3, 5)
    precision = performance_metrics['Precision']
    recall = performance_metrics['Recall']
    plt.scatter([recall], [precision], s=200, c='red', alpha=0.7, edgecolors='black')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall')
    plt.text(recall+0.02, precision-0.02, f'({recall:.3f}, {precision:.3f})', 
             fontsize=9)
    plt.grid(alpha=0.3)
    
    # 6. Accuracy Breakdown
    plt.subplot(3, 3, 6)
    total = TN + FP + FN + TP
    accuracy = (TN + TP) / total if total > 0 else 0
    categories = ['Correct', 'Incorrect']
    values = [TN + TP, FP + FN]
    colors = ['lightgreen', 'lightcoral']
    plt.pie(values, labels=categories, autopct='%1.2f%%', colors=colors, startangle=90)
    plt.title(f'Overall Accuracy: {accuracy:.4f}')
    
    # 7. ROC Curve Simulation
    plt.subplot(3, 3, 7)
    # Simulate ROC curve points based on confusion matrix
    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    
    # Create smooth ROC curve
    fpr_points = np.linspace(0, 1, 100)
    tpr_points = []
    for f in fpr_points:
        if f <= fpr:
            t = (f / fpr) * tpr if fpr > 0 else 0
        else:
            t = tpr + (1 - tpr) * ((f - fpr) / (1 - fpr)) if fpr < 1 else tpr
        tpr_points.append(min(t, 1.0))
    
    plt.plot(fpr_points, tpr_points, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {performance_metrics["AUC"]:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.scatter([fpr], [tpr], color='red', s=100, zorder=5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    
    # 8. Detection Rate Analysis
    plt.subplot(3, 3, 8)
    detection_data = {
        'Malicious\nDetected': TP,
        'Malicious\nMissed': FN,
        'Normal\nCorrect': TN,
        'Normal\nFalse Alarm': FP
    }
    
    bars = plt.bar(detection_data.keys(), detection_data.values(), 
                   color=['darkgreen', 'red', 'lightgreen', 'orange'])
    plt.yscale('log')
    plt.title('Detection Breakdown')
    plt.xticks(rotation=45)
    for bar, value in zip(bars, detection_data.values()):
        if value > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{value:,}', ha='center', va='bottom', fontsize=8)
    
    # 9. Model Summary
    plt.subplot(3, 3, 9)
    plt.axis('off')
    summary_text = f"""
Model: MAGIC
Dataset: {dataset_name}

Total Samples: {total:,}
Malicious: {FN+TP:,}
Normal: {TN+FP:,}

Key Metrics:
• Accuracy: {accuracy:.4f}
• AUC: {performance_metrics['AUC']:.4f}
• F1-Score: {performance_metrics['F1-Score']:.4f}
• Precision: {performance_metrics['Precision']:.4f}
• Recall: {performance_metrics['Recall']:.4f}

Confusion Matrix:
• True Negatives: {TN:,}
• False Positives: {FP:,}
• False Negatives: {FN:,}
• True Positives: {TP:,}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    plt.text(0.05, 0.8, summary_text, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
             transform=plt.gca().transAxes)
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'magic_results_{dataset_name.lower().replace(" ", "_")}_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Biểu đồ đã được lưu: {filename}")
    
    # Return filename for API response
    return filename

def main():
    """Main function để chạy visualization"""
    print("Đang lấy kết quả evaluation từ API...")
    
    result_data = get_evaluation_result()
    
    if result_data:
        print("Đã lấy được dữ liệu từ API")
        metrics = parse_metrics(result_data)
        
        if metrics:
            print("Đang tạo biểu đồ...")
            try:
                response = requests.get(f"{API_BASE_URL}/api/eval/status")
                if response.status_code == 200:
                    data = response.json()
                    dataset_name = data['data'].get('dataset', 'Unknown')
                else:
                    dataset_name = 'Unknown'
            except:
                dataset_name = 'Unknown'
            
            filename = visualize_magic_results(metrics, dataset_name.title())
            print(f"✅ Đã tạo file: {filename}")
        else:
            print("Không thể parse metrics từ API")
            filename = visualize_magic_results()
    else:
        print("Không thể lấy dữ liệu từ API, sử dụng dữ liệu mẫu")
        filename = visualize_magic_results()

if __name__ == "__main__":
    main()