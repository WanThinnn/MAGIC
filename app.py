from flask import Flask, request, jsonify, send_file
import subprocess
import os
import threading
import time
import json
from datetime import datetime
import glob

app = Flask(__name__)

# Global variables để track training status
training_status = {
    'is_training': False,
    'progress': 0,
    'message': 'Ready',
    'start_time': None,
    'dataset': None
}

evaluation_status = {
    'is_evaluating': False,
    'progress': 0,
    'message': 'Ready',
    'result': None,
    'dataset': None
}

def run_training(dataset_name):
    """Function để chạy training trong background thread"""
    global training_status
    
    try:
        training_status['is_training'] = True
        training_status['progress'] = 10
        training_status['message'] = f'Khởi tạo training cho dataset {dataset_name}...'
        training_status['start_time'] = datetime.now()
        training_status['dataset'] = dataset_name
        
        # Chạy train.py với dataset parameter
        cmd = ['python', 'train.py', '--dataset', dataset_name]
        training_status['progress'] = 20
        training_status['message'] = 'Đang chạy training...'
        
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            encoding='utf-8',  # ← Thêm encoding
            errors='ignore',   # ← Thêm errors handling
            cwd=os.getcwd()
        )
        
        # Simulate progress updates
        for i in range(30, 90, 10):
            time.sleep(2)
            if training_status['is_training']:
                training_status['progress'] = i
                training_status['message'] = f'Training đang chạy... {i}%'
        
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            training_status['progress'] = 100
            training_status['message'] = 'Training hoàn thành thành công!'
        else:
            training_status['message'] = f'Training thất bại: {stderr}'
            
    except Exception as e:
        training_status['message'] = f'Lỗi: {str(e)}'
    finally:
        training_status['is_training'] = False

def run_evaluation(dataset_name):
    """Function để chạy evaluation trong background thread"""
    global evaluation_status
    
    try:
        evaluation_status['is_evaluating'] = True
        evaluation_status['progress'] = 10
        evaluation_status['message'] = f'Khởi tạo evaluation cho dataset {dataset_name}...'
        evaluation_status['dataset'] = dataset_name
        evaluation_status['result'] = None  # Reset result
        
        # Chạy eval.py với dataset parameter
        cmd = ['python', 'eval.py', '--dataset', dataset_name]
        evaluation_status['progress'] = 30
        evaluation_status['message'] = 'Đang chạy evaluation...'
        
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            encoding='utf-8',  # ← Thêm encoding
            errors='ignore',   # ← Thêm errors handling
            cwd=os.getcwd()
        )
        
        evaluation_status['progress'] = 70
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            # Parse toàn bộ kết quả từ output
            result_data = {}
            lines = stdout.split('\n')
            
            for line in lines:
                line = line.strip()
                if 'AUC:' in line and not line.startswith('#'):
                    result_data['auc'] = line.split('AUC:')[1].strip()
                elif 'F1:' in line:
                    result_data['f1'] = line.split('F1:')[1].strip()
                elif 'PRECISION:' in line:
                    result_data['precision'] = line.split('PRECISION:')[1].strip()
                elif 'RECALL:' in line:
                    result_data['recall'] = line.split('RECALL:')[1].strip()
                elif 'TN:' in line:
                    result_data['tn'] = line.split('TN:')[1].strip()
                elif 'FN:' in line:
                    result_data['fn'] = line.split('FN:')[1].strip()
                elif 'TP:' in line:
                    result_data['tp'] = line.split('TP:')[1].strip()
                elif 'FP:' in line:
                    result_data['fp'] = line.split('FP:')[1].strip()
                elif '#Test_AUC:' in line:
                    result_data['test_auc'] = line.strip()
            
            evaluation_status['result'] = result_data
            evaluation_status['progress'] = 90
            evaluation_status['message'] = 'Đang tạo biểu đồ visualization...'
            
            # Tự động tạo visualization sau khi evaluation hoàn thành
            try:
                viz_cmd = ['python', 'visualize_result.py']
                viz_process = subprocess.Popen(
                    viz_cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE, 
                    text=True,
                    encoding='utf-8',  # ← Thêm encoding
                    errors='ignore',   # ← Thêm errors handling
                    cwd=os.getcwd()
                )
                viz_stdout, viz_stderr = viz_process.communicate()
                
                if viz_process.returncode == 0:
                    evaluation_status['progress'] = 100
                    evaluation_status['message'] = 'Evaluation và visualization hoàn thành thành công!'
                else:
                    evaluation_status['progress'] = 100
                    evaluation_status['message'] = 'Evaluation hoàn thành! (Visualization có lỗi)'
            except Exception as viz_error:
                evaluation_status['progress'] = 100
                evaluation_status['message'] = f'Evaluation hoàn thành! (Lỗi visualization: {str(viz_error)})'
        else:
            evaluation_status['message'] = f'Evaluation thất bại: {stderr}'
            
    except Exception as e:
        evaluation_status['message'] = f'Lỗi: {str(e)}'
    finally:
        evaluation_status['is_evaluating'] = False

# Cập nhật valid_datasets để bao gồm 'fivedirections'
@app.route('/api/train', methods=['POST'])
def start_training():
    """API endpoint để bắt đầu training"""
    try:
        data = request.get_json()
        dataset_name = data.get('dataset', 'theia')
        
        if training_status['is_training']:
            return jsonify({
                'success': False,
                'error': 'Training đang chạy, vui lòng đợi!'
            }), 400
        
        # Kiểm tra dataset hợp lệ - thêm 'fivedirections'
        valid_datasets = ['theia', 'cadets', 'trace', 'streamspot', 'wget', 'fivedirections']
        if dataset_name not in valid_datasets:
            return jsonify({
                'success': False,
                'error': f'Dataset không hợp lệ. Chọn từ: {valid_datasets}'
            }), 400
        
        # Reset status
        training_status.update({
            'is_training': False,
            'progress': 0,
            'message': 'Ready',
            'start_time': None,
            'dataset': None
        })
        
        # Bắt đầu training trong background thread
        thread = threading.Thread(target=run_training, args=(dataset_name,))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': f'Bắt đầu training cho dataset {dataset_name}',
            'dataset': dataset_name
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Lỗi server: {str(e)}'
        }), 500

@app.route('/api/train/status', methods=['GET'])
def get_training_status():
    """API endpoint để lấy trạng thái training"""
    return jsonify({
        'success': True,
        'data': training_status  # ← Sửa từ 'status' thành 'data'
    })

@app.route('/api/eval', methods=['POST'])
def start_evaluation():
    """API endpoint để bắt đầu evaluation"""
    try:
        # Debug log
        print(f"Received eval request: {request.get_json()}")
        
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'Không có dữ liệu JSON trong request!'
            }), 400
            
        dataset_name = data.get('dataset', 'theia')
        print(f"Dataset được chọn: {dataset_name}")
        
        if evaluation_status['is_evaluating']:
            print("Evaluation đang chạy!")
            return jsonify({
                'success': False,
                'error': 'Evaluation đang chạy, vui lòng đợi!'
            }), 400
        
        # Kiểm tra dataset hợp lệ - thêm 'fivedirections'
        valid_datasets = ['theia', 'cadets', 'trace', 'streamspot', 'wget', 'fivedirections']
        if dataset_name not in valid_datasets:
            print(f"Dataset không hợp lệ: {dataset_name}")
            return jsonify({
                'success': False,
                'error': f'Dataset không hợp lệ. Chọn từ: {valid_datasets}'
            }), 400
        
        # Kiểm tra có model đã train chưa
        checkpoint_path = f"./checkpoints/checkpoint-{dataset_name}.pt"
        print(f"Kiểm tra checkpoint: {checkpoint_path}")
        print(f"Checkpoint exists: {os.path.exists(checkpoint_path)}")
        
        if not os.path.exists(checkpoint_path):
            print(f"Không tìm thấy model cho dataset: {dataset_name}")
            return jsonify({
                'success': False,
                'error': f'Chưa có model cho dataset {dataset_name}. Vui lòng train trước!'
            }), 400
        
        # Reset status
        evaluation_status.update({
            'is_evaluating': False,
            'progress': 0,
            'message': 'Ready',
            'result': None,
            'dataset': None
        })
        
        print(f"Bắt đầu evaluation cho dataset: {dataset_name}")
        
        # Bắt đầu evaluation trong background thread
        thread = threading.Thread(target=run_evaluation, args=(dataset_name,))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': f'Bắt đầu evaluation cho dataset {dataset_name}',
            'dataset': dataset_name
        })
        
    except Exception as e:
        print(f"Exception trong start_evaluation: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Lỗi server: {str(e)}'
        }), 500

@app.route('/api/eval/status', methods=['GET'])
def get_evaluation_status():
    """API endpoint để lấy trạng thái evaluation"""
    return jsonify({
        'success': True,
        'data': evaluation_status  # ← Sửa từ 'status' thành 'data'
    })

@app.route('/api/datasets', methods=['GET'])
def get_available_datasets():
    """API endpoint để lấy danh sách datasets có sẵn"""
    # Danh sách datasets được hỗ trợ bởi MAGIC
    datasets = ['cadets', 'fivedirections', 'streamspot', 'theia', 'trace', 'wget']
    
    return jsonify({
        'success': True,
        'data': datasets  # ← Sửa từ 'datasets' thành 'data'
    })

@app.route('/api/models', methods=['GET'])
def get_trained_models():
    """API endpoint để lấy danh sách models đã train"""
    models = []
    checkpoint_dir = './checkpoints'
    
    if os.path.exists(checkpoint_dir):
        for file in os.listdir(checkpoint_dir):
            if file.startswith('checkpoint-') and file.endswith('.pt'):
                dataset_name = file.replace('checkpoint-', '').replace('.pt', '')
                file_path = os.path.join(checkpoint_dir, file)
                file_size = os.path.getsize(file_path)
                modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                models.append({
                    'dataset': dataset_name,
                    'file_size': file_size,
                    'modified_time': modified_time.isoformat(),
                    'file_path': file
                })
    
    return jsonify({
        'success': True,
        'data': models  # ← Sửa từ 'models' thành 'data'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'message': 'MAGIC API is running',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/eval/report', methods=['GET'])
def generate_eval_report():
    """
    Gọi visualize_result.py để vẽ biểu đồ sau khi evaluation hoàn tất
    """
    try:
        # Gọi script visualize_result.py với encoding handling
        process = subprocess.Popen(
            ['python', 'visualize_result.py'], 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            return jsonify({
                'success': True,
                'message': 'Đã chạy visualize_result.py để tạo biểu đồ',
                'output': stdout
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Lỗi khi gọi visualize_result.py: {stderr}'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Lỗi khi gọi visualize_result.py: {str(e)}'
        }), 500

@app.route('/api/visualize', methods=['POST'])
def generate_visualization():
    """API endpoint để tạo biểu đồ visualization"""
    try:
        # Kiểm tra có kết quả evaluation không
        if not evaluation_status['result']:
            return jsonify({
                'success': False,
                'error': 'Chưa có kết quả evaluation để visualize!'
            }), 400
        
        # Gọi script visualize_result.py
        cmd = ['python', 'visualize_result.py']
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            encoding='utf-8',  # ← Thêm encoding
            errors='ignore',   # ← Thêm errors handling
            cwd=os.getcwd()
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            return jsonify({
                'success': True,
                'message': 'Đã tạo biểu đồ visualization thành công!',
                'output': stdout
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Lỗi khi tạo biểu đồ: {stderr}'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Lỗi server: {str(e)}'
        }), 500

@app.route('/api/checkpoints', methods=['GET'])
def list_checkpoints():
    """Debug endpoint để kiểm tra checkpoints có sẵn"""
    try:
        checkpoint_dir = './checkpoints'
        checkpoints = []
        
        if os.path.exists(checkpoint_dir):
            for file in os.listdir(checkpoint_dir):
                if file.endswith('.pt'):
                    file_path = os.path.join(checkpoint_dir, file)
                    checkpoints.append({
                        'filename': file,
                        'path': file_path,
                        'exists': os.path.exists(file_path),
                        'size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
                    })
        
        return jsonify({
            'success': True,
            'data': {
                'checkpoint_dir': checkpoint_dir,
                'dir_exists': os.path.exists(checkpoint_dir),
                'checkpoints': checkpoints
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/latest-result-image', methods=['GET'])
def get_latest_result_image():
    """API endpoint để lấy file hình kết quả mới nhất"""
    try:
        # Tìm file .png mới nhất có pattern magic_results_*
        pattern = './magic_results_*.png'
        files = glob.glob(pattern)
        
        if not files:
            return jsonify({
                'success': False,
                'error': 'Không tìm thấy file kết quả nào!'
            }), 404
        
        # Lấy file mới nhất
        latest_file = max(files, key=os.path.getctime)
        
        if os.path.exists(latest_file):
            return send_file(latest_file, mimetype='image/png')
        else:
            return jsonify({
                'success': False,
                'error': 'File không tồn tại!'
            }), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Lỗi server: {str(e)}'
        }), 500

@app.route('/api/list-result-images', methods=['GET'])
def list_result_images():
    """API endpoint để list tất cả file kết quả"""
    try:
        pattern = './magic_results_*.png'
        files = glob.glob(pattern)
        
        result_files = []
        for file in files:
            if os.path.exists(file):
                stat = os.stat(file)
                result_files.append({
                    'filename': os.path.basename(file),
                    'path': file,
                    'size': stat.st_size,
                    'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        # Sắp xếp theo thời gian tạo (mới nhất trước)
        result_files.sort(key=lambda x: x['created'], reverse=True)
        
        return jsonify({
            'success': True,
            'data': result_files
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Tạo thư mục checkpoints nếu chưa có
    os.makedirs('./checkpoints', exist_ok=True)
    
    print("MAGIC API Server đang khởi động...")
    print("Available endpoints:")
    print("   - POST /api/train (Bắt đầu training)")
    print("   - GET /api/train/status (Trạng thái training)")
    print("   - POST /api/eval (Bắt đầu evaluation)")
    print("   - GET /api/eval/status (Trạng thái evaluation)")
    print("   - POST /api/visualize (Tạo biểu đồ visualization)")
    print("   - GET /api/datasets (Danh sách datasets)")
    print("   - GET /api/models (Danh sách models đã train)")
    print("   - GET /health (Health check)")
    
    app.run(debug=True, host='0.0.0.0', port=5000)