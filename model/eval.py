"""
Module đánh giá mô hình phát hiện bất thường sử dụng phương pháp KNN
"""

import os
import random
import time
import pickle as pkl
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.neighbors import NearestNeighbors
from utils.utils import set_random_seed
from utils.loaddata import transform_graph, load_batch_level_dataset


def batch_level_evaluation(model, pooler, device, method, dataset, n_dim=0, e_dim=0):
    """
    Đánh giá mô hình ở mức batch (tập hợp các đồ thị)
    
    Parameters:
    -----------
    model : torch.nn.Module
        Mô hình cần đánh giá
    pooler : callable
        Hàm pooling để tổng hợp embedding của các node thành embedding của đồ thị
    device : torch.device
        Thiết bị tính toán (CPU/GPU)
    method : str
        Phương pháp đánh giá ('knn')
    dataset : str
        Tên dataset ('wget', 'streamspot', etc.)
    n_dim : int, optional
        Số chiều của node features
    e_dim : int, optional
        Số chiều của edge features
        
    Returns:
    --------
    tuple
        (test_auc, test_std): AUC trung bình và độ lệch chuẩn
    """
    # Chuyển mô hình sang chế độ đánh giá (không cập nhật trọng số)
    model.eval()
    
    # Khởi tạo danh sách để lưu embedding và nhãn
    x_list = []
    y_list = []
    
    # Tải dữ liệu từ dataset
    data = load_batch_level_dataset(dataset)
    full = data['full_index']  # Lấy chỉ số của tất cả các đồ thị
    graphs = data['dataset']    # Lấy dữ liệu đồ thị
    
    # Tắt gradient để tối ưu bộ nhớ khi đánh giá
    with torch.no_grad():
        # Duyệt qua từng đồ thị
        for i in full:
            # Chuyển đổi đồ thị thành định dạng phù hợp và đưa lên device
            g = transform_graph(graphs[i][0], n_dim, e_dim).to(device)
            label = graphs[i][1]  # Lấy nhãn của đồ thị
            
            # Tạo embedding cho đồ thị
            out = model.embed(g)
            
            # Áp dụng pooling để tổng hợp embedding của các node
            if dataset != 'wget':
                out = pooler(g, out).cpu().numpy()
            else:
                # Xử lý đặc biệt cho dataset wget
                out = pooler(g, out, n_types=data['n_feat']).cpu().numpy()
            
            # Lưu embedding và nhãn
            y_list.append(label)
            x_list.append(out)
    
    # Ghép các embedding và nhãn thành ma trận
    x = np.concatenate(x_list, axis=0)
    y = np.array(y_list)
    
    # Chọn phương pháp đánh giá
    if 'knn' in method:
        test_auc, test_std = evaluate_batch_level_using_knn(100, dataset, x, y)
    else:
        raise NotImplementedError
    return test_auc, test_std


def evaluate_batch_level_using_knn(repeat, dataset, embeddings, labels):
    """
    Đánh giá mô hình sử dụng KNN ở mức batch
    
    Parameters:
    -----------
    repeat : int
        Số lần lặp lại đánh giá (-1 cho một lần duy nhất)
    dataset : str
        Tên dataset
    embeddings : np.ndarray
        Ma trận embedding của các đồ thị
    labels : np.ndarray
        Nhãn của các đồ thị (0: bình thường, 1: bất thường)
        
    Returns:
    --------
    tuple
        (auc_mean, auc_std): AUC trung bình và độ lệch chuẩn
    """
    x, y = embeddings, labels
    
    # Xác định số lượng mẫu train dựa trên dataset
    if dataset == 'streamspot':
        train_count = 400
    else:
        train_count = 100
    
    # Tính số lượng neighbors cho KNN (2% số lượng train hoặc tối đa 10)
    n_neighbors = min(int(train_count * 0.02), 10)
    
    # Tách chỉ số của các mẫu bình thường và bất thường
    benign_idx = np.where(y == 0)[0]  # Chỉ số của mẫu bình thường
    attack_idx = np.where(y == 1)[0]  # Chỉ số của mẫu bất thường
    
    if repeat != -1:  # Nếu cần lặp lại nhiều lần
        # Khởi tạo danh sách để lưu các metric
        prec_list = []
        rec_list = []
        f1_list = []
        tp_list = []
        fp_list = []
        tn_list = []
        fn_list = []
        auc_list = []
        
        # Lặp lại đánh giá nhiều lần
        for s in range(repeat):
            # Đặt seed ngẫu nhiên để đảm bảo tính tái lập
            set_random_seed(s)
            
            # Xáo trộn dữ liệu
            np.random.shuffle(benign_idx)
            np.random.shuffle(attack_idx)
            
            # Chia tập train và test
            x_train = x[benign_idx[:train_count]]  # Lấy train_count mẫu bình thường làm train
            x_test = np.concatenate([x[benign_idx[train_count:]], x[attack_idx]], axis=0)  # Phần còn lại làm test
            y_test = np.concatenate([y[benign_idx[train_count:]], y[attack_idx]], axis=0)
            
            # Chuẩn hóa dữ liệu
            x_train_mean = x_train.mean(axis=0)  # Tính giá trị trung bình
            x_train_std = x_train.std(axis=0)    # Tính độ lệch chuẩn
            x_train = (x_train - x_train_mean) / (x_train_std + 1e-6)  # Chuẩn hóa train
            x_test = (x_test - x_train_mean) / (x_train_std + 1e-6)    # Chuẩn hóa test

            # Huấn luyện KNN
            nbrs = NearestNeighbors(n_neighbors=n_neighbors)
            nbrs.fit(x_train)
            
            # Tính khoảng cách trung bình trên tập train
            distances, indexes = nbrs.kneighbors(x_train, n_neighbors=n_neighbors)
            mean_distance = distances.mean() * n_neighbors / (n_neighbors - 1)  # Điều chỉnh khoảng cách
            
            # Dự đoán trên tập test
            distances, indexes = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)
            score = distances.mean(axis=1) / mean_distance  # Tính điểm bất thường

            # Tính các metric
            auc = roc_auc_score(y_test, score)  # Tính AUC
            prec, rec, threshold = precision_recall_curve(y_test, score)  # Tính precision-recall
            f1 = 2 * prec * rec / (rec + prec + 1e-9)  # Tính F1-score
            
            # Tìm ngưỡng tốt nhất dựa trên F1-score
            max_f1_idx = np.argmax(f1)
            best_thres = threshold[max_f1_idx]
            
            # Lưu các metric
            prec_list.append(prec[max_f1_idx])
            rec_list.append(rec[max_f1_idx])
            f1_list.append(f1[max_f1_idx])

            # Tính confusion matrix
            tn = fn = tp = fp = 0
            for i in range(len(y_test)):
                if y_test[i] == 1.0 and score[i] >= best_thres:  # True Positive
                    tp += 1
                if y_test[i] == 1.0 and score[i] < best_thres:   # False Negative
                    fn += 1
                if y_test[i] == 0.0 and score[i] < best_thres:   # True Negative
                    tn += 1
                if y_test[i] == 0.0 and score[i] >= best_thres:  # False Positive
                    fp += 1
                    
            # Lưu confusion matrix
            tp_list.append(tp)
            fp_list.append(fp)
            fn_list.append(fn)
            tn_list.append(tn)
            auc_list.append(auc)

        # In kết quả trung bình và độ lệch chuẩn
        print('AUC: {}+{}'.format(np.mean(auc_list), np.std(auc_list)))
        print('F1: {}+{}'.format(np.mean(f1_list), np.std(f1_list)))
        print('PRECISION: {}+{}'.format(np.mean(prec_list), np.std(prec_list)))
        print('RECALL: {}+{}'.format(np.mean(rec_list), np.std(rec_list)))
        print('TN: {}+{}'.format(np.mean(tn_list), np.std(tn_list)))
        print('FN: {}+{}'.format(np.mean(fn_list), np.std(fn_list)))
        print('TP: {}+{}'.format(np.mean(tp_list), np.std(tp_list)))
        print('FP: {}+{}'.format(np.mean(fp_list), np.std(fp_list)))
        return np.mean(auc_list), np.std(auc_list)
    else:
        # Xử lý cho trường hợp chỉ chạy một lần
        set_random_seed(0)
        np.random.shuffle(benign_idx)
        np.random.shuffle(attack_idx)
        
        # Chia tập train và test
        x_train = x[benign_idx[:train_count]]  # Lấy train_count mẫu bình thường làm train
        x_test = np.concatenate([x[benign_idx[train_count:]], x[attack_idx]], axis=0)  # Phần còn lại làm test
        y_test = np.concatenate([y[benign_idx[train_count:]], y[attack_idx]], axis=0)
        
        # Chuẩn hóa dữ liệu
        x_train_mean = x_train.mean(axis=0)  # Tính giá trị trung bình
        x_train_std = x_train.std(axis=0)    # Tính độ lệch chuẩn
        x_train = (x_train - x_train_mean) / x_train_std  # Chuẩn hóa train
        x_test = (x_test - x_train_mean) / x_train_std    # Chuẩn hóa test

        # Huấn luyện KNN
        nbrs = NearestNeighbors(n_neighbors=n_neighbors)
        nbrs.fit(x_train)
        
        # Tính khoảng cách trung bình trên tập train
        distances, indexes = nbrs.kneighbors(x_train, n_neighbors=n_neighbors)
        mean_distance = distances.mean() * n_neighbors / (n_neighbors - 1)  # Điều chỉnh khoảng cách
        
        # Dự đoán trên tập test
        distances, indexes = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)
        score = distances.mean(axis=1) / mean_distance  # Tính điểm bất thường

        # Tính các metric
        auc = roc_auc_score(y_test, score)  # Tính AUC
        prec, rec, threshold = precision_recall_curve(y_test, score)  # Tính precision-recall
        f1 = 2 * prec * rec / (rec + prec + 1e-9)  # Tính F1-score
        
        # Tìm ngưỡng tốt nhất dựa trên F1-score
        best_idx = np.argmax(f1)
        best_thres = threshold[best_idx]

        # Tính confusion matrix
        tn = fn = tp = fp = 0
        for i in range(len(y_test)):
            if y_test[i] == 1.0 and score[i] >= best_thres:  # True Positive
                tp += 1
            if y_test[i] == 1.0 and score[i] < best_thres:   # False Negative
                fn += 1
            if y_test[i] == 0.0 and score[i] < best_thres:   # True Negative
                tn += 1
            if y_test[i] == 0.0 and score[i] >= best_thres:  # False Positive
                fp += 1
                
        # In kết quả
        print('AUC: {}'.format(auc))
        print('F1: {}'.format(f1[best_idx]))
        print('PRECISION: {}'.format(prec[best_idx]))
        print('RECALL: {}'.format(rec[best_idx]))
        print('TN: {}'.format(tn))
        print('FN: {}'.format(fn))
        print('TP: {}'.format(tp))
        print('FP: {}'.format(fp))
        return auc, 0.0

def evaluate_entity_level_using_knn(dataset, x_train, x_test, y_test):
    """
    Đánh giá mô hình sử dụng KNN ở mức entity (node)
    
    Parameters:
    -----------
    dataset : str
        Tên dataset ('cadets', 'trace', 'theia')
    x_train : np.ndarray
        Ma trận embedding của các node huấn luyện
    x_test : np.ndarray
        Ma trận embedding của các node kiểm thử
    y_test : np.ndarray
        Nhãn của các node kiểm thử (0: bình thường, 1: bất thường)
        
    Returns:
    --------
    tuple
        (auc, 0.0, None, None): AUC và các metric khác
    """
    # Chuẩn hóa dữ liệu
    x_train_mean = x_train.mean(axis=0)
    x_train_std = x_train.std(axis=0)
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    # Xác định số lượng neighbors dựa trên dataset
    if dataset == 'cadets':
        n_neighbors = 200  # Cadets cần nhiều neighbors hơn
    else:
        n_neighbors = 10

    # Khởi tạo và huấn luyện KNN
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)  # Sử dụng tất cả CPU cores
    nbrs.fit(x_train)

    # Đường dẫn lưu kết quả khoảng cách
    save_dict_path = './eval_result/distance_save_{}.pkl'.format(dataset)
    
    # Kiểm tra xem đã có kết quả lưu chưa
    if not os.path.exists(save_dict_path):
        # Tính toán và lưu khoảng cách nếu chưa có
        idx = list(range(x_train.shape[0]))
        random.shuffle(idx)
        # Tính khoảng cách cho một tập con của train (tối đa 50000 mẫu)
        distances, _ = nbrs.kneighbors(x_train[idx][:min(50000, x_train.shape[0])], n_neighbors=n_neighbors)
        del x_train  # Giải phóng bộ nhớ
        mean_distance = distances.mean()
        del distances
        # Tính khoảng cách cho tập test
        distances, _ = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)
        save_dict = [mean_distance, distances.mean(axis=1)]
        distances = distances.mean(axis=1)
        # Lưu kết quả
        with open(save_dict_path, 'wb') as f:
            pkl.dump(save_dict, f)
    else:
        # Đọc kết quả đã lưu
        with open(save_dict_path, 'rb') as f:
            mean_distance, distances = pkl.load(f)
            
    # Tính điểm bất thường
    score = distances / mean_distance
    del distances  # Giải phóng bộ nhớ
    
    # Tính các metric
    auc = roc_auc_score(y_test, score)
    prec, rec, threshold = precision_recall_curve(y_test, score)
    f1 = 2 * prec * rec / (rec + prec + 1e-9)
    
    # Tìm ngưỡng tốt nhất dựa trên recall cho từng dataset
    best_idx = -1
    for i in range(len(f1)):
        # Điều chỉnh ngưỡng recall cho từng dataset
        if dataset == 'trace' and rec[i] < 0.99979:
            best_idx = i - 1
            break
        if dataset == 'theia' and rec[i] < 0.99996:
            best_idx = i - 1
            break
        if dataset == 'cadets' and rec[i] < 0.9976:
            best_idx = i - 1
            break
    best_thres = threshold[best_idx]

    # Tính confusion matrix
    tn = fn = tp = fp = 0
    for i in range(len(y_test)):
        if y_test[i] == 1.0 and score[i] >= best_thres:  # True Positive
            tp += 1
        if y_test[i] == 1.0 and score[i] < best_thres:   # False Negative
            fn += 1
        if y_test[i] == 0.0 and score[i] < best_thres:   # True Negative
            tn += 1
        if y_test[i] == 0.0 and score[i] >= best_thres:  # False Positive
            fp += 1
            
    # In kết quả
    print('AUC: {}'.format(auc))
    print('F1: {}'.format(f1[best_idx]))
    print('PRECISION: {}'.format(prec[best_idx]))
    print('RECALL: {}'.format(rec[best_idx]))
    print('TN: {}'.format(tn))
    print('FN: {}'.format(fn))
    print('TP: {}'.format(tp))
    print('FP: {}'.format(fp))
    return auc, 0.0, None, None
