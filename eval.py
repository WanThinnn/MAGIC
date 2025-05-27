import torch
import warnings
from utils.loaddata import load_batch_level_dataset, load_entity_level_dataset, load_metadata
from model.autoencoder import build_model
from utils.poolers import Pooling
from utils.utils import set_random_seed
import numpy as np
from model.eval import batch_level_evaluation, evaluate_entity_level_using_knn
from utils.config import build_args

# Tắt cảnh báo để tránh in ra các cảnh báo không cần thiết
warnings.filterwarnings('ignore')


def main(main_args):
    '''
    Hàm chính để huấn luyện và đánh giá mô hình phát hiện bất thường.

    Tùy thuộc vào bộ dữ liệu, chương trình sẽ xử lý ở cấp độ lô (batch-level) hoặc thực thể (entity-level),
    tải mô hình đã lưu, trích xuất đặc trưng, và đánh giá bằng phương pháp KNN.

    :param main_args: đối tượng chứa các tham số cấu hình (dataset, thiết bị, số lớp, v.v.)
    :return: None
    '''
    # Thiết lập thiết bị (GPU hoặc CPU)
    device = main_args.device if main_args.device >= 0 else "cpu"
    device = torch.device(device)

    dataset_name = main_args.dataset

    # Cấu hình mô hình tùy thuộc vào tên bộ dữ liệu
    if dataset_name in ['streamspot', 'wget']:
        main_args.num_hidden = 256
        main_args.num_layers = 4
    else:
        main_args.num_hidden = 64
        main_args.num_layers = 3

    # Thiết lập hạt giống ngẫu nhiên để đảm bảo tái lập
    set_random_seed(0)

    if dataset_name in ['streamspot', 'wget']:
        # Dữ liệu cấp độ lô (batch-level)
        dataset = load_batch_level_dataset(dataset_name)
        n_node_feat = dataset['n_feat']  # số lượng đặc trưng của nút
        n_edge_feat = dataset['e_feat']  # số lượng đặc trưng của cạnh
        main_args.n_dim = n_node_feat
        main_args.e_dim = n_edge_feat

        # Khởi tạo và tải mô hình đã huấn luyện
        model = build_model(main_args)
        model.load_state_dict(torch.load(f"./checkpoints/checkpoint-{dataset_name}.pt", map_location=device))
        model = model.to(device)

        # Khởi tạo phương pháp pooling
        pooler = Pooling(main_args.pooling)

        # Đánh giá mô hình bằng phương pháp KNN
        test_auc, test_std = batch_level_evaluation(model, pooler, device, ['knn'], args.dataset,
                                                    main_args.n_dim, main_args.e_dim)
        
        
    else:
        # Dữ liệu cấp độ thực thể (entity-level)
        metadata = load_metadata(dataset_name)
        main_args.n_dim = metadata['node_feature_dim']
        main_args.e_dim = metadata['edge_feature_dim']

        # Khởi tạo và tải mô hình đã huấn luyện
        model = build_model(main_args)
        model.load_state_dict(torch.load(f"./checkpoints/checkpoint-{dataset_name}.pt", map_location=device))
        model = model.to(device)
        model.eval()  # chuyển sang chế độ đánh giá

        malicious, _ = metadata['malicious']
        n_train = metadata['n_train']
        n_test = metadata['n_test']

        # Trích xuất đặc trưng từ tập huấn luyện
        with torch.no_grad():
            x_train = []
            for i in range(n_train):
                g = load_entity_level_dataset(dataset_name, 'train', i).to(device)
                x_train.append(model.embed(g).cpu().numpy())  # trích xuất đặc trưng
                del g
            x_train = np.concatenate(x_train, axis=0)

            # Trích xuất đặc trưng từ tập kiểm thử
            skip_benign = 0
            x_test = []
            for i in range(n_test):
                g = load_entity_level_dataset(dataset_name, 'test', i).to(device)
                if i != n_test - 1:
                    skip_benign += g.number_of_nodes()  # loại bỏ phần dữ liệu trùng lặp với huấn luyện
                x_test.append(model.embed(g).cpu().numpy())
                del g
            x_test = np.concatenate(x_test, axis=0)

            # Gán nhãn dữ liệu kiểm thử
            n = x_test.shape[0]
            y_test = np.zeros(n)
            y_test[malicious] = 1.0

            malicious_dict = {m: i for i, m in enumerate(malicious)}

            # Loại bỏ các mẫu huấn luyện ra khỏi tập kiểm thử
            test_idx = []
            for i in range(x_test.shape[0]):
                if i >= skip_benign or y_test[i] == 1.0:
                    test_idx.append(i)

            result_x_test = x_test[test_idx]
            result_y_test = y_test[test_idx]
            del x_test, y_test

            # Đánh giá mô hình bằng KNN trên cấp độ thực thể
            test_auc, test_std, _, _ = evaluate_entity_level_using_knn(dataset_name, x_train,
                                                                       result_x_test, result_y_test)

    # In kết quả cuối cùng
    print(f"#Test_AUC: {test_auc:.4f}±{test_std:.4f}")
    return


if __name__ == '__main__':
    # Khởi tạo đối tượng chứa tham số cấu hình
    args = build_args()

    # Gọi hàm chính để chạy chương trình
    main(args)
