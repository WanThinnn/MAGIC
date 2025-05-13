import os
import random
import torch
import warnings
from tqdm import tqdm
from utils.loaddata import load_batch_level_dataset, load_entity_level_dataset, load_metadata
from model.autoencoder import build_model
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader
from model.train import batch_level_train
from utils.utils import set_random_seed, create_optimizer
from utils.config import build_args

# Tắt các cảnh báo không cần thiết để tránh rối khi in kết quả
warnings.filterwarnings('ignore')


def extract_dataloaders(entries, batch_size):
    '''
    Tạo DataLoader để huấn luyện từ danh sách chỉ số đồ thị (graph index).
    
    :param entries: danh sách các chỉ số đồ thị để huấn luyện
    :param batch_size: kích thước batch khi huấn luyện
    :return: GraphDataLoader với sampler ngẫu nhiên
    '''
    random.shuffle(entries)
    train_idx = torch.arange(len(entries))
    train_sampler = SubsetRandomSampler(train_idx)
    train_loader = GraphDataLoader(entries, batch_size=batch_size, sampler=train_sampler)
    return train_loader


def main(main_args):
    '''
    Hàm chính để huấn luyện mô hình AutoEncoder với dữ liệu đồ thị.

    Tùy thuộc vào bộ dữ liệu, chương trình sẽ xử lý dữ liệu ở cấp độ lô (batch-level) hoặc cấp độ thực thể (entity-level).
    Mô hình sau huấn luyện sẽ được lưu lại để sử dụng sau.

    :param main_args: tham số đầu vào chứa cấu hình mô hình và huấn luyện
    :return: None
    '''
    # Thiết lập thiết bị huấn luyện
    device = main_args.device if main_args.device >= 0 else "cpu"
    dataset_name = main_args.dataset

    # Cấu hình số lớp, số ẩn, và số epoch tùy theo tên bộ dữ liệu
    if dataset_name == 'streamspot':
        main_args.num_hidden = 256
        main_args.max_epoch = 5
        main_args.num_layers = 4
    elif dataset_name == 'wget':
        main_args.num_hidden = 256
        main_args.max_epoch = 2
        main_args.num_layers = 4
    else:
        main_args.num_hidden = 64
        main_args.max_epoch = 50
        main_args.num_layers = 3

    # Thiết lập hạt giống ngẫu nhiên
    set_random_seed(0)

    # --------------------------------------
    # Huấn luyện với dữ liệu cấp độ batch
    # --------------------------------------
    if dataset_name in ['streamspot', 'wget']:
        batch_size = 12 if dataset_name == 'streamspot' else 1
        dataset = load_batch_level_dataset(dataset_name)
        n_node_feat = dataset['n_feat']
        n_edge_feat = dataset['e_feat']
        graphs = dataset['dataset']
        train_index = dataset['train_index']

        # Cấu hình đầu vào cho mô hình
        main_args.n_dim = n_node_feat
        main_args.e_dim = n_edge_feat

        # Khởi tạo mô hình và tối ưu hóa
        model = build_model(main_args).to(device)
        optimizer = create_optimizer(main_args.optimizer, model, main_args.lr, main_args.weight_decay)

        # Huấn luyện mô hình
        model = batch_level_train(model, graphs, extract_dataloaders(train_index, batch_size),
                                  optimizer, main_args.max_epoch, device,
                                  main_args.n_dim, main_args.e_dim)

        # Lưu trạng thái mô hình đã huấn luyện
        torch.save(model.state_dict(), f"./checkpoints/checkpoint-{dataset_name}.pt")

    # --------------------------------------
    # Huấn luyện với dữ liệu cấp độ thực thể
    # --------------------------------------
    else:
        metadata = load_metadata(dataset_name)
        main_args.n_dim = metadata['node_feature_dim']
        main_args.e_dim = metadata['edge_feature_dim']
        model = build_model(main_args).to(device)
        model.train()
        optimizer = create_optimizer(main_args.optimizer, model, main_args.lr, main_args.weight_decay)

        n_train = metadata['n_train']
        epoch_iter = tqdm(range(main_args.max_epoch))  # hiển thị tiến trình huấn luyện

        # Huấn luyện qua nhiều epoch
        for epoch in epoch_iter:
            epoch_loss = 0.0
            for i in range(n_train):
                g = load_entity_level_dataset(dataset_name, 'train', i).to(device)
                model.train()
                loss = model(g)  # hàm forward trả về loss
                loss /= n_train  # chuẩn hóa loss
                optimizer.zero_grad()
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                del g
            epoch_iter.set_description(f"Epoch {epoch} | train_loss: {epoch_loss:.4f}")

        # Lưu trạng thái mô hình
        torch.save(model.state_dict(), f"./checkpoints/checkpoint-{dataset_name}.pt")

        # Xóa file lưu kết quả cũ (nếu có)
        save_dict_path = f'./eval_result/distance_save_{dataset_name}.pkl'
        if os.path.exists(save_dict_path):
            os.unlink(save_dict_path)

    return


if __name__ == '__main__':
    args = build_args()  # Đọc tham số cấu hình từ dòng lệnh hoặc file config
    main(args)
