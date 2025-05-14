
# model/train.py

"""
TỔNG QUAN:
Mã nguồn này triển khai một hàm huấn luyện cấp lô (`batch_level_train`) để huấn luyện một mô hình học sâu trên dữ liệu đồ thị sử dụng PyTorch và DGL (Deep Graph Library). 
Hàm nhận một mô hình, tập hợp đồ thị, bộ tải dữ liệu huấn luyện, bộ tối ưu hóa, số epoch tối đa, thiết bị tính toán (CPU/GPU), và các kích thước đặc trưng của nút/cạnh. 
Nó thực hiện huấn luyện theo từng epoch, xử lý dữ liệu theo lô, tính toán mất mát, cập nhật trọng số mô hình, và theo dõi tiến trình bằng thanh tiến độ. 
Hàm này phù hợp cho các tác vụ học trên đồ thị như phân loại nút, dự đoán liên kết, hoặc học biểu diễn đồ thị.
"""

import dgl  # Nhập thư viện DGL để xử lý dữ liệu đồ thị và thực hiện các thao tác trên đồ thị
import numpy as np  # Nhập NumPy để thực hiện các phép toán số học, như tính trung bình mất mát
from tqdm import tqdm  # Nhập tqdm để tạo thanh tiến độ theo dõi quá trình huấn luyện
from utils.loaddata import transform_graph  # Nhập hàm transform_graph từ utils để biến đổi đồ thị trước khi huấn luyện

# Hàm batch_level_train: Huấn luyện mô hình trên dữ liệu đồ thị theo lô qua nhiều epoch, sử dụng bộ tối ưu hóa
# để cập nhật trọng số, trả về mô hình đã được huấn luyện.
def batch_level_train(model, graphs, train_loader, optimizer, max_epoch, device, n_dim=0, e_dim=0):
    epoch_iter = tqdm(range(max_epoch))  # Tạo thanh tiến độ cho số epoch tối đa
    for epoch in epoch_iter:  # Lặp qua từng epoch
        model.train()  # Đặt mô hình ở chế độ huấn luyện
        loss_list = []  # Danh sách để lưu giá trị mất mát của từng lô
        for _, batch in enumerate(train_loader):  # Lặp qua từng lô trong bộ tải dữ liệu
            batch_g = [transform_graph(graphs[idx][0], n_dim, e_dim).to(device) for idx in batch]  # Biến đổi các đồ thị trong lô và chuyển sang thiết bị (CPU/GPU)
            batch_g = dgl.batch(batch_g)  # Gộp các đồ thị trong lô thành một đồ thị lớn
            model.train()  # Đảm bảo mô hình ở chế độ huấn luyện (lặp lại để chắc chắn)
            loss = model(batch_g)  # Tính mất mát bằng cách truyền đồ thị lô qua mô hình
            optimizer.zero_grad()  # Xóa gradient của các tham số trước khi lan truyền ngược
            loss.backward()  # Lan truyền ngược để tính gradient
            optimizer.step()  # Cập nhật trọng số mô hình dựa trên gradient
            loss_list.append(loss.item())  # Lưu giá trị mất mát của lô vào danh sách
            del batch_g  # Xóa đồ thị lô để giải phóng bộ nhớ
        epoch_iter.set_description(f"Epoch {epoch} | train_loss: {np.mean(loss_list):.4f}")  # Cập nhật thanh tiến độ với epoch và mất mát trung bình
    return model  # Trả về mô hình đã được huấn luyện


##########################################################################################################################################
##########################################################################################################################################

######################
## MÃ NGUỒN BAN ĐẦU ##
######################

import dgl
import numpy as np
from tqdm import tqdm
from utils.loaddata import transform_graph


def batch_level_train(model, graphs, train_loader, optimizer, max_epoch, device, n_dim=0, e_dim=0):
    epoch_iter = tqdm(range(max_epoch))
    for epoch in epoch_iter:
        model.train()
        loss_list = []
        for _, batch in enumerate(train_loader):
            batch_g = [transform_graph(graphs[idx][0], n_dim, e_dim).to(device) for idx in batch]
            batch_g = dgl.batch(batch_g)
            model.train()
            loss = model(batch_g)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            del batch_g
        epoch_iter.set_description(f"Epoch {epoch} | train_loss: {np.mean(loss_list):.4f}")
    return model
