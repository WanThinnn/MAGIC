
# model/mlp.py

"""
TỔNG QUAN:
Mã nguồn này triển khai một lớp `MLP` (Multi-Layer Perceptron) sử dụng PyTorch, đại diện cho một mạng nơ-ron feed-forward đơn giản.
Lớp `MLP` bao gồm hai tầng tuyến tính (fully connected) với một hàm kích hoạt ReLU và dropout ở giữa, thường được sử dụng làm thành phần
trong các mô hình học sâu như Transformer hoặc các mạng học trên đồ thị để biến đổi đặc trưng. Mô hình nhận đầu vào có kích thước `d_model`, 
biến đổi qua tầng ẩn có kích thước `d_ff`, rồi trả về đầu ra có kích thước `d_model`. Dropout được áp dụng để chống quá khớp.
Mã này phù hợp cho các tác vụ học biểu diễn, phân loại, hoặc hồi quy trong các kiến trúc mạng phức tạp.
"""

import torch.nn as nn  # Nhập mô-đun mạng nơ-ron của PyTorch để định nghĩa các lớp tầng
import torch.nn.functional as F  # Nhập mô-đun chức năng của PyTorch để sử dụng các hàm như ReLU


# Lớp MLP: Triển khai một mạng nơ-ron feed-forward với hai tầng tuyến tính, hàm kích hoạt ReLU và dropout,
# dùng để biến đổi đặc trưng từ kích thước d_model qua d_ff rồi trở lại d_model.
class MLP(nn.Module):
    # Hàm __init__: Khởi tạo mô hình MLP với các tham số như kích thước đầu vào (d_model),
    # kích thước tầng ẩn (d_ff), và tỷ lệ dropout.
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(MLP, self).__init__()  # Khởi tạo lớp cha nn.Module
        self.w_1 = nn.Linear(d_model, d_ff)  # Tầng tuyến tính đầu tiên biến đổi từ d_model sang d_ff
        self.w_2 = nn.Linear(d_ff, d_model)  # Tầng tuyến tính thứ hai biến đổi từ d_ff về d_model
        self.dropout = nn.Dropout(dropout)  # Tầng dropout với tỷ lệ được chỉ định để chống quá khớp

    # Hàm forward: Thực hiện lan truyền xuôi của MLP, biến đổi đặc trưng đầu vào qua tầng tuyến tính,
    # hàm ReLU, dropout, rồi tầng tuyến tính cuối cùng.
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))  # Biến đổi x qua w_1, áp dụng ReLU, dropout, rồi w_2


##########################################################################################################################################
##########################################################################################################################################

######################
## MÃ NGUỒN BAN ĐẦU ##
######################

import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
