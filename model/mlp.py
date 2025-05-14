
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


class MLP(nn.Module):
    '''
    Lớp MLP (Multi-Layer Perceptron) định nghĩa một mạng nơ-ron feed-forward đơn giản với hai tầng tuyến tính.
    Nó bao gồm một tầng đầu vào với kích thước `d_model`, một tầng ẩn với kích thước `d_ff`,
    và một tầng đầu ra với kích thước `d_model`.
    '''
    def __init__(self, d_model, d_ff, dropout=0.1):
        '''
        Hàm khởi tạo cho lớp MLP.

        :param d_model: Kích thước đầu vào và đầu ra của mạng (số lượng đặc trưng).
        :param d_ff: Kích thước của tầng ẩn (số lượng đặc trưng trong tầng ẩn).
        :param dropout: Tỷ lệ dropout để áp dụng giữa các tầng (mặc định là 0.1).

        :return: None.
        '''
        super(MLP, self).__init__()  # Khởi tạo lớp cha nn.Module
        self.w_1 = nn.Linear(d_model, d_ff)  # Tầng tuyến tính đầu tiên biến đổi từ d_model sang d_ff
        self.w_2 = nn.Linear(d_ff, d_model)  # Tầng tuyến tính thứ hai biến đổi từ d_ff về d_model
        self.dropout = nn.Dropout(dropout)  # Tầng dropout với tỷ lệ được chỉ định để chống quá khớp


    def forward(self, x):
        '''
        Hàm lan truyền xuôi cho lớp MLP.
        
        :param x: Tensor đầu vào với kích thước (batch_size, d_model).
        
        :return: Tensor đầu ra với kích thước (batch_size, d_model).'''
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
