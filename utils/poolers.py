'''
Lớp Pooling được định nghĩa trong đoạn mã này là một lớp con của nn.Module trong PyTorch.
Nó được sử dụng để thực hiện các phép toán pooling trên các nút trong một đồ thị, với khả năng xử lý các loại nút khác nhau.


'''

import torch.nn as nn
import torch
import numpy as np


class Pooling(nn.Module):
    '''
    Lớp Pooling được định nghĩa trong đoạn mã này là một lớp con của nn.Module trong PyTorch.
    Nó được sử dụng để thực hiện các phép toán pooling trên các nút trong một đồ thị, với khả năng xử lý các loại nút khác nhau.
    Các phương thức chính trong lớp này bao gồm:
    - __init__: Khởi tạo lớp với một phương thức pooling cụ thể (mean, sum, max).
    - forward: Thực hiện phép toán pooling trên các nút trong đồ thị.

    '''
    def __init__(self, pooler):
        '''
        Hàm khởi tạo lớp Pooling.
        
        :param pooler: Phương thức pooling được sử dụng (mean, sum, max).
        :type pooler: str
        
        '''
        super(Pooling, self).__init__() # Khởi tạo lớp cha
        self.pooler = pooler # Phương thức pooling được sử dụng

    def forward(self, graph, feat, n_types=None):
        '''
        Hàm forward thực hiện phép toán pooling trên các nút trong đồ thị.
        
        :param graph: Đồ thị đầu vào.
        :type graph: dgl.DGLGraph
        :param feat: Tính năng đầu vào cho các nút trong đồ thị.
        :type feat: torch.Tensor
        :param n_types: Số lượng loại nút trong đồ thị.
        :type n_types: int, optional
        :return: Kết quả của phép toán pooling.
        :rtype: torch.Tensor
        
        '''
        # Implement node type-specific pooling
        with graph.local_scope(): # Đảm bảo rằng các thay đổi trên đồ thị chỉ ảnh hưởng đến phạm vi cục bộ
            if not n_types: # Nếu không có loại nút nào được chỉ định
                if self.pooler == 'mean': # Nếu phương thức pooling là mean
                    return feat.mean(0, keepdim=True) # Tính trung bình của các tính năng
                elif self.pooler == 'sum': # Nếu phương thức pooling là sum
                    return feat.sum(0, keepdim=True) # Tính tổng của các tính năng
                elif self.pooler == 'max': # Nếu phương thức pooling là max
                    return feat.max(0, keepdim=True) # Tính giá trị lớn nhất của các tính năng
                else: # Nếu phương thức pooling không được hỗ trợ
                    raise NotImplementedError
            else:
                result = [] # Danh sách để lưu trữ kết quả của các loại nút khác nhau
                for i in range(n_types): # Duyệt qua từng loại nút
                    mask = (graph.ndata['type'] == i) # Tạo mặt nạ cho loại nút hiện tại
                    if not mask.any(): # Nếu không có nút nào thuộc loại này
                        result.append(torch.zeros((1, feat.shape[-1]), device=feat.device)) # Thêm một tensor không có giá trị
                    elif self.pooler == 'mean': # Nếu phương thức pooling là mean
                        result.append(feat[mask].mean(0, keepdim=True)) # Tính trung bình của các tính năng
                    elif self.pooler == 'sum': # Nếu phương thức pooling là sum
                        result.append(feat[mask].sum(0, keepdim=True)) # Tính tổng của các tính năng
                    elif self.pooler == 'max': # Nếu phương thức pooling là max
                        result.append(feat[mask].max(0, keepdim=True)) # Tính giá trị lớn nhất của các tính năng
                    else: # Nếu phương thức pooling không được hỗ trợ
                        raise NotImplementedError
                result = torch.cat(result, dim=-1) # Kết hợp các kết quả của các loại nút khác nhau
                return result # Trả về kết quả cuối cùng
                
