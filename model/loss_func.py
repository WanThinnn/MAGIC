
# model/loss_func.py

"""
TỔNG QUAN:
Mã nguồn này triển khai một hàm mất mát tùy chỉnh có tên là `sce_loss` (Scaled Cosine Embedding Loss) sử dụng PyTorch. 
Hàm tính toán mất mát dựa trên độ tương đồng cosine giữa hai tập hợp vector (thường là đặc trưng dự đoán và đặc trưng mục tiêu), 
sau đó nâng lũy thừa để khuếch đại sự khác biệt và lấy trung bình để trả về giá trị mất mát cuối cùng. 
Mất mát này thường được sử dụng trong các tác vụ học biểu diễn, đối sánh đặc trưng, hoặc học độ tương đồng, 
như trong các mô hình học sâu liên quan đến đồ thị hoặc nhúng vector.
"""

import torch.nn.functional as F  # Nhập mô-đun chức năng của PyTorch để sử dụng các hàm như chuẩn hóa và phép toán tensor


def sce_loss(x, y, alpha=3):
    '''
    Tính toán hàm mất mát SCE (Scaled Cosine Embedding Loss) giữa hai tensor x và y.
    Hàm này sử dụng độ tương đồng cosine giữa hai tensor để tính toán mất mát,
    sau đó nâng lũy thừa để khuếch đại sự khác biệt giữa chúng.
    
    :param x: Tensor đầu vào đầu tiên (thường là đặc trưng dự đoán).
    :param y: Tensor đầu vào thứ hai (thường là đặc trưng mục tiêu).
    :param alpha: Hệ số khuếch đại (mặc định là 3).
    
    :return: Giá trị mất mát SCE.
    '''
    x = F.normalize(x, p=2, dim=-1)  # Chuẩn hóa L2 vector x theo chiều cuối cùng (dim=-1) để có độ dài đơn vị
    y = F.normalize(y, p=2, dim=-1)  # Chuẩn hóa L2 vector y theo chiều cuối cùng (dim=-1) để có độ dài đơn vị
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)  # Tính độ tương đồng cosine, lấy 1 trừ đi, rồi nâng lũy thừa alpha
    loss = loss.mean()  # Tính giá trị trung bình của mất mát trên tất cả mẫu
    return loss  # Trả về giá trị mất mát cuối cùng


##########################################################################################################################################
##########################################################################################################################################

######################
## MÃ NGUỒN BAN ĐẦU ##
######################

# import torch.nn.functional as F


# def sce_loss(x, y, alpha=3):
#     x = F.normalize(x, p=2, dim=-1)
#     y = F.normalize(y, p=2, dim=-1)
#     loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
#     loss = loss.mean()
#     return loss
