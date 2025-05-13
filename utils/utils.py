import torch
import torch.nn as nn
from functools import partial
import numpy as np
import random
import torch.optim as optim


def create_optimizer(opt, model, lr, weight_decay):
    """
    Tạo optimizer cho model dựa trên các tham số được cung cấp.
    
    Args:
        opt (str): Tên của optimizer (adam, adamw, adadelta, radam, sgd)
        model (nn.Module): Model PyTorch cần được tối ưu hóa
        lr (float): Learning rate cho optimizer
        weight_decay (float): Hệ số regularization để tránh overfitting
        
    Returns:
        torch.optim.Optimizer: Optimizer được khởi tạo với các tham số phù hợp
    """
    opt_lower = opt.lower()  # Chuyển tên optimizer về chữ thường
    parameters = model.parameters()  # Lấy tất cả các tham số của model
    opt_args = dict(lr=lr, weight_decay=weight_decay)  # Tạo dict chứa các tham số cơ bản
    optimizer = None
    opt_split = opt_lower.split("_")  # Tách tên optimizer nếu có prefix
    opt_lower = opt_split[-1]  # Lấy phần cuối cùng của tên optimizer
    
    # Khởi tạo optimizer tương ứng với tên được chọn
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9  # Thêm momentum cho SGD
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
    return optimizer


def random_shuffle(x, y):
    """
    Trộn ngẫu nhiên dữ liệu x và nhãn y tương ứng.
    
    Args:
        x (numpy.ndarray): Dữ liệu đầu vào
        y (numpy.ndarray): Nhãn tương ứng
        
    Returns:
        tuple: (x đã trộn, y đã trộn)
    """
    idx = list(range(len(x)))  # Tạo danh sách chỉ số
    random.shuffle(idx)  # Trộn ngẫu nhiên các chỉ số
    return x[idx], y[idx]  # Trả về dữ liệu và nhãn đã được trộn


def set_random_seed(seed):
    """
    Thiết lập seed cho tất cả các thư viện ngẫu nhiên để đảm bảo tính tái lập.
    
    Args:
        seed (int): Giá trị seed để khởi tạo
    """
    random.seed(seed)  # Seed cho random module
    np.random.seed(seed)  # Seed cho numpy
    torch.manual_seed(seed)  # Seed cho PyTorch CPU
    torch.cuda.manual_seed(seed)  # Seed cho PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # Seed cho tất cả các GPU
    torch.backends.cudnn.determinstic = True  # Đảm bảo tính tái lập cho cuDNN


def create_activation(name):
    """
    Tạo hàm kích hoạt (activation function) dựa trên tên.
    
    Args:
        name (str): Tên của hàm kích hoạt ('relu', 'gelu', 'prelu', 'elu' hoặc None)
        
    Returns:
        nn.Module: Module kích hoạt tương ứng
        
    Raises:
        NotImplementedError: Nếu tên hàm kích hoạt không được hỗ trợ
    """
    if name == "relu":
        return nn.ReLU()  # Rectified Linear Unit
    elif name == "gelu":
        return nn.GELU()  # Gaussian Error Linear Unit
    elif name == "prelu":
        return nn.PReLU()  # Parametric ReLU
    elif name is None:
        return nn.Identity()  # Hàm đồng nhất
    elif name == "elu":
        return nn.ELU()  # Exponential Linear Unit
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name):
    """
    Tạo lớp chuẩn hóa (normalization layer) dựa trên tên.
    
    Args:
        name (str): Tên của lớp chuẩn hóa ('layernorm', 'batchnorm', 'graphnorm')
        
    Returns:
        nn.Module or None: Lớp chuẩn hóa tương ứng hoặc None nếu không hỗ trợ
    """
    if name == "layernorm":
        return nn.LayerNorm  # Layer Normalization
    elif name == "batchnorm":
        return nn.BatchNorm1d  # Batch Normalization cho dữ liệu 1D
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")  # Graph Normalization
    else:
        return None


class NormLayer(nn.Module):
    """
    Lớp chuẩn hóa tùy chỉnh cho đồ thị (graph).
    
    Attributes:
        norm (nn.Module): Module chuẩn hóa được sử dụng
        weight (nn.Parameter): Tham số trọng số học được
        bias (nn.Parameter): Tham số bias học được
        mean_scale (nn.Parameter): Tham số tỷ lệ trung bình học được
    """
    def __init__(self, hidden_dim, norm_type):
        """
        Args:
            hidden_dim (int): Kích thước của vector ẩn
            norm_type (str): Loại chuẩn hóa ('batchnorm', 'layernorm', 'graphnorm')
        """
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            # Khởi tạo các tham số học được
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))
            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError

    def forward(self, graph, x):
        """
        Thực hiện chuẩn hóa cho dữ liệu đồ thị.
        
        Args:
            graph: Đối tượng đồ thị chứa thông tin về batch
            x (torch.Tensor): Tensor đầu vào cần chuẩn hóa
            
        Returns:
            torch.Tensor: Tensor đã được chuẩn hóa
        """
        tensor = x
        # Xử lý các trường hợp chuẩn hóa thông thường
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        # Xử lý chuẩn hóa đồ thị
        batch_list = graph.batch_num_nodes  # Số node trong mỗi batch
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        # Tạo chỉ số batch cho mỗi node
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        
        # Tính toán giá trị trung bình
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        # Chuẩn hóa bằng cách trừ đi giá trị trung bình
        sub = tensor - mean * self.mean_scale

        # Tính toán độ lệch chuẩn
        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        
        # Áp dụng chuẩn hóa cuối cùng
        return self.weight * sub / std + self.bias
