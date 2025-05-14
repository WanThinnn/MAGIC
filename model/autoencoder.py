"""
Module này định nghĩa một mô hình Graph Masked Autoencoder (GMAE) để học biểu diễn đồ thị
thông qua việc tái tạo các thuộc tính nút và cấu trúc đồ thị.
"""

# Import các thư viện cần thiết
from .gat import GAT  # Graph Attention Network - mạng nơ-ron sử dụng cơ chế attention cho đồ thị
from utils.utils import create_norm  # Hàm tiện ích để tạo các lớp chuẩn hóa
from functools import partial  # Cho phép tạo các hàm một phần với một số tham số cố định
from itertools import chain  # Dùng để kết hợp nhiều iterator thành một
from .loss_func import sce_loss  # Hàm mất mát Scaled Cosine Error
import torch  # Thư viện deep learning chính
import torch.nn as nn  # Module neural network của PyTorch
import dgl  # Thư viện Deep Graph Library để xử lý đồ thị
import random  # Thư viện để tạo số ngẫu nhiên


def build_model(args):
    """
    Hàm xây dựng và trả về một instance của GMAEModel với các tham số từ args.
    
    :param args: Đối tượng chứa các tham số cấu hình cho mô hình
    :return: Instance của GMAEModel
    :rtype: GMAEModel
    :raises NotImplementedError: Nếu hàm mất mát không được hỗ trợ
    """
    # Lấy các tham số từ args
    num_hidden = args.num_hidden  # Kích thước của các lớp ẩn
    num_layers = args.num_layers  # Số lớp trong encoder/decoder
    negative_slope = args.negative_slope  # Hệ số độ dốc âm cho LeakyReLU
    mask_rate = args.mask_rate  # Tỷ lệ nút bị che trong quá trình huấn luyện
    alpha_l = args.alpha_l  # Hệ số alpha cho hàm mất mát SCE
    n_dim = args.n_dim  # Kích thước vector đặc trưng nút
    e_dim = args.e_dim  # Kích thước vector đặc trưng cạnh

    # Khởi tạo và trả về model GMAE với các tham số đã cấu hình
    model = GMAEModel(
        n_dim=n_dim,
        e_dim=e_dim,
        hidden_dim=num_hidden,
        n_layers=num_layers,
        n_heads=4,  # Số lượng attention heads cố định là 4
        activation="prelu",  # Sử dụng PReLU làm hàm kích hoạt
        feat_drop=0.1,  # Tỷ lệ dropout cho đặc trưng là 0.1
        negative_slope=negative_slope,
        residual=True,  # Bật kết nối tàn dư
        mask_rate=mask_rate,
        norm='BatchNorm',  # Sử dụng Batch Normalization
        loss_fn='sce',  # Sử dụng hàm mất mát SCE
        alpha_l=alpha_l
    )
    return model


class GMAEModel(nn.Module):
    """
    Mô hình Graph Masked Autoencoder (GMAE) kết hợp GAT để học biểu diễn đồ thị.
    
    Mô hình này thực hiện hai nhiệm vụ chính:
    1. Tái tạo thuộc tính nút bị che (masked node attributes)
    2. Tái tạo cấu trúc đồ thị (edge reconstruction)
    
    Các phương thức chính trong lớp này bao gồm:
    - __init__: Khởi tạo mô hình với các tham số cấu hình.
    - setup_loss_fn: Thiết lập hàm mất mát cho mô hình.
    - encoding_mask_noise: Tạo nhiễu bằng cách che một số nút trong đồ thị.
    - forward: Hàm chính để chạy mô hình và tính toán hàm mất mát.
    - compute_loss: Tính toán tổng hàm mất mát cho cả hai nhiệm vụ.
    - embed: Tạo biểu diễn nhúng cho các nút trong đồ thị.
    - enc_params: Trả về các tham số của encoder.
    - dec_params: Trả về các tham số của decoder và lớp chuyển đổi encoder-to-decoder.
    - output_hidden_dim: Kích thước đầu ra của lớp ẩn.
    
    """
    def __init__(self, n_dim, e_dim, hidden_dim, n_layers, n_heads, activation,
                 feat_drop, negative_slope, residual, norm, mask_rate=0.5, loss_fn="sce", alpha_l=2):
        '''
        Hàm khởi tạo mô hình GMAEModel.
        Các tham số đầu vào bao gồm kích thước đặc trưng nút, cạnh, kích thước ẩn,
        số lớp, số heads, hàm kích hoạt, tỷ lệ dropout, hệ số độ dốc âm,
        chế độ kết nối tàn dư, loại chuẩn hóa, tỷ lệ che nút, hàm mất mát và hệ số alpha.
        :param n_dim: Kích thước đặc trưng nút đầu vào
        :param e_dim: Kích thước đặc trưng cạnh đầu vào
        :param hidden_dim: Kích thước đầu ra của lớp ẩn
        :param n_layers: Số lớp trong encoder/decoder
        :param n_heads: Số lượng attention heads
        :param activation: Hàm kích hoạt sử dụng
        :param feat_drop: Tỷ lệ dropout cho đặc trưng
        :param negative_slope: Hệ số độ dốc âm cho LeakyReLU
        :param residual: Sử dụng kết nối tàn dư hay không
        :param norm: Loại chuẩn hóa sử dụng (BatchNorm, LayerNorm, GraphNorm)
        :param mask_rate: Tỷ lệ nút bị che trong quá trình huấn luyện
        :param loss_fn: Hàm mất mát sử dụng (sce)
        :param alpha_l: Hệ số alpha cho hàm mất mát SCE
        :return: None
        :rtype: None
        
        
        '''
        super(GMAEModel, self).__init__()
        # Lưu các tham số cấu hình
        self._mask_rate = mask_rate  # Tỷ lệ nút bị che
        self._output_hidden_size = hidden_dim  # Kích thước đầu ra của lớp ẩn
        # Khởi tạo hàm mất mát BCE cho việc tái tạo cạnh
        self.recon_loss = nn.BCELoss(reduction='mean')

        def init_weights(m):
            """Hàm khởi tạo trọng số cho các lớp Linear"""
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)  # Khởi tạo trọng số theo phân phối Xavier
                nn.init.constant_(m.bias, 0)  # Khởi tạo bias bằng 0

        # Xây dựng mạng MLP để tái tạo cạnh
        self.edge_recon_fc = nn.Sequential(
            nn.Linear(hidden_dim * n_layers * 2, hidden_dim),  # Lớp đầu vào: kết hợp đặc trưng của 2 nút
            nn.LeakyReLU(negative_slope),  # Hàm kích hoạt LeakyReLU
            nn.Linear(hidden_dim, 1),  # Lớp ẩn
            nn.Sigmoid()  # Hàm kích hoạt sigmoid để dự đoán xác suất tồn tại cạnh
        )
        self.edge_recon_fc.apply(init_weights)  # Áp dụng khởi tạo trọng số

        # Kiểm tra tính hợp lệ của kích thước ẩn và số lượng heads
        assert hidden_dim % n_heads == 0
        enc_num_hidden = hidden_dim // n_heads  # Kích thước ẩn cho mỗi head
        enc_nhead = n_heads  # Số lượng attention heads

        # Cấu hình cho decoder
        dec_in_dim = hidden_dim  # Kích thước đầu vào của decoder
        dec_num_hidden = hidden_dim  # Kích thước ẩn của decoder

        # Xây dựng encoder sử dụng GAT
        self.encoder = GAT(
            n_dim=n_dim,  # Kích thước đặc trưng nút đầu vào
            e_dim=e_dim,  # Kích thước đặc trưng cạnh
            hidden_dim=enc_num_hidden,  # Kích thước ẩn cho mỗi head
            out_dim=enc_num_hidden,  # Kích thước đầu ra cho mỗi head
            n_layers=n_layers,  # Số lớp GAT
            n_heads=enc_nhead,  # Số lượng attention heads
            n_heads_out=enc_nhead,  # Số lượng heads đầu ra
            concat_out=True,  # Kết hợp đầu ra của các heads
            activation=activation,  # Hàm kích hoạt
            feat_drop=feat_drop,  # Tỷ lệ dropout cho đặc trưng
            attn_drop=0.0,  # Tỷ lệ dropout cho attention
            negative_slope=negative_slope,  # Hệ số độ dốc âm cho LeakyReLU
            residual=residual,  # Sử dụng kết nối tàn dư
            norm=create_norm(norm),  # Lớp chuẩn hóa
            encoding=True,  # Chế độ encoding
        )

        # Xây dựng decoder sử dụng GAT
        self.decoder = GAT(
            n_dim=dec_in_dim,  # Kích thước đặc trưng nút đầu vào
            e_dim=e_dim,  # Kích thước đặc trưng cạnh
            hidden_dim=dec_num_hidden,  # Kích thước ẩn
            out_dim=n_dim,  # Kích thước đầu ra (bằng kích thước đặc trưng nút ban đầu)
            n_layers=1,  # Chỉ sử dụng 1 lớp GAT cho decoder
            n_heads=n_heads,  # Số lượng attention heads
            n_heads_out=1,  # Chỉ sử dụng 1 head đầu ra
            concat_out=True,  # Kết hợp đầu ra của các heads
            activation=activation,  # Hàm kích hoạt
            feat_drop=feat_drop,  # Tỷ lệ dropout cho đặc trưng
            attn_drop=0.0,  # Tỷ lệ dropout cho attention
            negative_slope=negative_slope,  # Hệ số độ dốc âm cho LeakyReLU
            residual=residual,  # Sử dụng kết nối tàn dư
            norm=create_norm(norm),  # Lớp chuẩn hóa
            encoding=False,  # Chế độ decoding
        )

        # Khởi tạo token mask cho encoder
        self.enc_mask_token = nn.Parameter(torch.zeros(1, n_dim))
        # Lớp chuyển đổi từ encoder sang decoder
        self.encoder_to_decoder = nn.Linear(dec_in_dim * n_layers, dec_in_dim, bias=False)

        # Thiết lập hàm mất mát
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def encoding_mask_noise(self, g, mask_rate=0.3):
        """
        Tạo nhiễu bằng cách che (mask) một số nút trong đồ thị.
        
        :param g: Đồ thị đầu vào
        :param mask_rate: Tỷ lệ nút bị che
        :return: Đồ thị mới với các nút bị che và chỉ số của các nút bị che
        :rtype: tuple (DGLGraph, tuple)
        
        """
        new_g = g.clone()  # Tạo bản sao của đồ thị để không ảnh hưởng đến đồ thị gốc
        num_nodes = g.num_nodes()  # Lấy số lượng nút
        # Tạo hoán vị ngẫu nhiên của các chỉ số nút
        perm = torch.randperm(num_nodes, device=g.device)

        # Tính số lượng nút sẽ bị che
        num_mask_nodes = int(mask_rate * num_nodes)
        # Chia các nút thành hai nhóm: bị che và không bị che
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        # Thay thế đặc trưng của các nút bị che bằng token mask
        new_g.ndata["attr"][mask_nodes] = self.enc_mask_token

        return new_g, (mask_nodes, keep_nodes)

    def forward(self, g):
        loss = self.compute_loss(g)
        return loss

    def compute_loss(self, g):
        """
        Tính toán tổng hàm mất mát cho cả hai nhiệm vụ: tái tạo thuộc tính và cấu trúc.
        
        :param g: Đồ thị đầu vào
        :return: Tổng hàm mất mát
        :rtype: torch.Tensor
     
        """
        # Tạo nhiễu bằng cách che một số nút
        pre_use_g, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, self._mask_rate)
        # Lấy đặc trưng của các nút
        pre_use_x = pre_use_g.ndata['attr'].to(pre_use_g.device)
        use_g = pre_use_g
        # Chạy encoder và lấy biểu diễn ẩn
        enc_rep, all_hidden = self.encoder(use_g, pre_use_x, return_hidden=True)
        # Kết hợp các biểu diễn ẩn từ tất cả các lớp
        enc_rep = torch.cat(all_hidden, dim=1)
        # Chuyển đổi biểu diễn từ encoder sang decoder
        rep = self.encoder_to_decoder(enc_rep)

        # Tái tạo đặc trưng nút
        recon = self.decoder(pre_use_g, rep)
        # Lấy đặc trưng gốc và đặc trưng tái tạo của các nút bị che
        x_init = g.ndata['attr'][mask_nodes]
        x_rec = recon[mask_nodes]
        # Tính hàm mất mát cho việc tái tạo đặc trưng
        loss = self.criterion(x_rec, x_init)

        # Tái tạo cấu trúc đồ thị
        threshold = min(10000, g.num_nodes())  # Giới hạn số lượng cạnh để xử lý

        # Lấy mẫu cạnh âm ngẫu nhiên
        negative_edge_pairs = dgl.sampling.global_uniform_negative_sampling(g, threshold)
        # Lấy mẫu cạnh dương ngẫu nhiên
        positive_edge_pairs = random.sample(range(g.number_of_edges()), threshold)
        positive_edge_pairs = (g.edges()[0][positive_edge_pairs], g.edges()[1][positive_edge_pairs])
        
        # Kết hợp đặc trưng của các cặp nút (cả cạnh dương và âm)
        sample_src = enc_rep[torch.cat([positive_edge_pairs[0], negative_edge_pairs[0]])].to(g.device)
        sample_dst = enc_rep[torch.cat([positive_edge_pairs[1], negative_edge_pairs[1]])].to(g.device)
        
        # Dự đoán xác suất tồn tại cạnh
        y_pred = self.edge_recon_fc(torch.cat([sample_src, sample_dst], dim=-1)).squeeze(-1)
        # Tạo nhãn: 1 cho cạnh dương, 0 cho cạnh âm
        y = torch.cat([torch.ones(len(positive_edge_pairs[0])), torch.zeros(len(negative_edge_pairs[0]))]).to(g.device)
        
        # Cộng thêm hàm mất mát tái tạo cạnh
        loss += self.recon_loss(y_pred, y)
        return loss

    def embed(self, g):
        """
        Tạo biểu diễn nhúng cho các nút trong đồ thị.
        
        :param g: Đồ thị đầu vào
        :return: Biểu diễn nhúng cho các nút
        :rtype: torch.Tensor
        
        """
        # Lấy đặc trưng nút và chuyển sang device phù hợp
        x = g.ndata['attr'].to(g.device)
        # Chạy encoder để lấy biểu diễn nhúng
        rep = self.encoder(g, x)
        return rep

    @property
    def enc_params(self):
        """
        Trả về các tham số của encoder.
        
        Returns:
        --------
        generator: Iterator chứa các tham số của encoder
        """
        return self.encoder.parameters()

    @property
    def dec_params(self):
        """
        Trả về các tham số của decoder và lớp chuyển đổi encoder-to-decoder.
        
        Returns:
        --------
        generator: Iterator chứa các tham số của decoder và encoder_to_decoder
        """
        # Kết hợp các tham số của encoder_to_decoder và decoder
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])