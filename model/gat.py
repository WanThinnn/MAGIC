
# model/gat.py

"""
TỔNG QUAN:
Mã nguồn này triển khai một Mạng Chú ý Đồ thị (Graph Attention Network - GAT) sử dụng PyTorch và thư viện DGL (Deep Graph Library). 
Mô hình GAT xử lý dữ liệu đồ thị bằng cách áp dụng cơ chế chú ý để học cách gán trọng số cho các nút láng giềng, từ đó tổng hợp thông tin một cách hiệu quả. 
Mã bao gồm hai lớp chính:
1. Lớp `GAT`: Xây dựng một mạng GAT đa tầng, xếp chồng nhiều tầng `GATConv` để biến đổi đặc trưng nút qua các tầng, hỗ trợ chú ý đa đầu, kết nối dư, chuẩn hóa và phân loại.
2. Lớp `GATConv`: Triển khai một tầng chú ý đồ thị, tính điểm chú ý dựa trên đặc trưng nút nguồn, đích và cạnh, thực hiện truyền tin để cập nhật đặc trưng nút.
Mô hình hỗ trợ các cấu hình linh hoạt như chú ý đa đầu, dropout, kết nối dư, chuẩn hóa, và có thể xử lý đồ thị lưỡng phân hoặc đặc trưng riêng cho nút nguồn/đích. 
Mã được thiết kế để sử dụng trong các tác vụ học trên đồ thị như phân loại nút, dự đoán liên kết hoặc mã hóa đồ thị.
"""

import torch  # Nhập thư viện PyTorch để thực hiện các phép toán tensor và mô-đun mạng nơ-ron
import torch.nn as nn  # Nhập mô-đun mạng nơ-ron của PyTorch để định nghĩa các lớp tầng
from dgl.ops import edge_softmax  # Nhập hàm edge_softmax của DGL để chuẩn hóa điểm chú ý
import dgl.function as fn  # Nhập các hàm của DGL để xử lý truyền tin và giảm trừ
from dgl.utils import expand_as_pair  # Nhập hàm tiện ích để xử lý kích thước đầu vào dưới dạng cặp
from utils.utils import create_activation  # Nhập hàm tiện ích tùy chỉnh để tạo hàm kích hoạt


class GAT(nn.Module):
    '''
    Mạng Chú ý Đồ thị (Graph Attention Network - GAT) với nhiều tầng, hỗ trợ chú ý đa đầu, kết nối dư và chuẩn hóa.
    Mô hình này có thể được sử dụng cho các tác vụ như phân loại nút, dự đoán liên kết hoặc mã hóa đồ thị.
    '''
    def __init__(self,
                 n_dim,  # Kích thước đặc trưng đầu vào của nút
                 e_dim,  # Kích thước đặc trưng đầu vào của cạnh
                 hidden_dim,  # Kích thước tầng ẩn
                 out_dim,  # Kích thước đầu ra
                 n_layers,  # Số lượng tầng GAT
                 n_heads,  # Số lượng đầu chú ý trong các tầng trung gian
                 n_heads_out,  # Số lượng đầu chú ý trong tầng đầu ra
                 activation,  # Tên hàm kích hoạt (ví dụ: 'relu')
                 feat_drop,  # Tỷ lệ dropout cho đặc trưng nút
                 attn_drop,  # Tỷ lệ dropout cho trọng số chú ý
                 negative_slope,  # Độ dốc âm cho LeakyReLU
                 residual,  # Có sử dụng kết nối dư (residual) hay không
                 norm,  # Lớp chuẩn hóa (ví dụ: LayerNorm)
                 concat_out=False,  # Có nối các đầu ra đa đầu (True) hay lấy trung bình (False)
                 encoding=False  # Mô hình có được sử dụng để mã hóa hay không (ảnh hưởng đến kích hoạt/dư cuối)
                 ):
        '''
        Khởi tạo mô hình GAT với các tham số cấu hình như số tầng, số đầu chú ý,
        kích thước đặc trưng, dropout, kết nối dư, chuẩn hóa và chế độ mã hóa.

        :param n_dim: Kích thước đặc trưng đầu vào của nút
        :param e_dim: Kích thước đặc trưng đầu vào của cạnh
        :param hidden_dim: Kích thước tầng ẩn
        :param out_dim: Kích thước đầu ra
        :param n_layers: Số lượng tầng GAT
        :param n_heads: Số lượng đầu chú ý trong các tầng trung gian
        :param n_heads_out: Số lượng đầu chú ý trong tầng đầu ra
        :param activation: Tên hàm kích hoạt (ví dụ: 'relu')
        :param feat_drop: Tỷ lệ dropout cho đặc trưng nút
        :param attn_drop: Tỷ lệ dropout cho trọng số chú ý
        :param negative_slope: Độ dốc âm cho LeakyReLU
        :param residual: Có sử dụng kết nối dư (residual) hay không
        :param norm: Lớp chuẩn hóa (ví dụ: LayerNorm)
        :param concat_out: Có nối các đầu ra đa đầu (True) hay lấy trung bình (False)
        :param encoding: Mô hình có được sử dụng để mã hóa hay không (ảnh hưởng đến kích hoạt/dư cuối)

        :return: None
        '''

        super(GAT, self).__init__()  # Khởi tạo lớp cha nn.Module
        self.out_dim = out_dim  # Lưu kích thước đầu ra
        self.n_heads = n_heads  # Lưu số lượng đầu chú ý cho các tầng trung gian
        self.n_layers = n_layers  # Lưu số lượng tầng GAT
        self.gats = nn.ModuleList()  # Khởi tạo danh sách để chứa các tầng GATConv
        self.concat_out = concat_out  # Lưu trạng thái có nối các đầu ra đa đầu hay không

        last_activation = create_activation(activation) if encoding else None  # Sử dụng kích hoạt nếu đang mã hóa, ngược lại là None
        last_residual = (encoding and residual)  # Sử dụng kết nối dư ở tầng cuối chỉ khi đang mã hóa và residual là True
        last_norm = norm if encoding else None  # Sử dụng chuẩn hóa ở tầng cuối chỉ khi đang mã hóa

        if self.n_layers == 1: # Nếu chỉ có một tầng, thêm một tầng GATConv duy nhất với kích thước đầu ra và số đầu được chỉ định
            self.gats.append(GATConv(
                n_dim, e_dim, out_dim, n_heads_out, feat_drop, attn_drop, negative_slope,
                last_residual, norm=last_norm, concat_out=self.concat_out
            ))  # Thêm một tầng GATConv duy nhất với kích thước đầu ra và số đầu được chỉ định
        else: # Nếu có nhiều tầng, thêm tầng đầu vào với kích thước đầu vào và số đầu chú ý
            self.gats.append(GATConv(
                n_dim, e_dim, hidden_dim, n_heads, feat_drop, attn_drop, negative_slope,
                residual, create_activation(activation),
                norm=norm, concat_out=self.concat_out
            ))  # Thêm tầng GATConv đầu tiên biến đổi từ đầu vào sang kích thước ẩn
            for _ in range(1, self.n_layers - 1):
                self.gats.append(GATConv(
                    hidden_dim * self.n_heads, e_dim, hidden_dim, n_heads,
                    feat_drop, attn_drop, negative_slope,
                    residual, create_activation(activation),
                    norm=norm, concat_out=self.concat_out
                ))  # Thêm các tầng GATConv trung gian với đầu vào nối từ các đầu trước
            self.gats.append(GATConv(
                hidden_dim * self.n_heads, e_dim, out_dim, n_heads_out,
                feat_drop, attn_drop, negative_slope,
                last_residual, last_activation, norm=last_norm, concat_out=self.concat_out
            ))  # Thêm tầng GATConv cuối biến đổi sang kích thước đầu ra
        self.head = nn.Identity()  # Khởi tạo đầu ra là hàm đồng nhất (không biến đổi gì)


    def forward(self, g, input_feature, return_hidden=False):
        '''
        Thực hiện lan truyền xuôi của mô hình GAT, truyền đặc trưng nút qua các tầng GATConv
        và trả về đầu ra cuối cùng hoặc cả trạng thái ẩn nếu yêu cầu.

        :param g: Đồ thị đầu vào (DGLGraph)
        :param input_feature: Đặc trưng đầu vào của nút (tensor)
        :param return_hidden: Có trả về trạng thái ẩn của các tầng hay không (bool)

        :return: Đầu ra cuối cùng hoặc cả đầu ra và trạng thái ẩn (tuple)
        '''
        h = input_feature  # Khởi tạo đặc trưng hiện tại là đặc trưng đầu vào
        hidden_list = []  # Danh sách để lưu trạng thái ẩn của mỗi tầng
        for layer in range(self.n_layers):
            h = self.gats[layer](g, h)  # Truyền đặc trưng qua tầng GATConv hiện tại
            hidden_list.append(h)  # Lưu đặc trưng đầu ra
        if return_hidden:
            return self.head(h), hidden_list  # Trả về đầu ra cuối và danh sách trạng thái ẩn
        else:
            return self.head(h)  # Chỉ trả về đầu ra cuối


    def reset_classifier(self, num_classes):
        '''
        Thay đổi tầng đầu ra của mô hình để phù hợp với số lớp mới cho tác vụ phân loại.
        :param num_classes: Số lớp đầu ra mới (int)
        :return: None
        '''
        self.head = nn.Linear(self.num_heads * self.out_dim, num_classes)  # Thay đầu ra bằng tầng tuyến tính cho phân loại


class GATConv(nn.Module):
    '''
    Tầng Chú ý Đồ thị (Graph Attention Layer) với khả năng xử lý các đặc trưng nút và cạnh,
    tính toán điểm chú ý và thực hiện truyền tin giữa các nút trong đồ thị.
    '''
    def __init__(self,
                 in_dim,  # Kích thước đặc trưng đầu vào của nút (có thể là tuple cho src/dst)
                 e_dim,  # Kích thước đặc trưng đầu vào của cạnh
                 out_dim,  # Kích thước đầu ra cho mỗi đầu
                 n_heads,  # Số lượng đầu chú ý
                 feat_drop=0.0,  # Tỷ lệ dropout cho đặc trưng nút
                 attn_drop=0.0,  # Tỷ lệ dropout cho trọng số chú ý
                 negative_slope=0.2,  # Độ dốc âm cho LeakyReLU
                 residual=False,  # Có sử dụng kết nối dư hay không
                 activation=None,  # Hàm kích hoạt (ví dụ: ReLU)
                 allow_zero_in_degree=False,  # Cho phép nút có bậc vào bằng 0
                 bias=True,  # Có bao gồm bias trong đầu ra hay không
                 norm=None,  # Lớp chuẩn hóa (ví dụ: LayerNorm)
                 concat_out=True):  # Có nối các đầu ra đa đầu hay không
        '''
        Khởi tạo tầng GATConv với các tham số cấu hình như kích thước đầu vào, đầu ra,
        số đầu chú ý, dropout, kết nối dư, chuẩn hóa và chế độ cho phép bậc vào bằng 0.

        :param in_dim: Kích thước đặc trưng đầu vào của nút (có thể là tuple cho src/dst)
        :param e_dim: Kích thước đặc trưng đầu vào của cạnh
        :param out_dim: Kích thước đầu ra cho mỗi đầu
        :param n_heads: Số lượng đầu chú ý
        :param feat_drop: Tỷ lệ dropout cho đặc trưng nút
        :param attn_drop: Tỷ lệ dropout cho trọng số chú ý
        :param negative_slope: Độ dốc âm cho LeakyReLU
        :param residual: Có sử dụng kết nối dư hay không
        :param activation: Hàm kích hoạt (ví dụ: ReLU)
        :param allow_zero_in_degree: Cho phép nút có bậc vào bằng 0
        :param bias: Có bao gồm bias trong đầu ra hay không
        :param norm: Lớp chuẩn hóa (ví dụ: LayerNorm)
        :param concat_out: Có nối các đầu ra đa đầu hay không

        :return: None
        '''
        super(GATConv, self).__init__()  # Khởi tạo lớp cha nn.Module
        self.n_heads = n_heads  # Lưu số lượng đầu chú ý
        self.src_feat, self.dst_feat = expand_as_pair(in_dim)  # Tách kích thước đầu vào thành kích thước nguồn và đích
        self.edge_feat = e_dim  # Lưu kích thước đặc trưng cạnh
        self.out_feat = out_dim  # Lưu kích thước đầu ra cho mỗi đầu
        self.allow_zero_in_degree = allow_zero_in_degree  # Lưu trạng thái cho phép nút có bậc vào bằng 0
        self.concat_out = concat_out  # Lưu trạng thái có nối các đầu ra đa đầu hay không

        if isinstance(in_dim, tuple): # Nếu đầu vào là tuple, tức là có đặc trưng riêng cho nút nguồn và đích
            self.fc_node_embedding = nn.Linear(
                self.src_feat, self.out_feat * self.n_heads, bias=False)  # Tầng tuyến tính cho nhúng nút
            self.fc_src = nn.Linear(self.src_feat, self.out_feat * self.n_heads, bias=False)  # Tầng tuyến tính cho nút nguồn
            self.fc_dst = nn.Linear(self.dst_feat, self.out_feat * self.n_heads, bias=False)  # Tầng tuyến tính cho nút đích
        else: # Nếu đầu vào không phải là tuple, tức là sử dụng cùng một đặc trưng cho cả nút nguồn và đích
            self.fc_node_embedding = nn.Linear(
                self.src_feat, self.out_feat * self.n_heads, bias=False)  # Tầng tuyến tính cho nhúng nút (không sử dụng)
            self.fc = nn.Linear(self.src_feat, self.out_feat * self.n_heads, bias=False)  # Tầng tuyến tính cho tất cả nút
        self.edge_fc = nn.Linear(self.edge_feat, self.out_feat * self.n_heads, bias=False)  # Tầng tuyến tính cho đặc trưng cạnh
        self.attn_h = nn.Parameter(torch.FloatTensor(size=(1, self.n_heads, self.out_feat)))  # Tham số chú ý cho nút nguồn
        self.attn_e = nn.Parameter(torch.FloatTensor(size=(1, self.n_heads, self.out_feat)))  # Tham số chú ý cho cạnh
        self.attn_t = nn.Parameter(torch.FloatTensor(size=(1, self.n_heads, self.out_feat)))  # Tham số chú ý cho nút đích
        self.feat_drop = nn.Dropout(feat_drop)  # Tầng dropout cho đặc trưng nút
        self.attn_drop = nn.Dropout(attn_drop)  # Tầng dropout cho trọng số chú ý
        self.leaky_relu = nn.LeakyReLU(negative_slope)  # Hàm kích hoạt LeakyReLU cho điểm chú ý
        if bias: # Nếu sử dụng bias, khởi tạo tham số bias cho đầu ra
            self.bias = nn.Parameter(torch.FloatTensor(size=(1, self.n_heads, self.out_feat)))  # Tham số bias cho đầu ra
        else: # Nếu không sử dụng bias, đăng ký None làm bias
            self.register_buffer('bias', None)  # Đăng ký None làm bias nếu không sử dụng
        if residual: # Nếu sử dụng kết nối dư, khởi tạo tầng tuyến tính cho kết nối dư
            if self.dst_feat != self.n_heads * self.out_feat: # Nếu kích thước đầu vào và đầu ra không khớp, cần biến đổi
                self.res_fc = nn.Linear(
                    self.dst_feat, self.n_heads * self.out_feat, bias=False)  # Tầng tuyến tính cho kết nối dư
            else: # Nếu kích thước đầu vào và đầu ra khớp, không cần biến đổi
                self.res_fc = nn.Identity()  # Hàm đồng nhất nếu kích thước đầu vào và đầu ra khớp
        else: # Nếu không sử dụng kết nối dư, đăng ký None làm kết nối dư
            self.register_buffer('res_fc', None)  # Đăng ký None nếu không có kết nối dư
        self.reset_parameters()  # Khởi tạo các tham số
        self.activation = activation  # Lưu hàm kích hoạt
        self.norm = norm  # Lưu lớp chuẩn hóa
        if norm is not None: # Nếu có chuẩn hóa, khởi tạo lớp chuẩn hóa với kích thước đầu ra
            self.norm = norm(self.n_heads * self.out_feat)  # Khởi tạo chuẩn hóa với kích thước đầu ra


    def reset_parameters(self):
        '''
        Khởi tạo lại các tham số của tầng GATConv, bao gồm trọng số và bias.
        Sử dụng phương pháp khởi tạo Xavier với hàm kích hoạt ReLU để đảm bảo các trọng số được khởi tạo tốt.
        
        :param self: Đối tượng lớp GATConv
        
        :return: None
        '''
        gain = nn.init.calculate_gain('relu')  # Tính gain cho hàm kích hoạt kiểu ReLU
        nn.init.xavier_normal_(self.edge_fc.weight, gain=gain)  # Khởi tạo trọng số tầng tuyến tính cạnh
        if hasattr(self, 'fc'): # Nếu có tầng tuyến tính cho tất cả nút, khởi tạo trọng số tầng này
            nn.init.xavier_normal_(self.fc.weight, gain=gain)  # Khởi tạo trọng số tầng tuyến tính nút
        else: # Nếu không có tầng tuyến tính cho tất cả nút, khởi tạo trọng số cho các tầng nguồn/đích
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)  # Khởi tạo trọng số tầng tuyến tính nút nguồn
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)  # Khởi tạo trọng số tầng tuyến tính nút đích
        nn.init.xavier_normal_(self.attn_h, gain=gain)  # Khởi tạo tham số chú ý nút nguồn
        nn.init.xavier_normal_(self.attn_e, gain=gain)  # Khởi tạo tham số chú ý cạnh
        nn.init.xavier_normal_(self.attn_t, gain=gain)  # Khởi tạo tham số chú ý nút đích
        if self.bias is not None: # Nếu có bias, khởi tạo bias với giá trị 0
            nn.init.constant_(self.bias, 0)  # Khởi tạo bias bằng 0
        if isinstance(self.res_fc, nn.Linear): # Nếu có tầng tuyến tính cho kết nối dư, khởi tạo trọng số tầng này
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)  # Khởi tạo trọng số tầng tuyến tính dư


    def set_allow_zero_in_degree(self, set_value):
        '''
        Cập nhật trạng thái cho phép các nút có bậc vào bằng 0.
        
        :param set_value: Giá trị True hoặc False để chỉ định trạng thái cho phép
        
        :return: None
        '''
        self.allow_zero_in_degree = set_value  # Cập nhật cờ cho phép


    def forward(self, graph, feat, get_attention=False):
        '''
        Thực hiện lan truyền xuôi của tầng GATConv, tính toán điểm chú ý và thực hiện truyền tin giữa các nút trong đồ thị.
        
        :param graph: Đồ thị đầu vào (DGLGraph)
        :param feat: Đặc trưng đầu vào của nút (tensor hoặc tuple cho src/dst)
        :param get_attention: Có trả về trọng số chú ý hay không (bool)
        
        :return: Đầu ra cuối cùng hoặc cả đầu ra và trọng số chú ý (tuple)
        '''
        edge_feature = graph.edata['attr']  # Lấy đặc trưng cạnh từ đồ thị
        with graph.local_scope():  # Tạo phạm vi cục bộ để tránh thay đổi đồ thị
            if isinstance(feat, tuple): # Nếu đầu vào là tuple, tức là có đặc trưng riêng cho nút nguồn và đích
                src_prefix_shape = feat[0].shape[:-1]  # Hình dạng đặc trưng nguồn (loại trừ chiều đặc trưng)
                dst_prefix_shape = feat[1].shape[:-1]  # Hình dạng đặc trưng đích (loại trừ chiều đặc trưng)
                h_src = self.feat_drop(feat[0])  # Áp dụng dropout cho đặc trưng nút nguồn
                h_dst = self.feat_drop(feat[1])  # Áp dụng dropout cho đặc trưng nút đích
                if not hasattr(self, 'fc_src'): # Nếu không có tầng tuyến tính cho nút nguồn, tức là sử dụng cùng một tầng cho cả hai
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self.n_heads, self.out_feat)  # Biến đổi đặc trưng nguồn và định hình lại
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self.n_heads, self.out_feat)  # Biến đổi đặc trưng đích và định hình lại
                else: # Nếu có tầng tuyến tính riêng cho nút nguồn và đích
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self.n_heads, self.out_feat)  # Biến đổi đặc trưng nguồn và định hình lại
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self.n_heads, self.out_feat)  # Biến đổi đặc trưng đích và định hình lại
            else: # Nếu đầu vào không phải là tuple, tức là sử dụng cùng một đặc trưng cho cả nút nguồn và đích
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]  # Hình dạng đặc trưng (loại trừ chiều đặc trưng)
                h_src = h_dst = self.feat_drop(feat)  # Áp dụng dropout cho đặc trưng nút
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self.n_heads, self.out_feat)  # Biến đổi đặc trưng và định hình lại
                if graph.is_block: # Nếu đồ thị là khối (block), tức là đang trong quá trình huấn luyện
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]  # Chọn đặc trưng nút đích
                    h_dst = h_dst[:graph.number_of_dst_nodes()]  # Chọn đặc trưng thô của nút đích
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]  # Cập nhật hình dạng đích
            edge_prefix_shape = edge_feature.shape[:-1]  # Hình dạng đặc trưng cạnh (loại trừ chiều đặc trưng)
            eh = (feat_src * self.attn_h).sum(-1).unsqueeze(-1)  # Tính điểm chú ý cho nút nguồn
            et = (feat_dst * self.attn_t).sum(-1).unsqueeze(-1)  # Tính điểm chú ý cho nút đích

            graph.srcdata.update({'hs': feat_src, 'eh': eh})  # Lưu đặc trưng nguồn và điểm chú ý vào đồ thị
            graph.dstdata.update({'et': et})  # Lưu điểm chú ý đích vào đồ thị

            feat_edge = self.edge_fc(edge_feature).view(
                *edge_prefix_shape, self.n_heads, self.out_feat)  # Biến đổi đặc trưng cạnh và định hình lại
            ee = (feat_edge * self.attn_e).sum(-1).unsqueeze(-1)  # Tính điểm chú ý cho cạnh

            graph.edata.update({'ee': ee})  # Lưu điểm chú ý cạnh vào đồ thị
            graph.apply_edges(fn.u_add_e('eh', 'ee', 'ee'))  # Cộng điểm chú ý nguồn và cạnh
            graph.apply_edges(fn.e_add_v('ee', 'et', 'e'))  # Cộng điểm chú ý cạnh và đích để được chú ý cuối
            """
            graph.apply_edges(fn.u_add_v('eh', 'et', 'e'))  # Bị chú thích: Cách tính chú ý thay thế (chỉ nguồn + đích)
            """
            e = self.leaky_relu(graph.edata.pop('e'))  # Áp dụng LeakyReLU cho điểm chú ý
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))  # Chuẩn hóa điểm chú ý bằng softmax và áp dụng dropout
            # message passing

            graph.update_all(fn.u_mul_e('hs', 'a', 'm'),  # Nhân đặc trưng nguồn với điểm chú ý
                             fn.sum('m', 'hs'))  # Tổng hợp tin nhắn để cập nhật nút đích

            rst = graph.dstdata['hs'].view(-1, self.n_heads, self.out_feat)  # Định hình lại đặc trưng đầu ra

            if self.bias is not None: # Nếu có bias, cộng bias vào đầu ra
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)), self.n_heads, self.out_feat)  # Cộng bias vào đầu ra

            # residual

            if self.res_fc is not None: # Nếu có kết nối dư, biến đổi đầu vào dư
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self.out_feat)  # Biến đổi đầu vào dư
                rst = rst + resval  # Cộng kết nối dư vào đầu ra

            if self.concat_out: # Nếu nối các đầu ra đa đầu, nối các đầu ra lại với nhau
                rst = rst.flatten(1)  # Nối các đầu ra đa đầu
            else: # Nếu không nối, lấy trung bình các đầu ra đa đầu
                rst = torch.mean(rst, dim=1)  # Lấy trung bình các đầu ra đa đầu

            if self.norm is not None: # Nếu có chuẩn hóa, áp dụng chuẩn hóa cho đầu ra
                rst = self.norm(rst)  # Áp dụng chuẩn hóa

            if self.activation: # Nếu có hàm kích hoạt, áp dụng hàm kích hoạt cho đầu ra
                rst = self.activation(rst)  # Áp dụng hàm kích hoạt

            if get_attention: # Nếu yêu cầu trọng số chú ý, trả về đầu ra và trọng số chú ý
                return rst, graph.edata['a']  # Trả về đầu ra và trọng số chú ý
            else: # Nếu không yêu cầu trọng số chú ý, chỉ trả về đầu ra
                return rst  # Chỉ trả về đầu ra


##########################################################################################################################################
##########################################################################################################################################

######################
## MÃ NGUỒN BAN ĐẦU ##
######################

# import torch
# import torch.nn as nn
# from dgl.ops import edge_softmax
# import dgl.function as fn
# from dgl.utils import expand_as_pair
# from utils.utils import create_activation


# class GAT(nn.Module):
#     def __init__(self,
#                  n_dim,
#                  e_dim,
#                  hidden_dim,
#                  out_dim,
#                  n_layers,
#                  n_heads,
#                  n_heads_out,
#                  activation,
#                  feat_drop,
#                  attn_drop,
#                  negative_slope,
#                  residual,
#                  norm,
#                  concat_out=False,
#                  encoding=False
#                  ):
#         super(GAT, self).__init__()
#         self.out_dim = out_dim
#         self.n_heads = n_heads
#         self.n_layers = n_layers
#         self.gats = nn.ModuleList()
#         self.concat_out = concat_out

#         last_activation = create_activation(activation) if encoding else None
#         last_residual = (encoding and residual)
#         last_norm = norm if encoding else None

#         if self.n_layers == 1:
#             self.gats.append(GATConv(
#                 n_dim, e_dim, out_dim, n_heads_out, feat_drop, attn_drop, negative_slope,
#                 last_residual, norm=last_norm, concat_out=self.concat_out
#             ))
#         else:
#             self.gats.append(GATConv(
#                 n_dim, e_dim, hidden_dim, n_heads, feat_drop, attn_drop, negative_slope,
#                 residual, create_activation(activation),
#                 norm=norm, concat_out=self.concat_out
#             ))
#             for _ in range(1, self.n_layers - 1):
#                 self.gats.append(GATConv(
#                     hidden_dim * self.n_heads, e_dim, hidden_dim, n_heads,
#                     feat_drop, attn_drop, negative_slope,
#                     residual, create_activation(activation),
#                     norm=norm, concat_out=self.concat_out
#                 ))
#             self.gats.append(GATConv(
#                 hidden_dim * self.n_heads, e_dim, out_dim, n_heads_out,
#                 feat_drop, attn_drop, negative_slope,
#                 last_residual, last_activation, norm=last_norm, concat_out=self.concat_out
#             ))
#         self.head = nn.Identity()

#     def forward(self, g, input_feature, return_hidden=False):
#         h = input_feature
#         hidden_list = []
#         for layer in range(self.n_layers):
#             h = self.gats[layer](g, h)
#             hidden_list.append(h)
#         if return_hidden:
#             return self.head(h), hidden_list
#         else:
#             return self.head(h)

#     def reset_classifier(self, num_classes):
#         self.head = nn.Linear(self.num_heads * self.out_dim, num_classes)


# class GATConv(nn.Module):
#     def __init__(self,
#                  in_dim,
#                  e_dim,
#                  out_dim,
#                  n_heads,
#                  feat_drop=0.0,
#                  attn_drop=0.0,
#                  negative_slope=0.2,
#                  residual=False,
#                  activation=None,
#                  allow_zero_in_degree=False,
#                  bias=True,
#                  norm=None,
#                  concat_out=True):
#         super(GATConv, self).__init__()
#         self.n_heads = n_heads
#         self.src_feat, self.dst_feat = expand_as_pair(in_dim)
#         self.edge_feat = e_dim
#         self.out_feat = out_dim
#         self.allow_zero_in_degree = allow_zero_in_degree
#         self.concat_out = concat_out

#         if isinstance(in_dim, tuple):
#             self.fc_node_embedding = nn.Linear(
#                 self.src_feat, self.out_feat * self.n_heads, bias=False)
#             self.fc_src = nn.Linear(self.src_feat, self.out_feat * self.n_heads, bias=False)
#             self.fc_dst = nn.Linear(self.dst_feat, self.out_feat * self.n_heads, bias=False)
#         else:
#             self.fc_node_embedding = nn.Linear(
#                 self.src_feat, self.out_feat * self.n_heads, bias=False)
#             self.fc = nn.Linear(self.src_feat, self.out_feat * self.n_heads, bias=False)
#         self.edge_fc = nn.Linear(self.edge_feat, self.out_feat * self.n_heads, bias=False)
#         self.attn_h = nn.Parameter(torch.FloatTensor(size=(1, self.n_heads, self.out_feat)))
#         self.attn_e = nn.Parameter(torch.FloatTensor(size=(1, self.n_heads, self.out_feat)))
#         self.attn_t = nn.Parameter(torch.FloatTensor(size=(1, self.n_heads, self.out_feat)))
#         self.feat_drop = nn.Dropout(feat_drop)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.leaky_relu = nn.LeakyReLU(negative_slope)
#         if bias:
#             self.bias = nn.Parameter(torch.FloatTensor(size=(1, self.n_heads, self.out_feat)))
#         else:
#             self.register_buffer('bias', None)
#         if residual:
#             if self.dst_feat != self.n_heads * self.out_feat:
#                 self.res_fc = nn.Linear(
#                     self.dst_feat, self.n_heads * self.out_feat, bias=False)
#             else:
#                 self.res_fc = nn.Identity()
#         else:
#             self.register_buffer('res_fc', None)
#         self.reset_parameters()
#         self.activation = activation
#         self.norm = norm
#         if norm is not None:
#             self.norm = norm(self.n_heads * self.out_feat)

#     def reset_parameters(self):
#         gain = nn.init.calculate_gain('relu')
#         nn.init.xavier_normal_(self.edge_fc.weight, gain=gain)
#         if hasattr(self, 'fc'):
#             nn.init.xavier_normal_(self.fc.weight, gain=gain)
#         else:
#             nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
#             nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
#         nn.init.xavier_normal_(self.attn_h, gain=gain)
#         nn.init.xavier_normal_(self.attn_e, gain=gain)
#         nn.init.xavier_normal_(self.attn_t, gain=gain)
#         if self.bias is not None:
#             nn.init.constant_(self.bias, 0)
#         if isinstance(self.res_fc, nn.Linear):
#             nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

#     def set_allow_zero_in_degree(self, set_value):
#         self.allow_zero_in_degree = set_value

#     def forward(self, graph, feat, get_attention=False):
#         edge_feature = graph.edata['attr']
#         with graph.local_scope():
#             if isinstance(feat, tuple):
#                 src_prefix_shape = feat[0].shape[:-1]
#                 dst_prefix_shape = feat[1].shape[:-1]
#                 h_src = self.feat_drop(feat[0])
#                 h_dst = self.feat_drop(feat[1])
#                 if not hasattr(self, 'fc_src'):
#                     feat_src = self.fc(h_src).view(
#                         *src_prefix_shape, self.n_heads, self.out_feat)
#                     feat_dst = self.fc(h_dst).view(
#                         *dst_prefix_shape, self.n_heads, self.out_feat)
#                 else:
#                     feat_src = self.fc_src(h_src).view(
#                         *src_prefix_shape, self.n_heads, self.out_feat)
#                     feat_dst = self.fc_dst(h_dst).view(
#                         *dst_prefix_shape, self.n_heads, self.out_feat)
#             else:
#                 src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
#                 h_src = h_dst = self.feat_drop(feat)
#                 feat_src = feat_dst = self.fc(h_src).view(
#                     *src_prefix_shape, self.n_heads, self.out_feat)
#                 if graph.is_block:
#                     feat_dst = feat_src[:graph.number_of_dst_nodes()]
#                     h_dst = h_dst[:graph.number_of_dst_nodes()]
#                     dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]
#             edge_prefix_shape = edge_feature.shape[:-1]
#             eh = (feat_src * self.attn_h).sum(-1).unsqueeze(-1)
#             et = (feat_dst * self.attn_t).sum(-1).unsqueeze(-1)

#             graph.srcdata.update({'hs': feat_src, 'eh': eh})
#             graph.dstdata.update({'et': et})

#             feat_edge = self.edge_fc(edge_feature).view(
#                 *edge_prefix_shape, self.n_heads, self.out_feat)
#             ee = (feat_edge * self.attn_e).sum(-1).unsqueeze(-1)

#             graph.edata.update({'ee': ee})
#             graph.apply_edges(fn.u_add_e('eh', 'ee', 'ee'))
#             graph.apply_edges(fn.e_add_v('ee', 'et', 'e'))
#             """
#             graph.apply_edges(fn.u_add_v('eh', 'et', 'e'))
#             """
#             e = self.leaky_relu(graph.edata.pop('e'))
#             graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
#             # message passing

#             graph.update_all(fn.u_mul_e('hs', 'a', 'm'),
#                              fn.sum('m', 'hs'))

#             rst = graph.dstdata['hs'].view(-1, self.n_heads, self.out_feat)

#             if self.bias is not None:
#                 rst = rst + self.bias.view(
#                     *((1,) * len(dst_prefix_shape)), self.n_heads, self.out_feat)

#             # residual

#             if self.res_fc is not None:
#                 # Use -1 rather than self._num_heads to handle broadcasting
#                 resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self.out_feat)
#                 rst = rst + resval

#             if self.concat_out:
#                 rst = rst.flatten(1)
#             else:
#                 rst = torch.mean(rst, dim=1)

#             if self.norm is not None:
#                 rst = self.norm(rst)

#                 # activation
#             if self.activation:
#                 rst = self.activation(rst)

#             if get_attention:
#                 return rst, graph.edata['a']
#             else:
#                 return rst
