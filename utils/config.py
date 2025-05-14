'''
Sử dụng thư viện argparse để xây dựng một đối tượng phân tích tham số dòng lệnh.
Nó định nghĩa một hàm `build_args()` để tạo ra các tham số mà người dùng có thể cung cấp khi chạy ứng dụng từ dòng lệnh.
'''

import argparse


def build_args():
    '''
    Hàm này sử dụng thư viện argparse để xây dựng một đối tượng phân tích tham số dòng lệnh.
    Nó định nghĩa các tham số mà người dùng có thể cung cấp khi chạy ứng dụng từ dòng lệnh.
    
    :return: Đối tượng chứa các tham số đã được phân tích.
    :rtype: argparse.Namespace
    '''
    parser = argparse.ArgumentParser(description="MAGIC")
    parser.add_argument("--dataset", type=str, default="wget")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument("--negative_slope", type=float, default=0.2,
                        help="the negative slope of leaky relu for GAT")
    parser.add_argument("--mask_rate", type=float, default=0.5)
    parser.add_argument("--alpha_l", type=float, default=3, help="`pow`inddex for `sce` loss")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--loss_fn", type=str, default='sce')
    parser.add_argument("--pooling", type=str, default="mean")
    args = parser.parse_args()
    return args
