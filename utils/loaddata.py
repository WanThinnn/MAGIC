'''
Module này định nghĩa các lớp và hàm để tải và xử lý dữ liệu cho một bài toán học máy, cụ thể là một bài toán phân loại đồ thị.
Nó sử dụng các thư viện như DGL (Deep Graph Library) và NetworkX để làm việc với đồ thị.
Bao gồm các lớp `StreamspotDataset` và `WgetDataset` để tải dữ liệu từ các tệp JSON, cũng như các hàm để xử lý và chuyển đổi đồ thị.
Đồng thời bao gồm các hàm để tải dữ liệu đã được xử lý từ các tệp pickle và để chuyển đổi đồ thị thành định dạng mà mô hình có thể sử dụng.

'''

import pickle as pkl
import time
import torch.nn.functional as F
import dgl
import networkx as nx
import json
from tqdm import tqdm
import os


class StreamspotDataset(dgl.data.DGLDataset):
    '''
    Lớp StreamspotDataset kế thừa từ dgl.data.DGLDataset.
    Nó được sử dụng để tải và xử lý dữ liệu từ một tập dữ liệu cụ thể có tên là "streamspot".
    Trong hàm khởi tạo, nó tải dữ liệu từ các tệp JSON và chuyển đổi chúng thành đồ thị DGL.
    Nó cũng gán nhãn cho các đồ thị dựa trên chỉ số của chúng.
    Các phương thức chính bao gồm:
    - `__init__`: Khởi tạo lớp và tải dữ liệu từ tệp JSON.
    - `__getitem__`: Trả về đồ thị và nhãn tương ứng cho một chỉ số cụ thể.
    - `__len__`: Trả về số lượng đồ thị trong tập dữ liệu.
    - `process`: Phương thức này không được sử dụng trong lớp này, nhưng nó có thể được sử dụng để xử lý dữ liệu nếu cần.
    
    '''
    def process(self):
        '''
        Phương thức này không được sử dụng trong lớp này, nhưng có thể được sử dụng để xử lý dữ liệu nếu cần.
        (Nó có thể được sử dụng để thực hiện các bước xử lý dữ liệu bổ sung như chuẩn hóa, biến đổi hoặc tạo các đặc trưng mới từ dữ liệu gốc.)
        '''
        pass

    def __init__(self, name):
        '''
        Hàm khởi tạo lớp StreamspotDataset.
        Nó tải dữ liệu từ các tệp JSON và chuyển đổi chúng thành đồ thị DGL.
        Gán nhãn cho các đồ thị dựa trên chỉ số của chúng.
        
        :param name: Tên của tập dữ liệu (trong trường hợp này là "streamspot").   
        :type name: str
        :raises NotImplementedError: Nếu tên tập dữ liệu không phải là "streamspot".
        
        :return: None
        :rtype: None
        
        '''
        super(StreamspotDataset, self).__init__(name=name) # Gọi hàm khởi tạo của lớp cha
        if name == 'streamspot': # Kiểm tra xem tên tập dữ liệu có phải là "streamspot" không
            path = './data/streamspot' # Đường dẫn đến thư mục chứa dữ liệu
            num_graphs = 600 # Số lượng đồ thị trong tập dữ liệu
            self.graphs = [] # Danh sách để lưu trữ các đồ thị
            self.labels = [] # Danh sách để lưu trữ nhãn của các đồ thị
            print('Loading {} dataset...'.format(name)) # In thông báo đang tải tập dữ liệu
            for i in tqdm(range(num_graphs)): # Duyệt qua từng đồ thị trong tập dữ liệu
                idx = i # Chỉ số của đồ thị
                g = dgl.from_networkx( # Chuyển đổi đồ thị từ định dạng NetworkX sang DGL
                    nx.node_link_graph(json.load(open('{}/{}.json'.format(path, str(idx + 1))))), # Tải đồ thị từ tệp JSON
                    node_attrs=['type'], # Các thuộc tính của nút
                    edge_attrs=['type'] # Các thuộc tính của cạnh
                )
                self.graphs.append(g) # Thêm đồ thị vào danh sách
                if 300 <= idx <= 399: # Gán nhãn cho đồ thị
                    self.labels.append(1) # Nhãn 1 cho các đồ thị trong khoảng từ 300 đến 399
                else:
                    self.labels.append(0) # Nhãn 0 cho các đồ thị còn lại
        else:
            raise NotImplementedError # Nếu tên tập dữ liệu không phải là "streamspot", ném ra lỗi NotImplementedError

    def __getitem__(self, i):
        '''
        Hàm trả về đồ thị và nhãn tương ứng cho một chỉ số cụ thể.
        
        :param i: Chỉ số của đồ thị cần lấy.
        :type i: int
        :return: Đồ thị và nhãn tương ứng.
        :rtype: tuple (dgl.graph, int)
        '''
        return self.graphs[i], self.labels[i]

    def __len__(self):
        '''
        Hàm dùng để trả về số lượng đồ thị trong tập dữ liệu.
        
        :return: Số lượng đồ thị trong tập dữ liệu.
        :rtype: int
        '''
        return len(self.graphs)


class WgetDataset(dgl.data.DGLDataset):
    '''
    Lớp WgetDataset kế thừa từ dgl.data.DGLDataset.
    Nó được sử dụng để tải và xử lý dữ liệu từ một tập dữ liệu cụ thể có tên là "wget".
    Trong hàm khởi tạo, nó tải dữ liệu từ các tệp JSON và chuyển đổi chúng thành đồ thị DGL.
    Nó cũng gán nhãn cho các đồ thị dựa trên chỉ số của chúng.
    Các phương thức chính bao gồm:
    - `__init__`: Khởi tạo lớp và tải dữ liệu từ tệp JSON.
    - `__getitem__`: Trả về đồ thị và nhãn tương ứng cho một chỉ số cụ thể.
    - `__len__`: Trả về số lượng đồ thị trong tập dữ liệu.
    - `process`: Phương thức này không được sử dụng trong lớp này, nhưng nó có thể được sử dụng để xử lý dữ liệu nếu cần.
    
    '''
    def process(self):
        '''
        Hàm này không được sử dụng trong lớp này, nhưng có thể được sử dụng để xử lý dữ liệu nếu cần.
        (Nó có thể được sử dụng để thực hiện các bước xử lý dữ liệu bổ sung như chuẩn hóa, biến đổi hoặc tạo các đặc trưng mới từ dữ liệu gốc.)
        
        :return: None
        :rtype: None
        '''
        pass

    def __init__(self, name):
        '''
        Hàm khởi tạo lớp WgetDataset.
        Nó tải dữ liệu từ các tệp JSON và chuyển đổi chúng thành đồ thị DGL.
        Gán nhãn cho các đồ thị dựa trên chỉ số của chúng.
        
        :param name: Tên của tập dữ liệu (trong trường hợp này là "wget").
        :type name: str
        :raises NotImplementedError: Nếu tên tập dữ liệu không phải là "wget".
        :return: None
        :rtype: None
        '''
        super(WgetDataset, self).__init__(name=name) # Gọi hàm khởi tạo của lớp cha
        if name == 'wget': # Kiểm tra xem tên tập dữ liệu có phải là "wget" không
            path = './data/wget/final' # Đường dẫn đến thư mục chứa dữ liệu
            num_graphs = 150 # Số lượng đồ thị trong tập dữ liệu
            self.graphs = [] # Danh sách để lưu trữ các đồ thị
            self.labels = [] # Danh sách để lưu trữ nhãn của các đồ thị
            print('Loading {} dataset...'.format(name)) # In thông báo đang tải tập dữ liệu
            for i in tqdm(range(num_graphs)): # Duyệt qua từng đồ thị trong tập dữ liệu
                idx = i # Chỉ số của đồ thị
                g = dgl.from_networkx( # Chuyển đổi đồ thị từ định dạng NetworkX sang DGL
                    nx.node_link_graph(json.load(open('{}/{}.json'.format(path, str(idx))))), # Tải đồ thị từ tệp JSON
                    node_attrs=['type'], # Các thuộc tính của nút
                    edge_attrs=['type'] # Các thuộc tính của cạnh
                )
                self.graphs.append(g) # Thêm đồ thị vào danh sách
                if 0 <= idx <= 24: # Gán nhãn cho đồ thị
                    self.labels.append(1) # Nhãn 1 cho các đồ thị trong khoảng từ 0 đến 24
                else: 
                    self.labels.append(0) # Nhãn 0 cho các đồ thị còn lại
        else:
            raise NotImplementedError # Nếu tên tập dữ liệu không phải là "wget", ném ra lỗi NotImplementedError

    def __getitem__(self, i):
        '''
        Hàm trả về đồ thị và nhãn tương ứng cho một chỉ số cụ thể.
        
        :param i: Chỉ số của đồ thị cần lấy.
        :type i: int
        :return: Đồ thị và nhãn tương ứng.
        :rtype: tuple (dgl.graph, int)
        '''
        return self.graphs[i], self.labels[i]

    def __len__(self):
        '''
        Hàm dùng để trả về số lượng đồ thị trong tập dữ liệu.
        
        :return: Số lượng đồ thị trong tập dữ liệu.
        :rtype: int
        
        '''
        return len(self.graphs) 


def load_rawdata(name):
    '''
    Hàm load_rawdata được sử dụng để tải dữ liệu thô từ các tệp đã được xử lý trước đó.
    Nếu tệp đã tồn tại, nó sẽ tải dữ liệu từ tệp đó. Nếu không, nó sẽ tạo một đối tượng của lớp StreamspotDataset hoặc WgetDataset và lưu dữ liệu vào tệp.
    
    :param name: Tên của tập dữ liệu cần tải (có thể là "streamspot" hoặc "wget").
    :type name: str
    :raises NotImplementedError: Nếu tên tập dữ liệu không phải là "streamspot" hoặc "wget".
    :return: Dữ liệu thô đã được tải.
    :rtype: object
    
    
    '''
    if name == 'streamspot': # Kiểm tra xem tên tập dữ liệu có phải là "streamspot" không
        path = './data/streamspot' # Đường dẫn đến thư mục chứa dữ liệu
        if os.path.exists(path + '/graphs.pkl'): # Kiểm tra xem tệp đã tồn tại chưa
            print('Loading processed {} dataset...'.format(name)) # In thông báo đang tải tập dữ liệu
            raw_data = pkl.load(open(path + '/graphs.pkl', 'rb')) # Tải dữ liệu từ tệp
        else:
            raw_data = StreamspotDataset(name) # Tạo đối tượng của lớp StreamspotDataset
            pkl.dump(raw_data, open(path + '/graphs.pkl', 'wb')) # Lưu dữ liệu vào tệp
    elif name == 'wget': # Kiểm tra xem tên tập dữ liệu có phải là "wget" không
        path = './data/wget' # Đường dẫn đến thư mục chứa dữ liệu
        if os.path.exists(path + '/graphs.pkl'): # Kiểm tra xem tệp đã tồn tại chưa
            print('Loading processed {} dataset...'.format(name)) # In thông báo đang tải tập dữ liệu
            raw_data = pkl.load(open(path + '/graphs.pkl', 'rb')) # Tải dữ liệu từ tệp
        else:
            raw_data = WgetDataset(name) # Tạo đối tượng của lớp WgetDataset
            pkl.dump(raw_data, open(path + '/graphs.pkl', 'wb')) # Lưu dữ liệu vào tệp
    else:
        raise NotImplementedError # Nếu tên tập dữ liệu không phải là "streamspot" hoặc "wget", ném ra lỗi NotImplementedError
    return raw_data # Trả về dữ liệu thô đã được tải


def load_batch_level_dataset(dataset_name):
    '''
    Hàm load_batch_level_dataset được sử dụng để tải dữ liệu theo lô từ các tệp đã được xử lý trước đó.
    Nếu tệp đã tồn tại, nó sẽ tải dữ liệu từ tệp đó. Nếu không, nó sẽ tạo một đối tượng của lớp StreamspotDataset hoặc WgetDataset và lưu dữ liệu vào tệp.
    
    :param dataset_name: Tên của tập dữ liệu cần tải (có thể là "streamspot" hoặc "wget").
    :type dataset_name: str
    :raises NotImplementedError: Nếu tên tập dữ liệu không phải là "streamspot" hoặc "wget".
    :return: Dữ liệu theo lô đã được tải.
    :rtype: dict
     
    '''
    dataset = load_rawdata(dataset_name) # Tải dữ liệu thô
    graph, _ = dataset[0] # Lấy đồ thị đầu tiên trong tập dữ liệu
    node_feature_dim = 0 #  Kích thước đặc trưng của nút
    for g, _ in dataset: # Duyệt qua từng đồ thị trong tập dữ liệu
        node_feature_dim = max(node_feature_dim, g.ndata["type"].max().item()) # Tìm kích thước lớn nhất của đặc trưng nút
    edge_feature_dim = 0 # Kích thước đặc trưng của cạnh
    for g, _ in dataset: # Duyệt qua từng đồ thị trong tập dữ liệu
        edge_feature_dim = max(edge_feature_dim, g.edata["type"].max().item()) # Tìm kích thước lớn nhất của đặc trưng cạnh
    node_feature_dim += 1 # Tăng kích thước đặc trưng nút lên 1
    edge_feature_dim += 1 # Tăng kích thước đặc trưng cạnh lên 1
    full_dataset = [i for i in range(len(dataset))] # Tạo danh sách chứa tất cả các chỉ số của đồ thị
    train_dataset = [i for i in range(len(dataset)) if dataset[i][1] == 0] # Tạo danh sách chứa các chỉ số của đồ thị huấn luyện
    print('[n_graph, n_node_feat, n_edge_feat]: [{}, {}, {}]'.format(len(dataset), node_feature_dim, edge_feature_dim)) # In thông báo về số lượng đồ thị, kích thước đặc trưng nút và kích thước đặc trưng cạnh

    return {'dataset': dataset, # Trả về tập dữ liệu
            'train_index': train_dataset, # Trả về chỉ số của đồ thị huấn luyện
            'full_index': full_dataset, # Trả về chỉ số của tất cả các đồ thị
            'n_feat': node_feature_dim, # Kích thước đặc trưng nút
            'e_feat': edge_feature_dim} # Trả về kích thước đặc trưng cạnh


def transform_graph(g, node_feature_dim, edge_feature_dim):
    '''
    Hàm transform_graph được sử dụng để chuyển đổi đồ thị thành định dạng mà mô hình có thể sử dụng.
    Nó tạo ra các đặc trưng cho các nút và cạnh bằng cách sử dụng one-hot encoding.
    
    :param g: Đồ thị cần chuyển đổi.
    :type g: dgl.graph
    :param node_feature_dim: Kích thước đặc trưng của nút.
    :type node_feature_dim: int
    :param edge_feature_dim: Kích thước đặc trưng của cạnh.
    :type edge_feature_dim: int
    :return: Đồ thị đã được chuyển đổi.
    :rtype: dgl.graph
    
    '''
    new_g = g.clone() # Tạo một bản sao của đồ thị
    new_g.ndata["attr"] = F.one_hot(g.ndata["type"].view(-1), num_classes=node_feature_dim).float() # Tạo đặc trưng cho các nút bằng one-hot encoding
    new_g.edata["attr"] = F.one_hot(g.edata["type"].view(-1), num_classes=edge_feature_dim).float() # Tạo đặc trưng cho các cạnh bằng one-hot encoding
    return new_g # Trả về đồ thị đã được chuyển đổi


def preload_entity_level_dataset(path):
    '''
    Hàm preload_entity_level_dataset được sử dụng để tải và chuyển đổi dữ liệu theo lô từ các tệp đã được xử lý trước đó.
    Nếu tệp đã tồn tại, nó sẽ tải dữ liệu từ tệp đó. Nếu không, nó sẽ tạo một đối tượng của lớp StreamspotDataset hoặc WgetDataset và lưu dữ liệu vào tệp.
    
    :param path: Đường dẫn đến thư mục chứa dữ liệu.
    :type path: str
    :raises NotImplementedError: Nếu tên tập dữ liệu không phải là "streamspot" hoặc "wget".
    :return: None
    
    '''
    path = './data/' + path # Đường dẫn đến thư mục chứa dữ liệu
    if os.path.exists(path + '/metadata.json'): # Kiểm tra xem tệp đã tồn tại chưa
        pass # Nếu tệp đã tồn tại, không làm gì cả
    else: # Nếu tệp chưa tồn tại, tiến hành tải và chuyển đổi dữ liệu
        print('transforming') # In thông báo đang chuyển đổi dữ liệu
        train_gs = [dgl.from_networkx( # Chuyển đổi đồ thị từ định dạng NetworkX sang DGL
            nx.node_link_graph(g), # Tải đồ thị từ tệp JSON
            node_attrs=['type'], # Các thuộc tính của nút
            edge_attrs=['type'] # Các thuộc tính của cạnh
        ) for g in pkl.load(open(path + '/train.pkl', 'rb'))] # Tải dữ liệu huấn luyện từ tệp pickle
        print('transforming') # In thông báo đang chuyển đổi dữ liệu
        test_gs = [dgl.from_networkx( # Chuyển đổi đồ thị từ định dạng NetworkX sang DGL
            nx.node_link_graph(g), # Tải đồ thị từ tệp JSON
            node_attrs=['type'], # Các thuộc tính của nút
            edge_attrs=['type'] # Các thuộc tính của cạnh
        ) for g in pkl.load(open(path + '/test.pkl', 'rb'))] # Tải dữ liệu kiểm tra từ tệp pickle
        malicious = pkl.load(open(path + '/malicious.pkl', 'rb')) # Tải dữ liệu độc hại từ tệp pickle

        node_feature_dim = 0 # Kích thước đặc trưng của nút
        for g in train_gs: # Duyệt qua từng đồ thị trong tập dữ liệu huấn luyện
            node_feature_dim = max(g.ndata["type"].max().item(), node_feature_dim) # Tìm kích thước lớn nhất của đặc trưng nút
        for g in test_gs: # Duyệt qua từng đồ thị trong tập dữ liệu kiểm tra
            node_feature_dim = max(g.ndata["type"].max().item(), node_feature_dim) # Tìm kích thước lớn nhất của đặc trưng nút
        node_feature_dim += 1 # Tăng kích thước đặc trưng nút lên 1
        edge_feature_dim = 0 # Kích thước đặc trưng của cạnh
        for g in train_gs: # Duyệt qua từng đồ thị trong tập dữ liệu huấn luyện
            edge_feature_dim = max(g.edata["type"].max().item(), edge_feature_dim) # Tìm kích thước lớn nhất của đặc trưng cạnh
        for g in test_gs: # Duyệt qua từng đồ thị trong tập dữ liệu kiểm tra
            edge_feature_dim = max(g.edata["type"].max().item(), edge_feature_dim) # Tìm kích thước lớn nhất của đặc trưng cạnh
        edge_feature_dim += 1 # Tăng kích thước đặc trưng cạnh lên 1
        result_test_gs = [] # Danh sách để lưu trữ các đồ thị kiểm tra đã được chuyển đổi
        for g in test_gs: # Duyệt qua từng đồ thị trong tập dữ liệu kiểm tra
            g = transform_graph(g, node_feature_dim, edge_feature_dim) # Chuyển đổi đồ thị thành định dạng mà mô hình có thể sử dụng
            result_test_gs.append(g) # Thêm đồ thị đã được chuyển đổi vào danh sách
        result_train_gs = [] # Danh sách để lưu trữ các đồ thị huấn luyện đã được chuyển đổi 
        for g in train_gs:  # Duyệt qua từng đồ thị trong tập dữ liệu huấn luyện
            g = transform_graph(g, node_feature_dim, edge_feature_dim) # Chuyển đổi đồ thị thành định dạng mà mô hình có thể sử dụng
            result_train_gs.append(g) # Thêm đồ thị đã được chuyển đổi vào danh sách
        metadata = { # Tạo một từ điển chứa thông tin về tập dữ liệu
            'node_feature_dim': node_feature_dim, # Kích thước đặc trưng của nút
            'edge_feature_dim': edge_feature_dim, # Kích thước đặc trưng của cạnh
            'malicious': malicious, # Dữ liệu độc hại
            'n_train': len(result_train_gs), # Số lượng đồ thị trong tập dữ liệu huấn luyện
            'n_test': len(result_test_gs) # Số lượng đồ thị trong tập dữ liệu kiểm tra
        }
        with open(path + '/metadata.json', 'w', encoding='utf-8') as f: # Mở tệp metadata.json để ghi
            json.dump(metadata, f) # Lưu thông tin về tập dữ liệu vào tệp
        for i, g in enumerate(result_train_gs): # Duyệt qua từng đồ thị trong tập dữ liệu huấn luyện
            with open(path + '/train{}.pkl'.format(i), 'wb') as f: # Mở tệp để ghi
                pkl.dump(g, f) # Lưu đồ thị đã được chuyển đổi vào tệp
        for i, g in enumerate(result_test_gs): # Duyệt qua từng đồ thị trong tập dữ liệu kiểm tra
            with open(path + '/test{}.pkl'.format(i), 'wb') as f: # Mở tệp để ghi
                pkl.dump(g, f) # Lưu đồ thị đã được chuyển đổi vào tệp


def load_metadata(path):
    '''
    Hàm load_metadata được sử dụng để tải thông tin về tập dữ liệu từ tệp metadata.json.
    Nếu tệp đã tồn tại, nó sẽ tải thông tin từ tệp đó. Nếu không, nó sẽ gọi hàm preload_entity_level_dataset để tạo và lưu thông tin vào tệp.
    
    :param path: Đường dẫn đến thư mục chứa dữ liệu.
    :type path: str
    :return: Thông tin về tập dữ liệu.
    :rtype: dict
    
    '''
    preload_entity_level_dataset(path) # Gọi hàm preload_entity_level_dataset để tạo và lưu thông tin vào tệp
    with open('./data/' + path + '/metadata.json', 'r', encoding='utf-8') as f: # Mở tệp metadata.json để đọc
        metadata = json.load(f) # Tải thông tin về tập dữ liệu từ tệp
    return metadata # Trả về thông tin về tập dữ liệu


def load_entity_level_dataset(path, t, n):
    '''
    Hàm load_entity_level_dataset được sử dụng để tải dữ liệu theo lô từ các tệp đã được xử lý trước đó.
    Nếu tệp đã tồn tại, nó sẽ tải dữ liệu từ tệp đó. Nếu không, nó sẽ gọi hàm preload_entity_level_dataset để tạo và lưu dữ liệu vào tệp.
    
    :param path: Đường dẫn đến thư mục chứa dữ liệu.
    :type path: str
    :param t: Chỉ số của đồ thị cần tải.
    :type t: int
    :param n: Chỉ số của đồ thị cần tải.
    :type n: int
    :return: Dữ liệu theo lô đã được tải.
    :rtype: dgl.graph
    '''
    preload_entity_level_dataset(path) # Gọi hàm preload_entity_level_dataset để tạo và lưu thông tin vào tệp
    with open('./data/' + path + '/{}{}.pkl'.format(t, n), 'rb') as f: # Mở tệp để đọc
        data = pkl.load(f) # Tải dữ liệu từ tệp
    return data # Trả về dữ liệu đã được tải
