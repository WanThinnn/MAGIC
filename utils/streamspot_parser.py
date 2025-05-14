'''
Module này được sử dụng để phân tích và chuyển đổi dữ liệu từ định dạng tsv sang định dạng json cho các đồ thị trong dự án StreamSpot.
'''

import networkx as nx
from tqdm import tqdm
import json
raw_path = '../data/streamspot/'

NUM_GRAPHS = 600 # Số lượng đồ thị tối đa
node_type_dict = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'] # Danh sách các loại nút
edge_type_dict = ['i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 
                  'q', 't', 'u', 'v', 'w', 'y', 'z', 'A', 'C', 'D', 'E', 'G'] # Danh sách các loại cạnh
node_type_set = set(node_type_dict) # Tập hợp các loại nút
edge_type_set = set(edge_type_dict) # Tập hợp các loại cạnh

count_graph = 0 # Biến đếm số lượng đồ thị đã được xử lý
with open(raw_path + 'all.tsv', 'r', encoding='utf-8') as f: # Mở tệp tsv chứa dữ liệu
    lines = f.readlines() # Đọc tất cả các dòng trong tệp
    g = nx.DiGraph() # Tạo một đồ thị có hướng
    node_map = {} # Từ điển ánh xạ các nút
    count_node = 0 # Biến đếm số lượng nút đã được xử lý
    for line in tqdm(lines): # Duyệt qua từng dòng trong tệp
        src, src_type, dst, dst_type, etype, graph_id = line.strip('\n').split('\t') # Tách các trường trong dòng
        graph_id = int(graph_id) # Chuyển đổi id đồ thị thành số nguyên
        if src_type not in node_type_set or dst_type not in node_type_set: # Kiểm tra loại nút
            continue # Nếu loại nút không hợp lệ, bỏ qua dòng này
        if etype not in edge_type_set: # Kiểm tra loại cạnh
            continue # Nếu loại cạnh không hợp lệ, bỏ qua dòng này
        if graph_id != count_graph: # Kiểm tra xem có phải là đồ thị mới không
            count_graph += 1 # Tăng biến đếm đồ thị
            for n in g.nodes(): # Duyệt qua tất cả các nút trong đồ thị
                g.nodes[n]['type'] = node_type_dict.index(g.nodes[n]['type']) # Chuyển đổi loại nút thành chỉ số
            for e in g.edges(): # Duyệt qua tất cả các cạnh trong đồ thị
                g.edges[e]['type'] = edge_type_dict.index(g.edges[e]['type']) # Chuyển đổi loại cạnh thành chỉ số
            f1 = open(raw_path + str(count_graph) + '.json', 'w', encoding='utf-8') # Mở tệp json để lưu đồ thị
            json.dump(nx.node_link_data(g), f1) # Lưu đồ thị vào tệp json
            assert graph_id == count_graph # Kiểm tra xem id đồ thị có đúng không
            g = nx.DiGraph() # Tạo một đồ thị mới
            count_node = 0 # Đặt lại biến đếm nút
        if src not in node_map: # Kiểm tra xem nút nguồn đã được ánh xạ chưa
            node_map[src] = count_node # Nếu chưa, ánh xạ nút nguồn
            g.add_node(count_node, type=src_type) # Thêm nút nguồn vào đồ thị
            count_node += 1 # Tăng biến đếm nút
        if dst not in node_map: # Kiểm tra xem nút đích đã được ánh xạ chưa
            node_map[dst] = count_node # Nếu chưa, ánh xạ nút đích
            g.add_node(count_node, type=dst_type) # Thêm nút đích vào đồ thị
            count_node += 1 # Tăng biến đếm nút
        if not g.has_edge(node_map[src], node_map[dst]): # Kiểm tra xem cạnh đã tồn tại chưa
            g.add_edge(node_map[src], node_map[dst], type=etype) # Nếu chưa, thêm cạnh vào đồ thị
    count_graph += 1 # Tăng biến đếm đồ thị
    for n in g.nodes(): # Duyệt qua tất cả các nút trong đồ thị
        g.nodes[n]['type'] = node_type_dict.index(g.nodes[n]['type']) # Chuyển đổi loại nút thành chỉ số
    for e in g.edges(): # Duyệt qua tất cả các cạnh trong đồ thị
        g.edges[e]['type'] = edge_type_dict.index(g.edges[e]['type']) # Chuyển đổi loại cạnh thành chỉ số
    f1 = open(raw_path + str(count_graph) + '.json', 'w', encoding='utf-8') # Mở tệp json để lưu đồ thị
    json.dump(nx.node_link_data(g), f1) # Lưu đồ thị vào tệp json
