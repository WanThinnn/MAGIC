# Import các thư viện cần thiết
import argparse  # Thư viện để xử lý tham số dòng lệnh
import json      # Thư viện để xử lý dữ liệu JSON
import os        # Thư viện để tương tác với hệ thống file
import random    # Thư viện để tạo số ngẫu nhiên
import re        # Thư viện để xử lý biểu thức chính quy

from tqdm import tqdm  # Thư viện để hiển thị thanh tiến trình
import networkx as nx  # Thư viện để xử lý đồ thị
import pickle as pkl   # Thư viện để lưu/đọc dữ liệu dạng binary

# Các biến toàn cục để lưu trữ thông tin về loại node và edge
node_type_dict = {}  # Dictionary ánh xạ loại node sang ID số (ví dụ: 'Process' -> 0, 'File' -> 1)
edge_type_dict = {}  # Dictionary ánh xạ loại edge sang ID số (ví dụ: 'READ' -> 0, 'WRITE' -> 1)
node_type_cnt = 0    # Bộ đếm số lượng loại node, tăng dần khi gặp loại node mới
edge_type_cnt = 0    # Bộ đếm số lượng loại edge, tăng dần khi gặp loại edge mới

# Metadata định nghĩa các file dữ liệu cho từng dataset
# Mỗi dataset có 2 phần: train (dữ liệu huấn luyện) và test (dữ liệu kiểm thử)
metadata = {
    'trace':{  # Dataset trace - chứa dữ liệu về các hoạt động hệ thống
        'train': ['ta1-trace-e3-official-1.json', 'ta1-trace-e3-official-1.json.1', 'ta1-trace-e3-official-1.json.2', 'ta1-trace-e3-official-1.json.3'],
        'test': ['ta1-trace-e3-official-1.json', 'ta1-trace-e3-official-1.json.1', 'ta1-trace-e3-official-1.json.2', 'ta1-trace-e3-official-1.json.3', 'ta1-trace-e3-official-1.json.4']
    },
    'theia':{  # Dataset theia - chứa dữ liệu về các hoạt động mạng
            'train': ['ta1-theia-e3-official-6r.json', 'ta1-theia-e3-official-6r.json.1', 'ta1-theia-e3-official-6r.json.2', 'ta1-theia-e3-official-6r.json.3'],
            'test': ['ta1-theia-e3-official-6r.json.8']
    },
    'cadets':{  # Dataset cadets - chứa dữ liệu về các hoạt động bảo mật
            'train': ['ta1-cadets-e3-official.json','ta1-cadets-e3-official.json.1', 'ta1-cadets-e3-official.json.2', 'ta1-cadets-e3-official-2.json.1'],
            'test': ['ta1-cadets-e3-official-2.json']
    }
}

# Các pattern regex để trích xuất thông tin từ dữ liệu CDM (Cyber Data Model)
pattern_uuid = re.compile(r'uuid\":\"(.*?)\"')  # Pattern để lấy UUID của entity
pattern_src = re.compile(r'subject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')  # Pattern để lấy UUID của node nguồn
pattern_dst1 = re.compile(r'predicateObject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')  # Pattern để lấy UUID của node đích thứ nhất
pattern_dst2 = re.compile(r'predicateObject2\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')  # Pattern để lấy UUID của node đích thứ hai
pattern_type = re.compile(r'type\":\"(.*?)\"')  # Pattern để lấy loại của entity (Process, File, etc.)
pattern_time = re.compile(r'timestampNanos\":(.*?),')  # Pattern để lấy thời gian của sự kiện (đơn vị nanosecond)
pattern_file_name = re.compile(r'map\":\{\"path\":\"(.*?)\"')  # Pattern để lấy đường dẫn file
pattern_process_name = re.compile(r'map\":\{\"name\":\"(.*?)\"')  # Pattern để lấy tên process
pattern_netflow_object_name = re.compile(r'remoteAddress\":\"(.*?)\"')  # Pattern để lấy địa chỉ IP remote

def read_single_graph(dataset, malicious, path, test=False):
    """
    Đọc và chuyển đổi một file đồ thị đơn lẻ thành đồ thị NetworkX.
    
    :param dataset: dataset
    :param malicious: danh sách các entity độc hại
    :param path: đường dẫn đến file đồ thị
    :param test: biến boolean để xác định có phải là dữ liệu kiểm thử hay không
    :return: node_map, g
        - node_map: dictionary ánh xạ UUID sang ID số của node
        - g: đồ thị NetworkX đã được chuyển đổi
    :rtype: tuple
        
    
    """
    global node_type_cnt, edge_type_cnt  # Sử dụng biến toàn cục để đếm loại node và edge
    g = nx.DiGraph()  # Tạo đồ thị có hướng mới
    print('converting {} ...'.format(path))  # In thông báo đang xử lý file nào
    path = '../data/{}/'.format(dataset) + path + '.txt'  # Tạo đường dẫn đầy đủ đến file
    f = open(path, 'r')  # Mở file để đọc
    lines = []  # List để lưu các cạnh của đồ thị
    for l in f.readlines():  # Đọc từng dòng trong file
        split_line = l.split('\t')  # Tách dòng thành các trường dữ liệu
        src, src_type, dst, dst_type, edge_type, ts = split_line  # Gán giá trị cho các biến
        ts = int(ts)  # Chuyển timestamp sang số nguyên
        
        # Xử lý dữ liệu train: bỏ qua các cạnh liên quan đến entity độc hại (trừ MemoryObject)
        if not test:
            if src in malicious or dst in malicious:
                if src in malicious and src_type != 'MemoryObject':
                    continue
                if dst in malicious and dst_type != 'MemoryObject':
                    continue

        # Cập nhật dictionary loại node và edge
        if src_type not in node_type_dict:
            node_type_dict[src_type] = node_type_cnt
            node_type_cnt += 1
        if dst_type not in node_type_dict:
            node_type_dict[dst_type] = node_type_cnt
            node_type_cnt += 1
        if edge_type not in edge_type_dict:
            edge_type_dict[edge_type] = edge_type_cnt
            edge_type_cnt += 1

        # Xử lý hướng của cạnh dựa vào loại edge
        if 'READ' in edge_type or 'RECV' in edge_type or 'LOAD' in edge_type:
            lines.append([dst, src, dst_type, src_type, edge_type, ts])  # Đảo ngược hướng cho các edge đọc
        else:
            lines.append([src, dst, src_type, dst_type, edge_type, ts])  # Giữ nguyên hướng cho các edge khác

    lines.sort(key=lambda l: l[5])  # Sắp xếp các cạnh theo thời gian

    # Tạo mapping từ UUID sang node ID và xây dựng đồ thị
    node_map = {}  # Dictionary ánh xạ UUID sang node ID
    node_type_map = {}  # Dictionary lưu loại của mỗi node
    node_cnt = 0  # Bộ đếm node
    node_list = []  # List lưu danh sách UUID của các node
    for l in lines:
        src, dst, src_type, dst_type, edge_type = l[:5]  # Lấy thông tin của cạnh
        src_type_id = node_type_dict[src_type]  # Lấy ID của loại node nguồn
        dst_type_id = node_type_dict[dst_type]  # Lấy ID của loại node đích
        edge_type_id = edge_type_dict[edge_type]  # Lấy ID của loại edge

        # Thêm node nguồn nếu chưa tồn tại
        if src not in node_map:
            node_map[src] = node_cnt
            g.add_node(node_cnt, type=src_type_id)  # Thêm node với thuộc tính type
            node_list.append(src)
            node_type_map[src] = src_type
            node_cnt += 1

        # Thêm node đích nếu chưa tồn tại
        if dst not in node_map:
            node_map[dst] = node_cnt
            g.add_node(node_cnt, type=dst_type_id)
            node_type_map[dst] = dst_type
            node_list.append(dst)
            node_cnt += 1

        # Thêm cạnh nếu chưa tồn tại
        if not g.has_edge(node_map[src], node_map[dst]):
            g.add_edge(node_map[src], node_map[dst], type=edge_type_id)

    return node_map, g  # Trả về mapping và đồ thị

def preprocess_dataset(dataset):
    """
    Tiền xử lý dataset bằng cách đọc các file JSON và chuyển đổi thành định dạng đồ thị.
    
    :param dataset: tên dataset cần tiền xử lý ('trace', 'theia', hoặc 'cadets')
    :type dataset: str
    :return: None
    :rtype: None
    """
    id_nodetype_map = {}  # Dictionary ánh xạ UUID sang loại node
    id_nodename_map = {}  # Dictionary ánh xạ UUID sang tên node

    # Duyệt qua tất cả file JSON trong thư mục dataset
    for file in os.listdir('../data/{}/'.format(dataset)):
        if 'json' in file and not '.txt' in file and not 'names' in file and not 'types' in file and not 'metadata' in file:
            print('reading {} ...'.format(file))
            f = open('../data/{}/'.format(dataset) + file, 'r', encoding='utf-8')
            
            # Đọc từng dòng trong file JSON
            for line in tqdm(f):
                # Bỏ qua các dòng không cần thiết
                if 'com.bbn.tc.schema.avro.cdm18.Event' in line or 'com.bbn.tc.schema.avro.cdm18.Host' in line: continue
                if 'com.bbn.tc.schema.avro.cdm18.TimeMarker' in line or 'com.bbn.tc.schema.avro.cdm18.StartMarker' in line: continue
                if 'com.bbn.tc.schema.avro.cdm18.UnitDependency' in line or 'com.bbn.tc.schema.avro.cdm18.EndMarker' in line: continue
                
                # Kiểm tra và lấy UUID
                if len(pattern_uuid.findall(line)) == 0: print(line)
                uuid = pattern_uuid.findall(line)[0]
                subject_type = pattern_type.findall(line)

                # Xác định loại của entity
                if len(subject_type) < 1:
                    if 'com.bbn.tc.schema.avro.cdm18.MemoryObject' in line:
                        subject_type = 'MemoryObject'
                    if 'com.bbn.tc.schema.avro.cdm18.NetFlowObject' in line:
                        subject_type = 'NetFlowObject'
                    if 'com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject' in line:
                        subject_type = 'UnnamedPipeObject'
                else:
                    subject_type = subject_type[0]

                # Bỏ qua các entity không cần thiết
                if uuid == '00000000-0000-0000-0000-000000000000' or subject_type in ['SUBJECT_UNIT']:
                    continue

                # Lưu thông tin về loại và tên của entity
                id_nodetype_map[uuid] = subject_type
                if 'FILE' in subject_type and len(pattern_file_name.findall(line)) > 0:
                    id_nodename_map[uuid] = pattern_file_name.findall(line)[0]
                elif subject_type == 'SUBJECT_PROCESS' and len(pattern_process_name.findall(line)) > 0:
                    id_nodename_map[uuid] = pattern_process_name.findall(line)[0]
                elif subject_type == 'NetFlowObject' and len(pattern_netflow_object_name.findall(line)) > 0:
                    id_nodename_map[uuid] = pattern_netflow_object_name.findall(line)[0]

    # Xử lý từng file trong metadata
    for key in metadata[dataset]:
        for file in metadata[dataset][key]:
            if os.path.exists('../data/{}/'.format(dataset) + file + '.txt'):
                continue

            # Mở file để đọc và ghi
            f = open('../data/{}/'.format(dataset) + file, 'r', encoding='utf-8')
            fw = open('../data/{}/'.format(dataset) + file + '.txt', 'w', encoding='utf-8')
            print('processing {} ...'.format(file))

            # Xử lý từng dòng trong file
            for line in tqdm(f):
                if 'com.bbn.tc.schema.avro.cdm18.Event' in line:
                    # Trích xuất thông tin từ event
                    edgeType = pattern_type.findall(line)[0]
                    timestamp = pattern_time.findall(line)[0]
                    srcId = pattern_src.findall(line)

                    if len(srcId) == 0: continue
                    srcId = srcId[0]
                    if not srcId in id_nodetype_map:
                        continue
                    srcType = id_nodetype_map[srcId]

                    # Xử lý node đích thứ nhất
                    dstId1 = pattern_dst1.findall(line)
                    if len(dstId1) > 0 and dstId1[0] != 'null':
                        dstId1 = dstId1[0]
                        if not dstId1 in id_nodetype_map:
                            continue
                        dstType1 = id_nodetype_map[dstId1]
                        # Ghi thông tin cạnh vào file
                        this_edge1 = str(srcId) + '\t' + str(srcType) + '\t' + str(dstId1) + '\t' + str(dstType1) + '\t' + str(edgeType) + '\t' + str(timestamp) + '\n'
                        fw.write(this_edge1)

                    # Xử lý node đích thứ hai
                    dstId2 = pattern_dst2.findall(line)
                    if len(dstId2) > 0 and dstId2[0] != 'null':
                        dstId2 = dstId2[0]
                        if not dstId2 in id_nodetype_map:
                            continue
                        dstType2 = id_nodetype_map[dstId2]
                        # Ghi thông tin cạnh vào file
                        this_edge2 = str(srcId) + '\t' + str(srcType) + '\t' + str(dstId2) + '\t' + str(dstType2) + '\t' + str(edgeType) + '\t' + str(timestamp) + '\n'
                        fw.write(this_edge2)

            fw.close()
            f.close()

    # Lưu thông tin về tên và loại node vào file JSON
    if len(id_nodename_map) != 0:
        fw = open('../data/{}/'.format(dataset) + 'names.json', 'w', encoding='utf-8')
        json.dump(id_nodename_map, fw)
    if len(id_nodetype_map) != 0:
        fw = open('../data/{}/'.format(dataset) + 'types.json', 'w', encoding='utf-8')
        json.dump(id_nodetype_map, fw)

def read_graphs(dataset):
    """
    Đọc và xử lý toàn bộ đồ thị cho một dataset, bao gồm cả dữ liệu train và test.
    
    :param dataset: tên dataset cần xử lý ('trace', 'theia', hoặc 'cadets')
    :type dataset: str
    :return: None
    :rtype: None
    """
    # Đọc danh sách các entity độc hại
    malicious_entities = '../data/{}/{}.txt'.format(dataset, dataset)
    f = open(malicious_entities, 'r')
    malicious_entities = set()  # Tạo set để lưu các entity độc hại
    for l in f.readlines():
        malicious_entities.add(l.lstrip().rstrip())  # Thêm entity vào set, loại bỏ khoảng trắng

    # Tiền xử lý dataset
    preprocess_dataset(dataset)

    # Xử lý dữ liệu train
    train_gs = []  # List lưu các đồ thị train
    for file in metadata[dataset]['train']:
        _, train_g = read_single_graph(dataset, malicious_entities, file, False)
        train_gs.append(train_g)

    # Xử lý dữ liệu test
    test_gs = []  # List lưu các đồ thị test
    test_node_map = {}  # Dictionary ánh xạ UUID sang node ID cho dữ liệu test
    count_node = 0  # Bộ đếm node
    for file in metadata[dataset]['test']:
        node_map, test_g = read_single_graph(dataset, malicious_entities, file, True)
        assert len(node_map) == test_g.number_of_nodes()  # Kiểm tra số lượng node
        test_gs.append(test_g)
        # Cập nhật mapping cho dữ liệu test
        for key in node_map:
            if key not in test_node_map:
                test_node_map[key] = node_map[key] + count_node
        count_node += test_g.number_of_nodes()

    # Xử lý thông tin về các entity độc hại
    if os.path.exists('../data/{}/names.json'.format(dataset)) and os.path.exists('../data/{}/types.json'.format(dataset)):
        # Đọc thông tin về tên và loại node
        with open('../data/{}/names.json'.format(dataset), 'r', encoding='utf-8') as f:
            id_nodename_map = json.load(f)
        with open('../data/{}/types.json'.format(dataset), 'r', encoding='utf-8') as f:
            id_nodetype_map = json.load(f)

        # Ghi thông tin về các entity độc hại
        f = open('../data/{}/malicious_names.txt'.format(dataset), 'w', encoding='utf-8')
        final_malicious_entities = []  # List lưu ID của các entity độc hại
        malicious_names = []  # List lưu tên của các entity độc hại
        for e in malicious_entities:
            # Chỉ xử lý các entity có trong dữ liệu test và không phải MemoryObject/UnnamedPipeObject
            if e in test_node_map and e in id_nodetype_map and id_nodetype_map[e] != 'MemoryObject' and id_nodetype_map[e] != 'UnnamedPipeObject':
                final_malicious_entities.append(test_node_map[e])
                if e in id_nodename_map:
                    malicious_names.append(id_nodename_map[e])
                    f.write('{}\t{}\n'.format(e, id_nodename_map[e]))  # Ghi UUID và tên
                else:
                    malicious_names.append(e)
                    f.write('{}\t{}\n'.format(e, e))  # Ghi UUID nếu không có tên
    else:
        # Trường hợp không có file names.json và types.json
        f = open('../data/{}/malicious_names.txt'.format(dataset), 'w', encoding='utf-8')
        final_malicious_entities = []
        malicious_names = []
        for e in malicious_entities:
            if e in test_node_map:
                final_malicious_entities.append(test_node_map[e])
                malicious_names.append(e)
                f.write('{}\t{}\n'.format(e, e))  # Ghi UUID

    # Lưu kết quả vào các file pickle
    pkl.dump((final_malicious_entities, malicious_names), open('../data/{}/malicious.pkl'.format(dataset), 'wb'))
    pkl.dump([nx.node_link_data(train_g) for train_g in train_gs], open('../data/{}/train.pkl'.format(dataset), 'wb'))
    pkl.dump([nx.node_link_data(test_g) for test_g in test_gs], open('../data/{}/test.pkl'.format(dataset), 'wb'))

if __name__ == '__main__':
    # Parser command line arguments
    parser = argparse.ArgumentParser(description='CDM Parser')  # Tạo parser cho tham số dòng lệnh
    parser.add_argument("--dataset", type=str, default="trace",
                      help="Dataset để xử lý (trace, theia, hoặc cadets)")  # Thêm tham số dataset
    args = parser.parse_args()  # Phân tích tham số dòng lệnh
    
    # Kiểm tra dataset có hợp lệ không
    if args.dataset not in ['trace', 'theia', 'cadets']:
        raise NotImplementedError("Dataset không được hỗ trợ")  # Báo lỗi nếu dataset không hợp lệ
    
    # Xử lý dataset
    read_graphs(args.dataset)  # Gọi hàm xử lý dataset

