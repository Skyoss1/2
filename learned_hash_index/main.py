import torch
import pickle
import os
import time
import psutil
from model import MLPHash

def load_hash_index(index_path):
    with open(index_path, 'rb') as f:
        hash_index, id_to_idx = pickle.load(f)
    return hash_index, id_to_idx

def search_review_id(review_id, model, hash_index, id_to_idx, device, num_buckets=1024):
    # Kiểm tra review_id hợp lệ
    if review_id not in id_to_idx:
        return None
    rid_idx = id_to_idx[review_id]

    x_tensor = torch.tensor([[rid_idx]], dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(x_tensor)
        _, predicted_bucket = torch.max(outputs, 1)
        bucket_id = predicted_bucket.item()

    bucket_list = hash_index.get(bucket_id, [])
    if review_id in bucket_list:
        return review_id
    else:
        return None

if __name__ == '__main__':
    model_path = os.path.join('models', 'saved_model.pth')
    index_path = os.path.join('models', 'hash_index.pkl')

    process = psutil.Process(os.getpid())
    
    # Đo bộ nhớ trước khi load index
    mem_before_load = process.memory_info().rss / (1024 * 1024)  # MB

    # Load hash index
    hash_index, id_to_idx = load_hash_index(index_path)

    # Đo bộ nhớ sau khi load index
    mem_after_load = process.memory_info().rss / (1024 * 1024)
    memory_used_load = mem_after_load - mem_before_load
    print(f"Memory used after loading index: {memory_used_load:.2f} MB")

    # Load mô hình
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_buckets = 1024
    model = MLPHash(output_dim=num_buckets).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Đo bộ nhớ sau khi load model (nếu muốn)
    mem_after_model = process.memory_info().rss / (1024 * 1024)
    memory_used_model = mem_after_model - mem_after_load
    print(f"Memory used after loading model: {memory_used_model:.2f} MB")

    # Nhập review_id thủ công
    input_review_id = input("Nhập review_id bạn muốn tìm: ").strip()

    # Đo thời gian và bộ nhớ trước truy vấn
    mem_before_query = process.memory_info().rss / (1024 * 1024)
    start_time = time.time()

    result = search_review_id(input_review_id, model, hash_index, id_to_idx, device)

    end_time = time.time()
    mem_after_query = process.memory_info().rss / (1024 * 1024)

    total_time = end_time - start_time

    # Đo memory usage trong và sau query
    memory_used_query = mem_after_query - mem_before_query

    # Độ chính xác cho 1 query
    accuracy = 1.0 if result is not None else 0.0

    if result is not None:
        print(f"Found review_id: {result}")
    else:
        print("Review_id not found.")

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Query time: {total_time:.6f} seconds")
    print(f"Additional memory used during query: {memory_used_query:.2f} MB")
