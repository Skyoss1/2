import os
import torch
from torch.utils.data import DataLoader
from utils import load_review_ids, create_id_mapping, bucket_function
from model import MLPHash

# Hàm tạo hash-index dựa trên mô hình
def build_hash_index(model_path, data_path, limit=None, num_buckets=1024):
    # Load dữ liệu
    review_ids = load_review_ids(data_path, limit=limit)
    unique_review_ids, id_to_idx = create_id_mapping(review_ids)

    # Load mô hình
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPHash(output_dim=num_buckets).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Tạo hash index: dict {bucket_id: [list_of_review_ids]}
    hash_index = {b: [] for b in range(num_buckets)}

    # Gán mỗi review_id vào bucket bằng mô hình
    with torch.no_grad():
        for rid in unique_review_ids:
            rid_idx = id_to_idx[rid]
            x_tensor = torch.tensor([[rid_idx]], dtype=torch.float32).to(device)
            outputs = model(x_tensor)
            _, predicted_bucket = torch.max(outputs, 1)
            bucket_id = predicted_bucket.item()
            hash_index[bucket_id].append(rid)

    return hash_index, id_to_idx

if __name__ == '__main__':
    data_path = os.path.join('data', 'yelp_academic_dataset_review.json')
    model_path = os.path.join('models', 'saved_model.pth')
    hash_index, id_to_idx = build_hash_index(model_path, data_path)

    # Lưu hash_index ra file (tuỳ chọn)
    import pickle
    with open('models/hash_index.pkl', 'wb') as f:
        pickle.dump((hash_index, id_to_idx), f)

    print("Hash index built and saved.")
