import json
import os
from sklearn.model_selection import train_test_split

def load_review_ids(data_path, limit=None):
    review_ids = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            review_ids.append(data['review_id'])
            if limit is not None and i+1 >= limit:
                break
    return review_ids

def create_id_mapping(review_ids):
    unique_review_ids = list(set(review_ids))
    id_to_idx = {rid: i for i, rid in enumerate(unique_review_ids)}
    return unique_review_ids, id_to_idx

def split_data(indices, test_size=0.2, random_state=42):
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
    return train_idx, test_idx

def bucket_function(review_id_str, num_buckets=1024):
    # Sử dụng hàm băm Python làm target ban đầu
    return hash(review_id_str) % num_buckets
