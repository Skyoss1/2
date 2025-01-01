import json
import time
import pandas as pd
from memory_profiler import memory_usage
from B_Tree import BTree
from Hash_Index import HashIndex
from ElasticSearch_Index import ElasticSearchIndex

FILES = ['D:/project/Data/yelp_academic_dataset_review.json']

def load_data(filepaths, chunk_size=10000):
    for filepath in filepaths:
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
                if len(data) >= chunk_size:
                    yield pd.DataFrame(data)
                    data = []
        if data:
            yield pd.DataFrame(data)

def measure_time_and_memory(func, *args, **kwargs):
    start_time = time.time()
    mem_usage = memory_usage((func, args, kwargs))
    end_time = time.time()
    return (end_time - start_time), max(mem_usage)

def build_btree_index(data, key_field="review_id"):
    btree = BTree(t=3)
    records = data.to_dict('records') if isinstance(data, pd.DataFrame) else data
    for d in records:
        if key_field in d and d[key_field]:
            btree.insert(d[key_field], d)
    return btree

def build_hash_index(data, key_field="review_id"):
    h_index = HashIndex()
    records = data.to_dict('records')
    for d in records:
        if key_field in d and d[key_field]:
            h_index.insert(d[key_field], d)
    return h_index

def build_elasticsearch_index(data):
    es_index = ElasticSearchIndex()
    mapping = {
        "mappings": {
            "properties": {
                "review_id": {"type": "keyword"},
                "text": {"type": "text", "analyzer": "standard"},
                "date": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss"}
            }
        }
    }

    es_index.create_index(mapping=mapping)
    es_index.index_documents(data.to_dict('records'))
    return es_index

def calculate_accuracy_elasticsearch(results, query_text):
    if not results:
        return 0.0
    query_text_lower = query_text.lower()
    count_match = sum(1 for r in results if r.get("text", "").lower().count(query_text_lower) > 0)
    return count_match / len(results)

def calculate_exact_match_accuracy(results, expected_id):
    # Nếu không có kết quả hoặc bất kỳ kết quả nào sai id => accuracy = 0
    if not results:
        return 0.0
    for r in results:
        if r.get("review_id") != expected_id:
            return 0.0
    return 1.0

def main():
    # Để tổng hợp số liệu cho tất cả các chunk
    es_times = []
    es_mems = []
    es_accuracies = []
    es_counts = []

    btree_times = []
    btree_mems = []
    btree_accuracies = []
    btree_counts = []

    hash_times = []
    hash_mems = []
    hash_accuracies = []
    hash_counts = []

    query_text = "good"

    for chunk_index, chunk in enumerate(load_data(FILES, chunk_size=10000)):
        print(f"\nProcessing chunk {chunk_index + 1}...")
        if 'date' in chunk.columns:
            chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
            chunk['date'] = chunk['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

        if chunk.empty:
            print("No data in this chunk, skipping.")
            continue

        # Build indexes
        print("Building B-Tree index...")
        btree_index = build_btree_index(chunk)

        print("Building Hash index...")
        hash_index = build_hash_index(chunk)

        print("Building Elasticsearch index...")
        es_index = build_elasticsearch_index(chunk)

        # Chọn 1 review_id ví dụ để tìm exact match
        example_review_id = chunk.iloc[0]['review_id']

        # === Elasticsearch Search ===
        print("\n=== Full-text search with Elasticsearch ===")
        es_time, es_mem = measure_time_and_memory(es_index.search, query_text)
        es_results = es_index.search(query_text)
        es_count = len(es_results)
        es_accuracy = calculate_accuracy_elasticsearch(es_results, query_text)
        es_times.append(es_time)
        es_mems.append(es_mem)
        es_accuracies.append(es_accuracy)
        es_counts.append(es_count)

        print(f"Elasticsearch found {es_count} documents")
        print(f"Time: {es_time:.4f}s, Memory: {es_mem:.2f} MiB, Accuracy: {es_accuracy:.2f}")
        print("Sample Elasticsearch results:")
        for hit in es_results:
            print(hit)

        # === B-Tree Search ===
        print("\n=== Exact match search with B-Tree ===")
        def btree_search_func():
            return btree_index.search(example_review_id)
        btree_time, btree_mem = measure_time_and_memory(btree_search_func)
        btree_results = btree_index.search(example_review_id)  # Bây giờ là danh sách kết quả
        btree_count = len(btree_results)
        btree_accuracy = calculate_exact_match_accuracy(btree_results, example_review_id)
        btree_times.append(btree_time)
        btree_mems.append(btree_mem)
        btree_accuracies.append(btree_accuracy)
        btree_counts.append(btree_count)

        if btree_results:
            print(f"B-Tree Search found {btree_count} documents:")
            for doc in btree_results[:3]:
                print(doc)
        else:
            print("B-Tree Search Result: Not Found")
        print(f"B-Tree Time: {btree_time:.4f}s, Memory: {btree_mem:.2f} MiB, Accuracy: {btree_accuracy:.2f}")

        # === Hash Index Search ===
        print("\n=== Exact match search with Hash Index ===")
        def hash_search_func():
            return hash_index.search(example_review_id)
        hash_time, hash_mem = measure_time_and_memory(hash_search_func)
        hash_results = hash_index.search(example_review_id)  # Danh sách kết quả
        hash_count = len(hash_results)
        hash_accuracy = calculate_exact_match_accuracy(hash_results, example_review_id)
        hash_times.append(hash_time)
        hash_mems.append(hash_mem)
        hash_accuracies.append(hash_accuracy)
        hash_counts.append(hash_count)

        if hash_results:
            print(f"Hash Index Search found {hash_count} documents:")
            for doc in hash_results[:3]:
                print(doc)
        else:
            print("Hash Index Search Result: Not Found")
        print(f"Hash Time: {hash_time:.4f}s, Memory: {hash_mem:.2f} MiB, Accuracy: {hash_accuracy:.2f}")

        # Summary for this chunk
        print("\nSummary for this chunk:")
        print(f"- Elasticsearch: Count={es_count}, Time={es_time:.4f}s, Mem={es_mem:.2f}MiB, Acc={es_accuracy:.2f}")
        print(f"- B-Tree: Count={btree_count}, Time={btree_time:.4f}s, Mem={btree_mem:.2f}MiB, Acc={btree_accuracy:.2f}")
        print(f"- Hash: Count={hash_count}, Time={hash_time:.4f}s, Mem={hash_mem:.2f}MiB, Acc={hash_accuracy:.2f}")

    # After all chunks
    if es_times:
        avg_es_time = sum(es_times) / len(es_times)
        avg_es_mem = sum(es_mems) / len(es_mems)
        avg_es_acc = sum(es_accuracies) / len(es_accuracies)
        total_es_docs = sum(es_counts)

        avg_btree_time = sum(btree_times) / len(btree_times)
        avg_btree_mem = sum(btree_mems) / len(btree_mems)
        avg_btree_acc = sum(btree_accuracies) / len(btree_accuracies)
        total_btree_docs = sum(btree_counts)

        avg_hash_time = sum(hash_times) / len(hash_times)
        avg_hash_mem = sum(hash_mems) / len(hash_mems)
        avg_hash_acc = sum(hash_accuracies) / len(hash_accuracies)
        total_hash_docs = sum(hash_counts)

        print("\n=== Summary of All Chunks ===")
        print("Elasticsearch:")
        print(f"- Avg Time: {avg_es_time:.4f}s")
        print(f"- Avg Memory: {avg_es_mem:.2f} MiB")
        print(f"- Avg Accuracy: {avg_es_acc:.2f}")
        print(f"- Total Docs Found: {total_es_docs}")

        print("\nB-Tree:")
        print(f"- Avg Time: {avg_btree_time:.4f}s")
        print(f"- Avg Memory: {avg_btree_mem:.2f} MiB")
        print(f"- Avg Accuracy: {avg_btree_acc:.2f}")
        print(f"- Total Docs Found: {total_btree_docs}")

        print("\nHash Index:")
        print(f"- Avg Time: {avg_hash_time:.4f}s")
        print(f"- Avg Memory: {avg_hash_mem:.2f} MiB")
        print(f"- Avg Accuracy: {avg_hash_acc:.2f}")
        print(f"- Total Docs Found: {total_hash_docs}")
    else:
        print("\nNo data processed. No results available.")

if __name__ == "__main__":
    main()
