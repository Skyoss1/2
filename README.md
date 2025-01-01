# Phân Tích và Tối Ưu Hóa Thuật Toán Tìm Kiếm Trên Cơ Sở Dữ Liệu Lớn

## Giới thiệu

Đề tài này tập trung nghiên cứu, phân tích và đánh giá các thuật toán tìm kiếm phổ biến trên cơ sở dữ liệu lớn, bao gồm **B-Tree**, **Hash-based Indexing**, và **Elasticsearch**. Đồng thời, đề tài đề xuất và triển khai giải pháp **Learned Hash Index** - một phương pháp lập chỉ mục mới sử dụng mô hình học máy (cụ thể là mạng nơ-ron MLP) để tối ưu hóa quá trình lập chỉ mục và truy vấn.

## Mục tiêu

*   Phân tích ưu, nhược điểm và đánh giá hiệu suất của các thuật toán tìm kiếm phổ biến (B-Tree, Hash-based Indexing, Elasticsearch) trên các tập dữ liệu lớn.
*   Đề xuất và triển khai giải pháp Learned Hash Index, kết hợp giữa mô hình học máy và cấu trúc dữ liệu bảng băm để tối ưu hóa việc lập chỉ mục và truy vấn.
*   So sánh hiệu suất của Learned Hash Index với các thuật toán truyền thống về các tiêu chí:
    *   Thời gian truy vấn
    *   Mức sử dụng bộ nhớ
    *   Độ chính xác
*   Đưa ra các đề xuất và hướng phát triển trong tương lai.

## Nội dung chính

Báo cáo bao gồm các nội dung chính sau:

*   **Chương 1: Giới thiệu đề tài** - Trình bày tổng quan về bối cảnh, lý do thực hiện, mục tiêu, phạm vi và phương pháp nghiên cứu.
*   **Chương 2: Cơ sở lý thuyết** - Giới thiệu về các thuật toán tìm kiếm, dữ liệu lớn (Big Data), và các phương pháp, kỹ thuật liên quan.
*   **Chương 3: Phân tích vấn đề** - Mô tả chi tiết bài toán tìm kiếm trên cơ sở dữ liệu lớn, các thách thức và yêu cầu đặt ra.
*   **Chương 4: Tối ưu hóa thuật toán** - Trình bày về lý thuyết tối ưu hóa, các kỹ thuật tối ưu hóa ứng dụng, và giải pháp đề xuất Learned Hash Index.
*   **Chương 5: Thực nghiệm** - Mô tả chi tiết các bước triển khai, cài đặt, cấu hình thí nghiệm, bao gồm:
    *   Mô tả bộ dữ liệu sử dụng (Yelp Academic Dataset - Review).
    *   Triển khai các thuật toán B-Tree, Hash-based Indexing, Elasticsearch.
    *   Triển khai mô hình Learned Hash Index (bao gồm huấn luyện và đánh giá mô hình).
*   **Chương 6: Kết quả và đánh giá** - Trình bày kết quả thực nghiệm, so sánh và đánh giá hiệu suất của các thuật toán, bao gồm:
    *   Kết quả thực nghiệm của B-Tree, Hash-based Indexing, Elasticsearch.
    *   Kết quả huấn luyện và đánh giá mô hình Learned Hash Index.
    *   So sánh hiệu suất giữa Learned Hash Index và Hash-based Indexing.
*   **Chương 7: Kết luận và hướng phát triển** - Tóm tắt các kết quả đạt được, ưu nhược điểm và đề xuất các hướng phát triển trong tương lai, bao gồm đề xuất ứng dụng thuật toán QAOA (Quantum Approximate Optimization Algorithm).

## Công nghệ sử dụng

*   **Ngôn ngữ lập trình:** Python 3.12
*   **Thư viện:**
    *   `pandas`, `NumPy`: Xử lý dữ liệu.
        *   [pandas](https://pandas.pydata.org/): `pip install pandas`
        *   [NumPy](https://numpy.org/): `pip install numpy`
    *   `scikit-learn`: Triển khai các mô hình học máy.
        *   [scikit-learn](https://scikit-learn.org/stable/): `pip install scikit-learn`
    *   `TensorFlow`/`PyTorch`: Triển khai mô hình học sâu cho Learned Hash Index.
        *   [TensorFlow](https://www.tensorflow.org/): `pip install tensorflow`
        *   [PyTorch](https://pytorch.org/): `pip install torch torchvision torchaudio`
    *   `elasticsearch`, `elasticsearch-py`: Kết nối và thao tác với Elasticsearch.
        *   [elasticsearch](https://elasticsearch-py.readthedocs.io/en/v7.17.0/): `pip install elasticsearch==7.17.0`
        *   [elasticsearch-py](https://elasticsearch-py.readthedocs.io/en/v7.17.0/): `pip install elasticsearch-py==7.17.0`
    *   `json`: Xử lý dữ liệu JSON.
        *   [json](https://docs.python.org/3/library/json.html) (Thư viện tiêu chuẩn của Python)
    *   `time`: Đo thời gian thực thi.
        *   [time](https://docs.python.org/3/library/time.html) (Thư viện tiêu chuẩn của Python)
    *   `memory_profiler`: Theo dõi mức sử dụng bộ nhớ.
        *   [memory_profiler](https://pypi.org/project/memory-profiler/): `pip install memory-profiler`
    *   Thư viện B-Tree và Hash Index tự định nghĩa
*   **Cơ sở dữ liệu:** Elasticsearch 7.17
*   **Môi trường triển khai:** Visual Studio Code, Windows, Python, Elasticsearch

## Bộ dữ liệu

**Yelp Academic Dataset - Review**

*   **Nguồn:** [https://www.yelp.com/dataset](https://www.yelp.com/dataset)
*   **Mô tả:** Tập dữ liệu chứa hơn 8 triệu đánh giá (review) của người dùng trên Yelp, bao gồm các thông tin: `review_id`, `user_id`, `business_id`, `stars`, `text`, `date`, `useful`, `funny`, `cool`.
*   **Lý do sử dụng:**
    *   Kích thước lớn, phù hợp để đánh giá hiệu suất và khả năng mở rộng.
    *   Bao gồm cả dữ liệu cấu trúc và phi cấu trúc.
    *   Phân bố dữ liệu phức tạp, phản ánh thách thức thực tế.
    *   Được sử dụng rộng rãi trong các nghiên cứu.

## Cài đặt

1. **Cài đặt Python 3.12:** Tải và cài đặt Python 3.12 từ trang chủ [https://www.python.org/downloads/](https://www.python.org/downloads/).
2. **Cài đặt các thư viện Python:** Mở terminal (hoặc Command Prompt) và chạy các lệnh sau:
    ```bash
    pip install pandas numpy scikit-learn tensorflow torch torchvision torchaudio elasticsearch==7.17.0 elasticsearch-py==7.17.0 memory-profiler
    ```
3. **Cài đặt Elasticsearch 7.17:**
    *   Tải Elasticsearch 7.17 từ trang chủ [https://www.elastic.co/downloads/past-releases/elasticsearch-7-17-0](https://www.elastic.co/downloads/past-releases/elasticsearch-7-17-0).
    *   Làm theo hướng dẫn cài đặt cho hệ điều hành của bạn.
    *   Sau khi cài đặt, khởi động Elasticsearch.
4. **Tải bộ dữ liệu Yelp Academic Dataset:**
    *   Truy cập [https://www.yelp.com/dataset](https://www.yelp.com/dataset) và tải xuống tập dữ liệu `yelp_academic_dataset_review.json`.
5. **Clone repository:** (Nếu bạn có repository chứa code)
    ```bash
    git clone https://github.com/Skyoss1/Bao-cao-thuc-tap-2024.git
    ```

## Hướng dẫn sử dụng

1. **Chuẩn bị dữ liệu:**
    ```bash
    python index.py --input_file yelp_academic_dataset_review.json --output_dir data
    ```
2. **Huấn luyện mô hình Learned Hash Index:**
    ```bash
    python train.py --data_dir data --model_path models/saved_model.pth
    ```
3. **Chạy thử nghiệm:**
    ```bash
    python main.py --data_dir data --model_path models/saved_model.pth
    ```

## Kết quả nổi bật

*   **Learned Hash Index** cho thấy sự cải thiện đáng kể về hiệu suất so với **Hash-based Indexing** truyền thống:
    *   **Thời gian truy vấn nhanh hơn khoảng 250 lần.**
    *   **Tiết kiệm bộ nhớ hơn khoảng 188 lần.**
    *   **Độ chính xác tương đương (100%).**

## Hướng phát triển

*   Thử nghiệm trên các tập dữ liệu lớn hơn và đa dạng hơn.
*   Tối ưu hóa mô hình học máy trong Learned Hash Index.
*   Giải quyết các thách thức về dữ liệu không đồng đều và cập nhật mô hình.
*   Kết hợp Learned Hash Index với các cấu trúc dữ liệu khác.
*   Mở rộng ứng dụng sang các lĩnh vực khác.
*   Nghiên cứu ứng dụng thuật toán QAOA để tối ưu hóa tài nguyên tìm kiếm.

## Tác giả

*   **Nguyễn Duy Nhật Huy** - MSSV: 2151150040
*   **Giảng viên hướng dẫn:** TS. Lê Quốc Tuấn

## Tài liệu tham khảo

Danh sách tài liệu tham khảo được liệt kê chi tiết trong file báo cáo (trang 87).
