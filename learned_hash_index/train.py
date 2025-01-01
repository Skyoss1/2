import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ReviewHashDataset
from model import MLPHash
from utils import load_review_ids, create_id_mapping, split_data, bucket_function

def main():
    data_path = os.path.join('data', 'yelp_academic_dataset_review.json')
    # Giới hạn 50.000 review để demo, bạn có thể bỏ đi giới hạn này
    review_ids = load_review_ids(data_path, limit=None)

    unique_review_ids, id_to_idx = create_id_mapping(review_ids)
    indices = [id_to_idx[rid] for rid in review_ids]

    # Tạo targets
    num_buckets = 1024
    targets = [bucket_function(rid, num_buckets=num_buckets) for rid in unique_review_ids]

    train_idx, test_idx = split_data(indices)

    # Tạo dataset và dataloader
    train_dataset = ReviewHashDataset(train_idx, unique_review_ids, targets)
    test_dataset = ReviewHashDataset(test_idx, unique_review_ids, targets)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPHash(output_dim=num_buckets).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    for epoch in range(num_epochs):
        # Huấn luyện
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x_batch.size(0)

        train_loss = running_loss / len(train_dataset)

        # Đánh giá
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        test_acc = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f} - Test Acc: {test_acc:.4f}")

    # Lưu mô hình
    os.makedirs('models', exist_ok=True)
    save_path = os.path.join('models', 'saved_model.pth')
    torch.save(model.state_dict(), save_path)
    print("Model saved at:", save_path)

if __name__ == '__main__':
    main()
