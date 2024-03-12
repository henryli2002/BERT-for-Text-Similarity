from transformers import BertTokenizer, BertForSequenceClassification
import torch 
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from tqdm import tqdm

# 定义训练的参数
num_epochs = 10
batch_size = 64
learning_rate = 5e-5
max_length = 100
num_labels = 6

# 训练函数
def train(model, data_loader, optimizer, device, loss_fn=None):
    """训练模型的函数
    参数:
    model: 要训练的模型
    data_loader: 数据加载器
    optimizer: 优化器
    device: 设备（CPU或CUDA）
    loss_fn: 损失函数（可选，如果模型内部已定义，则不需要）
    """
    model.train()
    total_loss = 0.
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, batch in progress_bar:
        batch = [item.to(device) for item in batch]  # 将数据移动到指定设备
        input_ids, token_type_ids, attention_mask, labels = batch  # 解包数据

        optimizer.zero_grad()  # 清空梯度
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels.to(torch.int64))
        loss = outputs.loss  # 获取损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        total_loss += loss.item()
        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")

    avg_loss = total_loss / len(data_loader)
    print(f"Average loss: {avg_loss:.4f}")

# 测试函数
def test(model, data_loader, device, loss_fn=None):
    """评估模型的函数
    参数同上
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    results_list = []  # 用于收集预测结果
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
    with torch.no_grad():
        for step, batch in progress_bar:
            batch = [item.to(device) for item in batch]
            input_ids, token_type_ids, attention_mask, labels = batch

            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels.to(torch.int64))
            loss = outputs.loss  # 计算损失
            total_loss += loss.item()

            logits = outputs.logits  # 获取模型输出
            predictions = torch.argmax(logits, dim=-1)  # 获得预测结果
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            results_list.extend(predictions.cpu().numpy())  # 收集预测结果

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    return results_list  # 返回预测结果列表

# 主函数
def main():
    # 数据准备
    column_names = ['text1', 'text2', 'labels']
    df_train = pd.read_csv('Chinese-STS-B/sts-b-train.txt', sep='\t', names=column_names)
    df_dev = pd.read_csv('Chinese-STS-B/sts-b-dev.txt', sep='\t', names=column_names)
    df_test = pd.read_csv('Chinese-STS-B/sts-b-test.txt', sep='\t', names=column_names)

    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path='bert-base-chinese',
        cache_dir=None,
        force_download=False,
    )

    # 数据编码
    def encoder(df):
        """对文本数据进行编码的函数
        参数:
        df: 包含文本数据的DataFrame
        """
        train_data_encoded = tokenizer.batch_encode_plus(
            list(zip(df['text1'].values.tolist(), df['text2'].values.tolist())),
            add_special_tokens=True,
            truncation=True,
            padding='max_length', 
            max_length=max_length,
            return_tensors='pt'
        )
        train_labels = df['labels'].values.tolist()
        return train_data_encoded, train_labels

    # 模型、优化器和损失函数的准备
    train_data, train_labels = encoder(df_train)
    input_ids = train_data['input_ids']
    token_type_ids = train_data['token_type_ids']
    attention_mask = train_data['attention_mask']
    train_labels = torch.Tensor(train_labels)

    dataset = TensorDataset(input_ids, token_type_ids, attention_mask, train_labels)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=num_labels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 训练过程
    model.to(device)
    for epoch in range(num_epochs):  
        print(f"Epoch {epoch+1}/{num_epochs}")
        train(model=model, data_loader=data_loader, optimizer=optimizer, device=device)

    # 保存和加载模型，此处需要手动改模型路径
    torch.save(model.state_dict(), f'model_{epoch}e.pth')
    model.load_state_dict(torch.load('model_5e.pth')) 

    # 测试过程
    test_data, test_labels = encoder(df_test)
    test_dataset = TensorDataset(test_data['input_ids'], test_data['token_type_ids'], test_data['attention_mask'], torch.Tensor(test_labels))
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size)
    results_list = test(model=model, data_loader=test_data_loader, device=device)

    # 保存测试结果
    results_df = pd.DataFrame(results_list, columns=['predictions'])
    assert len(df_test) == len(results_df), "原始数据和测试结果数量不匹配！"
    final_df = pd.concat([df_test.reset_index(drop=True), results_df], axis=1)
    final_df.to_csv('./output/bert_output.txt', sep='\t', index=False, header=True, encoding='utf-8')


if __name__ == "__main__":
    main()
