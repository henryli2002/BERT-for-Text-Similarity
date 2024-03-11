from transformers import BertTokenizer, BertForSequenceClassification
import torch 
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path='bert-base-chinese',
    cache_dir=None,
    force_download=False,
)
column_names = ['text1', 'text2', 'labels']
df_train = pd.read_csv('Chinese-STS-B/sts-b-train.txt', sep='\t', names=column_names)
df_dev = pd.read_csv('Chinese-STS-B/sts-b-dev.txt', sep='\t', names=column_names)
df_test = pd.read_csv('Chinese-STS-B/sts-b-test.txt', sep='\t', names=column_names)

def encoder(df):
    train_data_encoded = tokenizer.batch_encode_plus(
    list(zip(df['text1'].values.tolist(), df['text2'].values.tolist())),
    add_special_tokens=True,
    truncation=True,
    padding='max_length', 
    max_length=30,
    return_tensors='pt'
)

    train_labels = df['labels'].values.tolist()
    return train_data_encoded, train_labels


train_data, train_labels = encoder(df_train)
input_ids = train_data['input_ids']
token_type_ids = train_data['token_type_ids']
attention_mask = train_data['attention_mask']
train_labels = torch.Tensor(train_labels)

# 创建数据集和数据加载器
dataset = TensorDataset(input_ids, token_type_ids, attention_mask, train_labels)
data_loader = DataLoader(dataset, batch_size=32)
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=6)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 3

model.to(device)
model.train()
for epoch in range(num_epochs):  # 如果需要多个epoch
    print(f"Epoch {epoch+1}/{num_epochs}")
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, batch in progress_bar:
        batch = [item.to(device) for item in batch]
        input_ids, token_type_ids, attention_mask, labels = batch

        model.zero_grad()  

        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels.to(torch.int64))
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        # 更新进度条
        progress_bar.set_description(f"Loss: {loss.item():.4f}")

        # 每隔100步打印loss
        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")

