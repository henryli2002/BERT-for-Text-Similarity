from transformers import BertTokenizer, BertForSequenceClassification
import torch 
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from tqdm import tqdm

num_epochs = 10
batch_size = 64
learning_rate = 5e-5
max_length = 100
num_labels = 6

def train(model, data_loader, optimizer, device, loss_fn=None):
    model.train()
    total_loss = 0.
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, batch in progress_bar:
        batch = [item.to(device) for item in batch]
        input_ids, token_type_ids, attention_mask, labels = batch

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels.to(torch.int64))
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")

    avg_loss = total_loss / len(data_loader)
    print(f"Average loss: {avg_loss:.4f}")



def test(model, data_loader, device, loss_fn=None):
    model.eval()  
    total_loss = 0
    total_correct = 0
    total_samples = 0
    results_list = [] 
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
    with torch.no_grad(): 
        for step, batch in progress_bar:
            batch = [item.to(device) for item in batch]
            input_ids, token_type_ids, attention_mask, labels = batch

            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels.to(torch.int64))
            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            results_list.extend(predictions.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    return results_list


column_names = ['text1', 'text2', 'labels']
df_train = pd.read_csv('Chinese-STS-B/sts-b-train.txt', sep='\t', names=column_names)
df_dev = pd.read_csv('Chinese-STS-B/sts-b-dev.txt', sep='\t', names=column_names)
df_test = pd.read_csv('Chinese-STS-B/sts-b-test.txt', sep='\t', names=column_names)

tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path='bert-base-chinese',
    cache_dir=None,
    force_download=False,
)

def encoder(df):
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


def main():
    train_data, train_labels = encoder(df_train)
    input_ids = train_data['input_ids']
    token_type_ids = train_data['token_type_ids']
    attention_mask = train_data['attention_mask']
    train_labels = torch.Tensor(train_labels)

    dataset = TensorDataset(input_ids, token_type_ids, attention_mask, train_labels)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=num_labels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model.to(device)
    # model.train()
    # for epoch in range(num_epochs):  
    #     print(f"Epoch {epoch+1}/{num_epochs}")
    #     train(model=model, data_loader=data_loader, optimizer=optimizer, device=device, loss_fn=None)
    # torch.save(model.state_dict(), f'model_{epoch}e.pth')

    model.load_state_dict(torch.load('model_5e.pth'))
    test_data, test_labels = encoder(df_test)
    test_input_ids = test_data['input_ids']
    test_token_type_ids = test_data['token_type_ids']
    test_attention_mask = test_data['attention_mask']
    test_labels = torch.Tensor(test_labels)

    dataset = TensorDataset(test_input_ids, test_token_type_ids, test_attention_mask, test_labels)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    results_list = test(model=model, data_loader=data_loader, device=device, loss_fn=None)
    results_df = pd.DataFrame(results_list, columns=['predictions'])
    print(df_test.values,len(df_test))
    print(results_df.values, len(results_df))
    assert len(df_test) == len(results_df), "原始数据和测试结果数量不匹配！"
    
    final_df = pd.concat([df_test, results_df], axis=1)
    final_df.to_csv('./output/bert_output.txt', sep='\t', index=False, header=True, encoding='utf-8')

if __name__ == "__main__":
    main()