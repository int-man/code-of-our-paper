import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.optimization import AdamW
from datasets import load_dataset

class CustomDataset(Dataset):
    def __init__(self, txt_file, tokenizer):
        self.tokenizer = tokenizer
        self.lines = open(txt_file, 'r').readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        line = self.lines[index]
        inputs = self.tokenizer.encode_plus(line, add_special_tokens=True, padding='max_length', truncation=True, max_length=128)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        return {'input_ids': torch.tensor(input_ids), 'attention_mask': torch.tensor(attention_mask)}

def fine_tune_gpt2(txt_file, model_name, output_dir, num_epochs, batch_size, learning_rate):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    dataset = CustomDataset(txt_file, tokenizer)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GPT2LMHeadModel.from_pretrained(model_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for i, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs[0]
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Step: {i+1}/{len(train_loader)}, Loss: {loss.item()}')

        average_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1} average loss: {average_loss}')

    model.save_pretrained(output_dir)

def compute_perplexity(txt_file, model_name):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    dataset = CustomDataset(txt_file, tokenizer)
    eval_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = GPT2LMHeadModel.from_pretrained(model_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.eval()

    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs[0]
            total_loss += loss.item()
            total_tokens += input_ids.size(1)

    average_loss = total_loss / len(eval_loader)
    perplexity = 2 ** average_loss

    print(f'Perplexity: {perplexity}')


fine_tune_gpt2("input/pattern.txt", "model/GPT2/GPT2","model/GPT2/GPT2withNextWordpre",8,16,1e-3)
compute_perplexity("input/pattern.txt","model/GPT2/GPT2withNextWordpre")