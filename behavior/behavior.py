import torch.utils.data as tud
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import torch
from config import CONFIG

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

NUM_EPOCHS = 5
LEARNING_RATE = 0.002
EMBEDDING_SIZE = 600
EMBEDDING_SIZE2 = 500
INPUT_VECTOR_SIZE = int(CONFIG.SERIES_WINDOW_LENGTH / CONFIG.SERIES_WINDOW_GRANULARITY)
OUTPUT_VECTOR_SIZE = int(CONFIG.SERIES_WINDOW_LENGTH / CONFIG.SERIES_WINDOW_GRANULARITY)
BATCH_SIZE = 64


class EmbeddingDataset(tud.Dataset):
    def __init__(self, input_data, output_data_pos, output_data_neg):
        super(EmbeddingDataset, self).__init__()
        self.input_time_series = []
        self.output_vectors_pos = []
        self.output_vectors_neg = []
        for i, x in enumerate(input_data):
            pos_y = output_data_pos[i]
            if output_data_neg:
                neg_y = output_data_neg[i]
            else:
                neg_y = None
            self.input_time_series.append(x.astype(np.float32))
            self.output_vectors_pos.append(pos_y.astype(np.float32))
            if neg_y is not None:
                self.output_vectors_neg.append(neg_y.astype(np.float32))
            else:
                self.output_vectors_neg.append(np.zeros(len(x), dtype=np.float32))

    def __len__(self):
        return len(self.input_time_series)

    def __getitem__(self, idx):
        input_vec = torch.from_numpy(self.input_time_series[idx])
        output_pos_vec = torch.from_numpy(self.output_vectors_pos[idx])
        output_neg_vec = torch.from_numpy(self.output_vectors_neg[idx])
        return input_vec, output_pos_vec, output_neg_vec


class EmbeddingModel(nn.Module):

    def __init__(self, input_embed_size, output_embed_size):
        global INPUT_VECTOR_SIZE, OUTPUT_VECTOR_SIZE
        INPUT_VECTOR_SIZE = int(CONFIG.SERIES_WINDOW_LENGTH / CONFIG.SERIES_WINDOW_GRANULARITY)
        OUTPUT_VECTOR_SIZE = int(CONFIG.SERIES_WINDOW_LENGTH / CONFIG.SERIES_WINDOW_GRANULARITY)

        super(EmbeddingModel, self).__init__()
        self.empty = torch.from_numpy(np.zeros(INPUT_VECTOR_SIZE, dtype=np.float32))
        self.input_length = INPUT_VECTOR_SIZE
        self.input_embed_size = input_embed_size
        self.output_embed_size = output_embed_size
        self.output_length = OUTPUT_VECTOR_SIZE

        self.in_embed = nn.Linear(self.input_length, self.input_embed_size)

        self.out_embed1 = nn.Linear(self.output_length, self.output_embed_size)
        self.relu = nn.ReLU()
        self.out_embed2 = nn.Linear(self.output_embed_size, self.input_embed_size)

    def forward(self, input_series, output_pos_vec, output_neg_vec):
        input_embedding = self.in_embed(input_series)

        pos_embedding1 = self.out_embed1(output_pos_vec)
        pos_embedding2 = self.relu(pos_embedding1)
        pos_embedding3 = self.out_embed2(pos_embedding2)
        pos_loss = torch.bmm(pos_embedding3.unsqueeze(1), input_embedding.unsqueeze(2)).squeeze()  # [batch_size]

        neg_loss = 0
        if not torch.equal(output_neg_vec, self.empty):
            neg_embedding1 = self.out_embed1(output_neg_vec)
            neg_embedding2 = self.relu(neg_embedding1)
            neg_embedding3 = self.out_embed2(neg_embedding2)
            neg_loss = torch.bmm(neg_embedding3.unsqueeze(1), input_embedding.unsqueeze(2)).squeeze()  # [batch_size]

        loss = F.logsigmoid(pos_loss - neg_loss)

        return -loss

    def embedding(self, raw_series):
        return self.in_embed(raw_series).data.numpy()


def get_dataloader(input_data_path, output_pos_path, output_neg_path=None):
    input_data = np.load(input_data_path, allow_pickle=True)
    output_pos = np.load(output_pos_path, allow_pickle=True)
    if output_neg_path:
        output_neg = np.load(output_neg_path, allow_pickle=True)
    else:
        output_neg = None
    full_dataset = EmbeddingDataset(input_data, output_pos, output_neg)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_dataloader = tud.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_dataloader = tud.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    return train_dataloader, test_dataloader


def train(model_path, input_data_path, output_pos_path, output_neg_path=None):
    global INPUT_VECTOR_SIZE, OUTPUT_VECTOR_SIZE
    INPUT_VECTOR_SIZE = int(CONFIG.SERIES_WINDOW_LENGTH / CONFIG.SERIES_WINDOW_GRANULARITY)
    OUTPUT_VECTOR_SIZE = int(CONFIG.SERIES_WINDOW_LENGTH / CONFIG.SERIES_WINDOW_GRANULARITY)

    train_dataloader, _ = get_dataloader(input_data_path, output_pos_path, output_neg_path)
    print('Start training behavior embedding model.')
    model = EmbeddingModel(EMBEDDING_SIZE, EMBEDDING_SIZE2)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

    train_losses = []

    for e in range(NUM_EPOCHS):
        for i, (input_series, output_pos_vec, output_neg_vec) in enumerate(train_dataloader):

            optimizer.zero_grad()
            loss = model(input_series, output_pos_vec, output_neg_vec).mean()
            loss.backward()
            optimizer.step()

            if not train_losses or loss < min(train_losses):
                train_losses.append(loss)
                torch.save(model.state_dict(), model_path)
            else:
                if scheduler:
                    scheduler.step()
            print('epoch', e, 'iteration', i, 'loss', loss.item())


def evaluate(input_data_path, output_data_path, model_path):
    global INPUT_VECTOR_SIZE, OUTPUT_VECTOR_SIZE
    INPUT_VECTOR_SIZE = int(CONFIG.SERIES_WINDOW_LENGTH / CONFIG.SERIES_WINDOW_GRANULARITY)
    OUTPUT_VECTOR_SIZE = int(CONFIG.SERIES_WINDOW_LENGTH / CONFIG.SERIES_WINDOW_GRANULARITY)

    _, test_dataloader = get_dataloader(input_data_path, output_data_path)
    best_model = EmbeddingModel(EMBEDDING_SIZE, EMBEDDING_SIZE2)
    best_model.load_state_dict(torch.load(model_path))

    best_model.eval()
    total_loss = 0.
    total_cnt = 0.
    with torch.no_grad():
        for i, (log_key_series, log_key_vectors) in enumerate(test_dataloader):
            loss = best_model(log_key_series, log_key_vectors).mean()

            total_loss += loss.item()
            total_cnt += log_key_vectors.size()[0]
    best_model.train()
    loss = total_loss * 1.0 / total_cnt
    print('test loss', loss)
    return loss
