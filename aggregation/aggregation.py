import torch.utils.data as tud
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

torch.manual_seed(1)

SEMANTIC_EMBED_SIZE = 768
#NUM_EPOCHS = 60
NUM_EPOCHS = 60
LEARNING_RATE = 0.005
BATCH_SIZE = 256


class SimilarityModel(nn.Module):

    def __init__(self, semantic_size, series_size):
        super(SimilarityModel, self).__init__()
        self.semantic_size = semantic_size
        self.series_size = series_size

        self.semantic = nn.Linear(self.semantic_size, 50)

        self.semantic2 = nn.Linear(50, 30)

        self.series = nn.Linear(self.series_size, 50)

        self.series2 = nn.Linear(50, 30)

        self.reduce1 = nn.Linear(30, 20)

        self.merge = nn.Linear(20, 2)

        self.mseloss = nn.MSELoss()

        self.dropout = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

    def core(self, semantic, series):
        #print("semantic_size is %d"%(self.semantic_size))
        #print(self.semantic_size)
        f1 = torch.sigmoid(self.semantic(semantic))
        f1 = self.dropout(f1)
        f1 = F.relu(self.semantic2(f1))

        f2 = torch.sigmoid(self.series(series))
        f2 = self.dropout2(f2)
        f2 = F.relu(self.series2(f2))

        f3 = torch.tanh(torch.mul(f1, f2))

        f4 = torch.sigmoid(self.reduce1(f3))
        f4 = F.softmax(self.merge(f4), dim=1)
        return f4

    def forward(self, semantic, series, labels):
        f4 = self.core(semantic, series)
        #print(f4.shape,labels.shape)
        loss = self.mseloss(f4, labels)
        return loss

    def has_relation(self, semantic, series):
        #print(semantic.shape,series.shape)
        semantic = semantic.view(1,768)
        f4 = self.core(semantic, series)[0]
        #print(self.core(semantic, series))
        if f4[0].item() > f4[1].item():
            return f4[0].item()
        return -1


class SimilarityDataset(tud.Dataset):
    def __init__(self, right_samples, negative_samples=None):
        super(SimilarityDataset, self).__init__()
        self.input_alarm_series = []
        self.input_alarm_semantic = []
        self.output_vectors = []
        for i, (tar_alarm_series, tar_alarm_semantic, other_alarm_series, other_alarm_semantic) in enumerate(
                right_samples):
            alarm_series = (tar_alarm_series - other_alarm_series) ** 2
            alarm_semantic = (tar_alarm_semantic - other_alarm_semantic) ** 2
            output_vector = np.array([1, 0])

            self.input_alarm_series.append(alarm_series.astype(np.float32))
            self.input_alarm_semantic.append(alarm_semantic.astype(np.float32))
            self.output_vectors.append(output_vector.astype(np.float32))

        if negative_samples is not None:
            for i, (tar_alarm_series, tar_alarm_semantic, other_alarm_series, other_alarm_semantic) in enumerate(
                    negative_samples):
                alarm_series = (tar_alarm_series - other_alarm_series) ** 2
                alarm_semantic = (tar_alarm_semantic - other_alarm_semantic) ** 2
                output_vector = np.array([0, 1])

                self.input_alarm_series.append(alarm_series.astype(np.float32))
                self.input_alarm_semantic.append(alarm_semantic.astype(np.float32))
                self.output_vectors.append(output_vector.astype(np.float32))

    def __len__(self):
        return len(self.input_alarm_series)

    def __getitem__(self, idx):
        alarm_series = torch.from_numpy(self.input_alarm_series[idx])
        alarm_semantic = torch.from_numpy(self.input_alarm_semantic[idx]).view(768)
        alarm_label = torch.from_numpy(self.output_vectors[idx])
        return alarm_series, alarm_semantic, alarm_label


def get_dataloader(right_samples_path, negative_samples_path=None):
    right_samples = np.load(right_samples_path, allow_pickle=True)
    #print(right_samples.shape)#(838,4)
    if negative_samples_path:
        negative_samples = np.load(negative_samples_path, allow_pickle=True)
    else:
        negative_samples = None
    full_dataset = SimilarityDataset(right_samples, negative_samples)

    #print(right_samples[0][0].shape)
    #print(right_samples[0][1].shape)
    # series_len = right_samples[0][0].shape[0]
    #print(right_samples.shape)#838,4

    series_len = len(right_samples[0][0])
    #sermantic_len = len(right_samples[0][1])
    # print(right_samples[0][1])
    sermantic_len = right_samples[0][1].shape[1]

    train_dataloader = tud.DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_dataloader = None

    return train_dataloader, test_dataloader, sermantic_len, series_len


def train(input_data_path, output_data_path, model_path, model_name):
    train_dataloader, _, sermantic_len, series_len = get_dataloader(input_data_path, output_data_path)
    model = SimilarityModel(sermantic_len, series_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_losses = []

    for e in range(NUM_EPOCHS):
        for i, (alarm_series, alarm_semantic, alarm_label) in enumerate(train_dataloader):

            optimizer.zero_grad()
            #print(alarm_semantic.shape)
            #print(alarm_semantic.shape, alarm_series.shape, alarm_label.shape)#torch.Size([256, 1, 768]) torch.Size([256, 600]) torch.Size([256, 2])
            loss = model(alarm_semantic, alarm_series, alarm_label)
            #print(model.core(alarm_semantic,alarm_series))
            loss.backward()
            optimizer.step()

            if not train_losses or loss < min(train_losses):
                train_losses.append(loss)
                torch.save(model.state_dict(), model_path)
            #print('epoch', e, 'iteration', i, 'loss', loss.item())


def evaluate(input_data_path, output_data_path, model_path):
    _, test_dataloader, sermantic_len, series_len = get_dataloader(input_data_path, output_data_path)
    best_model = SimilarityModel(sermantic_len, series_len)
    best_model.load_state_dict(torch.load(model_path))

    best_model.eval()
    right_cnt = 0
    total_cnt = 0
    with torch.no_grad():
        for i, (alarm_series, alarm_semantic, alarm_label) in enumerate(test_dataloader):
            result = best_model.core(alarm_semantic, alarm_series)
            for j in range(result.shape[0]):
                has_relation = result[j][0] > result[j][1]
                if alarm_label[j][0] > alarm_label[j][1] and has_relation:
                    right_cnt += 1
                if alarm_label[j][0] < alarm_label[j][1] and not has_relation:
                    right_cnt += 1
                total_cnt += 1
    print('total cnt', total_cnt)
    print('right cnt', right_cnt)
    print('precistion', round(right_cnt / total_cnt * 100, 2))
