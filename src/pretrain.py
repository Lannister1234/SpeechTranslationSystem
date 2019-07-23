"""
Refer to handout for details.
- Build scripts to train your model
- Submit your code to Autolab
"""
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import Levenshtein as Lev

char_list = ['<eos>', ' ', "'", '+', '-', '.', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_']

index_list = {'<eos>': 0, ' ': 1, "'": 2, '+': 3, '-': 4, '.': 5, 'A': 6, 'B': 7, 'C': 8, 'D': 9, 'E': 10, 'F': 11, 'G': 12, 'H': 13, 'I': 14, 'J': 15, 'K': 16, 'L': 17, 'M': 18, 'N': 19, 'O': 20, 'P': 21, 'Q': 22, 'R': 23, 'S': 24, 'T': 25, 'U': 26, 'V': 27, 'W': 28, 'X': 29, 'Y': 30, 'Z': 31, '_': 32}

num_chars = len(char_list)

BATCH_SIZE = 32
LR = 1e-4
EPOCH_NUM = 20
CUDA = True
WEIGHT_DECAY = 0.0001

# model parameters
LISTENER_HIDDEN_SIZE = 256
SPELLER_HIDDEN_SIZE = 256
LISTENER_LAYERS = 3
SPELLER_LAYERS = 3
DROPOUT = 0.3
DROPOUTH = 0.1
DROPOUTI = 0.2

KEY_SIZE = 128
VALUE_SIZE = 128
CONTEXT_SIZE = 128
QUERY_SIZE = 128
TEACHER_FORCE_RATE = 0.9
MAX_LEN = 220


def find_first_eos(pred):
    # pred: L, C
    chars = torch.max(pred, 1)[1]
    length = len(chars)
    for idx in range(length):
        if chars[idx] == index_list['<eos>']:
            return idx
    return length


# convert labels to strings
def get_label_str(labels, label_sizes):
    string_list = []
    for i in range(len(labels)):
        size = label_sizes[i]
        label = labels[i][:size]
        string_list.append("".join(char_list[idx] for idx in label))
    return string_list


# greedy decode
def decode(outputs):
    str_list = []
    for output in outputs:
        string = ""
        for res in output:
            max_idx = torch.max(res, dim=0)[1]
            string += char_list[max_idx]
            if max_idx == 0:  # end of sentence
                break
        str_list.append(string)
    return str_list


class SeqCEL(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets, label_sizes):
        loss = 0
        for pred, target, label_size in zip(preds, targets, label_sizes):
            target = target[:label_size]
            pred = pred[:label_size]
            loss += F.cross_entropy(pred, target, reduction="sum")
        return loss


class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data[:1000]
        self.labels = None
        self.num_phonemes = 0
        if labels is not None:
            self.labels = labels[:1000]
            for label in self.labels:
                for word in label:
                    self.num_phonemes += len(word)
                    self.num_phonemes += 1

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        frames = self.data[idx]
        label = []
        if self.labels is not None:
            for word in self.labels[idx]:
                word = word.decode("utf-8")
                label.extend([index_list[c] for c in word])
                label.append(index_list[" "])
            label[-1] = 0
            # append eos to the end
            label = np.array(label)
            return torch.from_numpy(frames).float(), torch.from_numpy(label).long()
        else:
            return torch.from_numpy(frames).float(), None


def collate(seq_list):
    # batch_size
    batch_size = len(seq_list)
    # sort this batch by seq_len, desc
    seq_list.sort(key=lambda x: x[0].shape[0], reverse=True)

    # get max length and frequency
    max_len = seq_list[0][0].size(0)
    freq = seq_list[0][0].size(1)
    # get max length of label
    max_len_label = 0
    for (data, label) in seq_list:
        max_len_label = max(max_len_label, label.size(0))

    # initialize all
    pad_batch_data = torch.zeros((max_len, batch_size, freq))
    seq_len_list = torch.zeros(batch_size, dtype=torch.int)
    pad_batch_labels = torch.zeros((batch_size, max_len_label), dtype=torch.long)
    label_len_list = torch.zeros(batch_size, dtype=torch.int)

    # for train and dev data
    for i in range(batch_size):
        seq_len_list[i] = len(seq_list[i][0])
        label_len_list[i] = len(seq_list[i][1])
        pad_batch_data[:seq_len_list[i], i, :] = seq_list[i][0]
        pad_batch_labels[i, :label_len_list[i]] = seq_list[i][1]

    return pad_batch_data, pad_batch_labels, seq_len_list, label_len_list


# implementation for speller model
class SpellerModel(nn.Module):
    def __init__(self):
        super(SpellerModel, self).__init__()
        self.num_layers = SPELLER_LAYERS
        self.output_size = num_chars
        self.hidden_size = SPELLER_HIDDEN_SIZE
        self.context_size = CONTEXT_SIZE
        # embed layer (output_size, hidden_size)
        self.embed = nn.Embedding(self.output_size, SPELLER_HIDDEN_SIZE)

        # 3 layers, 1st (hidden + context, hidden), 2nd & 3rd (hidden, hidden)
        self.rnns = nn.ModuleList()
        self.rnns.append(nn.LSTMCell(SPELLER_HIDDEN_SIZE + CONTEXT_SIZE, SPELLER_HIDDEN_SIZE))
        for i in range(self.num_layers - 1):
            self.rnns.append(nn.LSTMCell(SPELLER_HIDDEN_SIZE, SPELLER_HIDDEN_SIZE))

        # output layer
        self.scoring = nn.Sequential(
            nn.Linear(SPELLER_HIDDEN_SIZE + CONTEXT_SIZE, SPELLER_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(SPELLER_HIDDEN_SIZE, self.output_size),
        )

        self.query_layer = nn.Linear(SPELLER_HIDDEN_SIZE, QUERY_SIZE)
        self.h_0 = torch.nn.ParameterList()
        self.c_0 = torch.nn.ParameterList()
        self.init_state()

    def forward(self, labels):
        batch_size = len(labels)
        # get max iteration number
        if labels is not None:
            max_steps = labels.shape[1]
        else:
            max_steps = MAX_LEN

        # sos
        char = torch.zeros(batch_size, dtype=torch.long)

        # get initial state
        hidden = [h.repeat(batch_size, 1) for h in self.h_0]
        cell = [c.repeat(batch_size, 1) for c in self.c_0]
        context = torch.zeros((batch_size, CONTEXT_SIZE))

        score_list, predict_list = [], []
        for i in range(max_steps):
            if CUDA:
                char = char.cuda()
            # run one step
            predict, hidden, cell = self.forward_one_step(char, hidden, cell, context)
            predict_list.append(predict)
            teacher_force = labels is not None and self.training and np.random.uniform() < TEACHER_FORCE_RATE
            # use given transcripts if teacher force
            if teacher_force:
                char = labels[:, i]
            else:
                char = torch.max(predict, dim=1)[1]
        predict_list = torch.stack(predict_list, dim=1)
        return score_list, predict_list

    def forward_one_step(self, char, hidden, cell, context):
        # embedding of last predict result
        char_embed = self.embed(char)
        # print(char_embed.size())
        if CUDA:
            context = context.cuda()
        # cat embedding and context together
        rnn_input = torch.cat([char_embed, context], 1)

        # rnns
        i = 0
        for rnn in self.rnns:
            hidden[i], cell[i] = rnn(rnn_input, (hidden[i], cell[i]))
            rnn_input = hidden[i]
            i += 1

        # cat features
        cat_features = torch.cat([hidden[-1], context], 1)
        # get predict result
        predict = self.scoring(cat_features)
        return predict, hidden, cell

    def init_state(self):
        for i in range(self.num_layers):
            self.h_0.append(torch.nn.Parameter(torch.zeros((1, self.hidden_size))))
            self.c_0.append(torch.nn.Parameter(torch.zeros((1, self.hidden_size))))


# implementation for LAS model
class LASModel(nn.Module):
    def __init__(self):
        super(LASModel, self).__init__()
        self.speller = SpellerModel()
        self.max_len = MAX_LEN

    def forward(self, frames, seq_sizes, labels):
        # speller
        score_list, predict_list = self.speller(labels)

        return score_list, predict_list


class TranslateModelTrainer:
    def __init__(self, model, loaders, max_epochs=EPOCH_NUM, weight_decay=WEIGHT_DECAY):
        self.model = model
        self.train_loader = loaders[0]
        self.dev_loader = loaders[1]
        self.max_epochs = max_epochs
        self.lr = LR
        self.num_phonemes_train = self.train_loader.dataset.num_phonemes
        self.num_phonemes_dev = self.dev_loader.dataset.num_phonemes
        self.cuda = True
        self.epochs = 0
        # optimizer and criterion
        self.optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
        self.criterion = SeqCEL()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=2,
                                                                    threshold=0.01, verbose=True)

    def train(self):
        self.model.train()
        torch.set_grad_enabled(True)
        loss_sum = 0
        print("----training----")
        log = open(os.path.join('log2.txt'), 'a')
        for batch_num, (inputs, targets, seq_len_list, label_len_list) in enumerate(self.train_loader):
            print(batch_num)
            loss = self.train_batch(inputs, targets, seq_len_list, label_len_list)
            loss_sum += loss
        loss_sum = loss_sum / self.num_phonemes_train
        self.epochs += 1
        self.scheduler.step(loss_sum)
        print('[TRAIN]    Loss: %.4f ' % loss_sum)
        log.write('[TRAIN]   Loss: %.4f\n' % loss_sum)
        log.close()

    def train_batch(self, inputs, targets, seq_len_list, label_len_list):
        if self.cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        self.optimizer.zero_grad()
        # generate output
        score, predict = self.model(inputs, seq_len_list, targets)

        # get loss
        loss = self.criterion(predict, targets, label_len_list)

        loss.backward()

        self.optimizer.step()

        if self.cuda:
            loss = loss.cpu().detach().numpy()
        return loss

    def validate(self):
        self.model.eval()
        torch.set_grad_enabled(False)
        loss_sum, dis_sum = 0, 0
        print("----validate----")
        log = open(os.path.join('log2.txt'), 'a')
        for batch_num, (inputs, targets, seq_len_list, label_len_list) in enumerate(self.dev_loader):
            print(batch_num)
            loss, error = self.validate_batch(inputs, targets, seq_len_list, label_len_list)
            loss_sum += loss
            dis_sum += error
        loss_sum = loss_sum / self.num_phonemes_dev
        dis_sum = dis_sum / self.num_phonemes_dev
        print('[VAL]    Loss: %.4f  Dis: %.4f' % (loss_sum, dis_sum))
        log.write('[VAL]   Loss: %.4f  Dis: %.4f\n' % (loss_sum, dis_sum))
        log.close()
        return dis_sum

    def validate_batch(self, inputs, targets, seq_len_list, label_len_list):
        err = 0
        if self.cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        self.optimizer.zero_grad()

        score, predict = self.model(inputs, seq_len_list, targets)

        # get loss
        loss = self.criterion(predict, targets, label_len_list)

        # get decode output string
        decoded_str = decode(predict)
        print(decoded_str)

        # get decode label string
        label_str = get_label_str(targets, label_len_list)
        # print(label_str)
        # distance between output str and label str
        for i in range(len(label_str)):
            err += Lev.distance(decoded_str[i], label_str[i])

        if self.cuda:
            loss = loss.cpu().detach().numpy()
        return loss, err

def main():
    # loading data
    path = "/home/ubuntu/hw4/data/"
    train_data = np.load(path + 'train.npy', encoding='latin1')
    train_labels = np.load(path + 'train_transcripts.npy', encoding='latin1')
    dev_data = np.load(path + 'dev.npy', encoding='latin1')
    dev_labels = np.load(path + 'dev_transcripts.npy', encoding='latin1')
    test_data = np.load(path + 'test.npy', encoding='latin1')

    print("loading data complete")
    print(train_data.shape)
    print(train_labels.shape)
    print(dev_data.shape)
    print(dev_labels.shape)
    print(test_data.shape)
    # model training parameters
    seed = 1000

    # set seed
    torch.manual_seed(seed)
    if CUDA:
        torch.cuda.manual_seed(seed)

    loaders = []
    d1 = MyDataset(train_data, train_labels)
    train_loader = DataLoader(d1, num_workers=8, batch_size=BATCH_SIZE, shuffle=True,
                                                    pin_memory=True, collate_fn=collate)

    d2 = MyDataset(dev_data, dev_labels)
    dev_loader = DataLoader(d2, num_workers=8, batch_size=BATCH_SIZE, shuffle=False,
                                                    pin_memory=True, collate_fn=collate)

    # d3 = MyDataset(test_data, None)
    # test_loader = DataLoader(d3, num_workers=8, batch_size=1, shuffle=False,
    #                                                 pin_memory=True, collate_fn=collate)

    loaders.append(train_loader)
    loaders.append(dev_loader)

    model = LASModel()
    if CUDA:
        model.cuda()

    # model.load_state_dict(torch.load("/home/ubuntu/hw3p2/model/model-19-10.5392.txt"))
    trainer = TranslateModelTrainer(model, loaders, max_epochs=EPOCH_NUM)

    for epoch in range(EPOCH_NUM):
        print("starting epoch:")
        print(epoch)
        trainer.train()
        err = trainer.validate()
        model_path = os.path.join('experiments', 'model-{:.4f}.txt'.format(err))
        torch.save(model.state_dict(), model_path)
        print(str(epoch) + " with error: " + str(err))
    # test(model, test_loader, cuda, save_path)


if __name__ == '__main__':
    main()