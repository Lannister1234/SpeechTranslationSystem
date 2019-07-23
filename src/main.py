"""
Refer to handout for details.
- Build scripts to train your model
- Submit your code to Autolab
"""
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import Levenshtein as Lev

char_list = ['<eos>', ' ', "'", '+', '-', '.', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_']
index_list = {'<eos>': 0, ' ': 1, "'": 2, '+': 3, '-': 4, '.': 5, 'A': 6, 'B': 7, 'C': 8, 'D': 9, 'E': 10, 'F': 11, 'G': 12, 'H': 13, 'I': 14, 'J': 15, 'K': 16, 'L': 17, 'M': 18, 'N': 19, 'O': 20, 'P': 21, 'Q': 22, 'R': 23, 'S': 24, 'T': 25, 'U': 26, 'V': 27, 'W': 28, 'X': 29, 'Y': 30, 'Z': 31, '_': 32}

num_chars = len(char_list)

BATCH_SIZE = 5
LR = 1e-4
EPOCH_NUM = 2
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
            if max_idx == index_list['<eos>']:  # end of sentence
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
        self.data = data[:10]
        self.labels = None
        self.num_phonemes = 0
        if labels is not None:
            self.labels = labels[:10]
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
            label[-1] = index_list["<eos>"]
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

    # for test data which has no label
    if seq_list[0][1] is None:
        for i in range(batch_size):
            seq_len_list[i] = len(seq_list[i][0])
            pad_batch_data[:seq_len_list[i], i, :] = seq_list[i][0]
        return pad_batch_data, None, seq_len_list, None

    # for train and dev data
    for i in range(batch_size):
        seq_len_list[i] = len(seq_list[i][0])
        label_len_list[i] = len(seq_list[i][1])
        pad_batch_data[:seq_len_list[i], i, :] = seq_list[i][0]
        pad_batch_labels[i, :label_len_list[i]] = seq_list[i][1]

    return pad_batch_data, pad_batch_labels, seq_len_list, label_len_list


class LockDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, rate=0.5):
        if rate == 0 or not self.training:
            return x
        mask = x.data.new(1, x.size(1), x.size(2))
        mask = mask.bernoulli_(1 - rate)
        mask.requires_grad = False
        mask = mask / (1 - rate)
        mask = mask.expand_as(x)
        x = mask * x
        return x


# model for pyramid LSTM layer
class myLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(myLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True)

    def forward(self, x, seq_sizes):
        # get dim size
        seq_len, batch_size, feature_len = x.size(0), x.size(1), x.size(2)
        new_len = seq_len // 2
        x = x[:2 * new_len, :, :]
        x = x.permute(1, 0, 2)
        x = x.contiguous().view(batch_size, new_len, feature_len * 2).permute(1, 0, 2)
        # [32, 578, 512]
        x = pack_padded_sequence(x, seq_sizes)
        output, _ = self.rnn(x)
        output, _ = pad_packed_sequence(output)
        # reduce time dimension by dividing 2

        return output


# implementation for listener model
class ListenerModel(nn.Module):
    def __init__(self, input_size=40):
        super(ListenerModel, self).__init__()
        self.nlayers = LISTENER_LAYERS
        self.input_size = input_size
        self.hidden_size = LISTENER_HIDDEN_SIZE

        # embedding layer
        self.embedding = nn.Sequential(nn.BatchNorm1d(input_size),)

        # 3 layer Bidirectional LSTM
        self.rnns = nn.ModuleList()
        self.rnns.append(myLSTM(self.input_size * 2, self.hidden_size))
        for i in range(self.nlayers - 1):
            self.rnns.append(myLSTM(self.hidden_size * 4, self.hidden_size))

        self.dropout = LockDropout()
        self.key_layer = nn.Linear(self.hidden_size * 2, KEY_SIZE)
        self.val_layer = nn.Linear(self.hidden_size * 2, VALUE_SIZE)
        self.activate = torch.nn.ReLU()

    def forward(self, data, seq_sizes):
        data = data.permute(1, 2, 0)
        embed = self.embedding(data)
        output = embed.permute(2, 0, 1)
        i = 0
        for rnn in self.rnns:
            seq_sizes = torch.IntTensor([size // 2 for size in seq_sizes])
            output = rnn(output, seq_sizes)
            if i != self.nlayers - 1:
                output = self.dropout(output, DROPOUTH)
            i += 1

        output = self.dropout(output, DROPOUT)
        # 0. generate values
        value = self.val_layer(output)
        # 1. generate key, sequence wise
        key = output.contiguous().view(-1, output.size(2))
        key = self.key_layer(key)
        key = key.view(output.size(0), output.size(1), -1)
        return seq_sizes, key.permute(1, 2, 0), value.permute(1, 0, 2)


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
        self.con = torch.nn.Parameter()
        self.init_state()

    def forward(self, seq_sizes, key, value, labels):
        batch_size = seq_sizes.size(0)
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
        # context = torch.zeros((batch_size, self.context_size))
        context = self.con.repeat(batch_size, 1)
        score_list, predict_list = [], []
        for i in range(max_steps):
            if CUDA:
                char = char.cuda()
            # run one step
            score, predict, hidden, cell, context = self.forward_one_step(char, seq_sizes, hidden, cell, key, value, context)
            score_list.append(score)
            predict_list.append(predict)
            teacher_force = labels is not None and self.training and np.random.uniform() < TEACHER_FORCE_RATE
            # use given transcripts if teacher force
            if teacher_force:
                char = labels[:, i]
            else:
                char = torch.max(predict, dim=1)[1]
        predict_list = torch.stack(predict_list, dim=1)
        return score_list, predict_list

    def forward_one_step(self, char, seq_sizes, hidden, cell, key, value, context):
        context = context.cuda()
        # embedding of last predict result
        char_embed = self.embed(char)
        # cat embedding and context together

        rnn_input = torch.cat([char_embed, context], dim=1)

        new_hidden, new_cell = [None] * self.num_layers, [None] * self.num_layers
        # rnns
        i = 0
        for rnn in self.rnns:
            new_hidden[i], new_cell[i] = rnn(rnn_input, (hidden[i], cell[i]))
            rnn_input = new_hidden[i]
            i += 1

        # 1. generate query
        query = self.query_layer(new_hidden[-1]).unsqueeze(1)

        # 2. energy function: query(N, 1, A); keys(N, A, L); energy(N, 1, L) --> # energy(N, L)
        energy = torch.bmm(query, key).squeeze(1)

        # 3. SoftMax over energy over utterance
        score = F.softmax(energy, dim=1)

        mask = torch.zeros_like(score)
        for i in range(len(seq_sizes)):
            mask[i, :seq_sizes[i]] = 1
        if CUDA:
            mask = mask.cuda()
        score = score * mask
        # (N, L) -->  (N, 1) --> (N, L)
        score = torch.nn.functional.normalize(score, p=1, dim=1)

        # 5. bmm(attention(N, 1, L), values(N, L, B) ----> context(N, 1, B)  --> context(N, B)
        new_context = torch.bmm(score.unsqueeze(1), value).squeeze(1)
        # print("context")
        # print(new_context)
        # cat features
        cat_features = torch.cat([new_hidden[-1], new_context], dim=1)
        # get predict result
        predict = self.scoring(cat_features)
        return score, predict, new_hidden, new_cell, new_context

    def init_state(self):
        for i in range(self.num_layers):
            self.h_0.append(torch.nn.Parameter(torch.zeros((1, self.hidden_size))))
            self.c_0.append(torch.nn.Parameter(torch.zeros((1, self.hidden_size))))
        self.con = torch.nn.Parameter(torch.zeros((1, self.context_size)))


# implementation for LAS model
class LASModel(nn.Module):
    def __init__(self):
        super(LASModel, self).__init__()
        self.listener = ListenerModel(input_size=40)
        self.speller = SpellerModel()
        self.max_len = MAX_LEN

    def forward(self, frames, seq_sizes, labels):
        # listener
        seq_sizes, key, value = self.listener(frames, seq_sizes)
        # speller
        score_list, predict_list = self.speller(seq_sizes, key, value, labels)

        return score_list, predict_list


def random_decode(probs):
    return ""


class TranslateModelTrainer:
    def __init__(self, run_id, model, loaders, max_epochs=EPOCH_NUM, weight_decay=WEIGHT_DECAY):
        self.model = model
        self.run_id = run_id
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
        log = open(os.path.join('experiments', self.run_id, 'log.txt'), 'a')
        torch.set_grad_enabled(True)
        loss_sum = 0
        print("----training----")
        log = open(os.path.join('log2.txt'), 'a')
        for batch_num, (inputs, targets, seq_len_list, label_len_list) in enumerate(self.train_loader):
            print(batch_num)
            loss = self.train_batch(batch_num, inputs, targets, seq_len_list, label_len_list)
            loss_sum += loss
        loss_sum = loss_sum / self.num_phonemes_train
        self.epochs += 1
        self.scheduler.step(loss_sum)
        print('[TRAIN]    Loss: %.4f ' % loss_sum)
        log.write('[TRAIN]   Loss: %.4f\n' % loss_sum)
        log.close()

    def train_batch(self, batch_num, inputs, targets, seq_len_list, label_len_list):
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

        if batch_num % 100 == 0:
            at_path = os.path.join('experiments', self.run_id, 'at-{}.txt'.format(self.epochs))
            torch.save(score, at_path)

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
        self.scheduler.step(loss_sum)
        print('[VAL]    Loss: %.4f  Dis: %.4f' % (loss_sum, dis_sum))
        log.write('[VAL]   Loss: %.4f  Dis: %.4f\n' % (loss_sum, dis_sum))
        log.close()
        return dis_sum

    def validate_batch(self, inputs, targets, seq_len_list, label_len_list):
        err = 0
        if self.cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

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


def test(model, test_loader, cuda, save_path):
    model.eval()  # set to eval mode
    torch.set_grad_enabled(False)
    print("----test----")
    with open(os.path.join(save_path, 'submission.csv'), 'a') as file:
        file.write("Id,Predicted\n")
        id = 0
        for batch_num, (inputs, _, seq_len_list, _) in enumerate(test_loader):
            if cuda:
                inputs = inputs.cuda()
            outputs = model(inputs, seq_len_list)
            decode_str = random_decode(outputs, seq_len_list)
            for line in decode_str:
                file.write(str(id) + "," + line + "\n")
                id += 1
    print("test done")


def main():
    # loading data
    path = "/home/ubuntu/hw4/data/"
    save_path = "/home/ubuntu/hw4/model/"
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
    run_id = str(int(time.time()))
    if not os.path.exists('./experiments'):
        os.mkdir('./experiments')
    os.mkdir('./experiments/%s' % run_id)
    # model.load_state_dict(torch.load("/home/ubuntu/hw3p2/model/model-19-10.5392.txt"))
    trainer = TranslateModelTrainer(run_id, model, loaders, max_epochs=EPOCH_NUM)

    best_err = 1e3  # set to super large value at first
    for epoch in range(EPOCH_NUM):
        print("starting epoch:")
        print(epoch)
        trainer.train()
        err = trainer.validate()
        if err < 0.5:
            model_path = os.path.join('experiments', 'model-{:.4f}.txt'.format(err))
            torch.save(model.state_dict(), model_path)
        print(str(epoch) + " with error: " + str(err))
    # test(model, test_loader, cuda, save_path)


if __name__ == '__main__':
    main()