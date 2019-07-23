import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import numpy as np
import dropout


def make_mask(lens):
    max_len = max(lens)
    mask = torch.zeros((lens.shape[0], max_len))
    for ii in range(mask.shape[0]):
        mask[ii][:lens[ii]] = 1

    return mask


class Listener(nn.Module):
    def __init__(self):
        super(Listener, self).__init__()

        self.embed_size = 80
        self.hidden_size = 256
        self.key_size = 256
        self.value_size = 256

        self.dropout = [0.3, 0.1, 0.2]
        self.DropOutLayer = nn.ModuleList()
        self.rnn_layer = 3

        self.rnn1 = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size,
                            num_layers=1, bidirectional=True)  # Recurrent network

        self.rnn2 = nn.LSTM(input_size=self.hidden_size*4, hidden_size=self.hidden_size,
                            num_layers=1, bidirectional=True)  # Recurrent network

        self.rnn3 = nn.LSTM(input_size=self.hidden_size*4, hidden_size=self.hidden_size,
                            num_layers=1, bidirectional=True)  # Recurrent network

        self.key_fc = nn.Linear(self.hidden_size*2, self.key_size)
        self.value_fc = nn.Linear(self.hidden_size*2, self.value_size)

        for ii in range(self.rnn_layer):
            self.DropOutLayer.append(dropout.LockedDropout())

    def pyramid_pack_padded(self, padded, lens):
        padded = padded.permute(1, 0, 2)
        batch_size = padded.shape[0]
        length = padded.shape[1]
        dim = padded.shape[2]

        half_length = length//2
        reshaped_size = half_length*2*dim*batch_size
        reshaped_seq = padded.contiguous().view(batch_size * length * dim)[:reshaped_size].\
            view(batch_size, half_length, dim*2)
        reshaped_seq = reshaped_seq.permute(1, 0, 2)
        packed = rnn.pack_padded_sequence(reshaped_seq, [ll // 2 for ll in lens])

        return packed

    def forward(self, seq_batch, training=True):
        lens = [len(seq) for seq in seq_batch]
        padded = rnn.pad_sequence(seq_batch, batch_first=False)

        if torch.cuda.is_available():
            padded = padded.cuda()

        packed1 = self.pyramid_pack_padded(padded, lens)

        hidden1 = None
        rnn_out1, hidden1 = self.rnn1(packed1)
        padded1, lens1 = rnn.pad_packed_sequence(rnn_out1)
        output_dp1 = self.DropOutLayer[0](padded1, training, self.dropout[0])

        packed2 = self.pyramid_pack_padded(output_dp1, lens1)
        hidden2 = None
        rnn_out2, hidden2 = self.rnn2(packed2)
        padded2, lens2 = rnn.pad_packed_sequence(rnn_out2)
        output_dp2 = self.DropOutLayer[1](padded2, training, self.dropout[1])

        packed3 = self.pyramid_pack_padded(output_dp2, lens2)
        hidden3 = None
        rnn_out3, hidden3 = self.rnn3(packed3)
        padded3, lens3 = rnn.pad_packed_sequence(rnn_out3)
        output_dp3 = self.DropOutLayer[2](padded3, training, self.dropout[2])

        key = self.key_fc(output_dp3)
        value = self.value_fc(output_dp3)

        mask = make_mask(lens3)

        return key.permute(1, 2, 0), value.permute(1, 0, 2), mask


class Speller(nn.Module):
    def __init__(self, batch_size):
        super(Speller, self).__init__()
        self.max_len = 280
        self.input_size = 10
        self.hidden_size = 512
        self.n_classes = 33
        self.query_size = 256
        self.context_size = 256
        self.teacher_force_rate = 0.9
        self.batch_size = batch_size
        self.rnn_layer = 3
        self.hidden_init_state = []
        self.cell_init_state = []
        self.dropout = [0.1, 0.1, 0.3]
        # self.DropOutLayer = nn.ModuleList()

        self.RNNCell1 = nn.LSTMCell(self.hidden_size + self.context_size, self.hidden_size)
        self.RNNCell2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.RNNCell3 = nn.LSTMCell(self.hidden_size, self.hidden_size)

        for ii in range(self.rnn_layer):
            hidden_state = nn.Parameter(torch.rand(self.batch_size, self.hidden_size))
            self.hidden_init_state.append(hidden_state)
            cell_state = nn.Parameter(torch.rand(self.batch_size, self.hidden_size))
            self.cell_init_state.append(cell_state)

        self.embedding = nn.Linear(self.n_classes, self.hidden_size)
        self.query_projection = nn.Linear(self.hidden_size, self.query_size)
        self.fc = nn.Linear(self.hidden_size + self.context_size, self.hidden_size)
        self.activate = nn.ReLU()
        self.output_layer = nn.Linear(self.hidden_size, self.n_classes)

    def attention_calculations(self, key, query, value, mask):
        energy = torch.bmm(query.unsqueeze(1), key).squeeze(1)
        energy = F.softmax(energy, dim=1)
        masked = torch.mul(energy, mask)
        masked_sum = torch.sum(masked, dim=1).view(masked.shape[0], 1)
        attention = (masked / masked_sum).unsqueeze(1)
        c = torch.bmm(attention, value)

        return attention.squeeze(1), c.squeeze(1)

    def forward_one(self, input_char, key, value, mask, context, hidden_state, cell_state, training=True):
        if torch.cuda.is_available():
            input_char = input_char.cuda()
            context = context.cuda()
            hidden_state = [state.cuda() for state in hidden_state]
            cell_state = [state.cuda() for state in cell_state]
            key = key.cuda()
            value = value.cuda()
            mask = mask.cuda()

        char_embed = self.embedding(input_char)    # embedding layer
        # batch_size = char_embed.shape[0]
        lstm_input = torch.cat((char_embed, context), dim=1)

        new_hidden_state = []
        new_cell_state = []
        new_hidden, new_cell = self.RNNCell1(lstm_input, (hidden_state[0], cell_state[0]))
        # output_dp1 = self.DropOutLayer[0](new_hidden, training, self.dropout[0])
        new_hidden_state.append(new_hidden)
        new_cell_state.append(new_cell)

        new_hidden, new_cell = self.RNNCell2(new_hidden, (hidden_state[1], cell_state[1]))
        # output_dp2 = self.DropOutLayer[1](new_hidden, training, self.dropout[1])
        new_hidden_state.append(new_hidden)
        new_cell_state.append(new_cell)

        new_hidden, new_cell = self.RNNCell3(new_hidden, (hidden_state[2], cell_state[2]))
        # output_dp3 = self.DropOutLayer[2](new_hidden, training, self.dropout[2])
        new_hidden_state.append(new_hidden)
        new_cell_state.append(new_cell)

        query = self.query_projection(new_hidden)
        attention, context = self.attention_calculations(key, query, value, mask)
        hidden = torch.cat((new_hidden, context), dim=1)
        fc_output = self.fc(hidden)
        activated = self.activate(fc_output)
        logits = self.output_layer(activated)

        return new_hidden_state, new_cell_state, logits, attention, context

    def forward(self, key, value, mask, training=True, targets=None):
        if targets[0] is not None:
            targets_padded = rnn.pad_sequence(targets, batch_first=False)

        # initialize context with all zeros
        batch_size = key.shape[0]
        context = torch.zeros((batch_size, self.context_size))
        hidden_state = [state[:batch_size] for state in self.hidden_init_state]
        cell_state = [state[:batch_size] for state in self.cell_init_state]
        if targets[0] is not None:
            input_char = self.make_one_hot(targets_padded[0])
        else:
            input_char = self.make_one_hot(torch.zeros([len(targets), 1], dtype=torch.int64))

        seq_logits = []
        attentions = []
        for ii in range(targets_padded.shape[0]-1 if targets[0] is not None else self.max_len):
            hidden_state, cell_state, logits, attention, context = self.forward_one(input_char, key, value,
                                                                                    mask, context, hidden_state,
                                                                                    cell_state, training)
            seq_logits.append(logits)
            attentions.append(attention)

            if targets[0] is not None:
                if np.random.random_sample() < self.teacher_force_rate:
                    input_char = self.make_one_hot(targets_padded[ii+1])
                else:
                    input_char = logits
            else:
                input_char = logits

        return torch.stack(seq_logits, dim=1), attentions

    def make_one_hot(self, logits):
        batch_size = logits.shape[0]
        one_hot = torch.zeros((batch_size, self.n_classes))
        one_hot[range(batch_size), logits] = 1
        return one_hot


class LAS(nn.Module):
    def __init__(self, batch_size):
        super(LAS, self).__init__()
        self.listener = Listener()
        self.speller = Speller(batch_size)
        self.seq_loss = SeqLoss()

    def forward(self, x, targets, training=True):
        key, value, mask = self.listener(x, training=training)
        logits, attention = self.speller(key, value, mask, training=training, targets=targets)
        return logits, attention


class SeqLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        loss = 0
        if torch.cuda.is_available():
            preds = preds.cuda()
            targets = [target.cuda() for target in targets]

        for pred, target in zip(preds, targets):
            target = target[1:]
            target_len = target.shape[0]
            pred = pred[:target_len]
            loss += F.cross_entropy(pred, target, reduction='sum')

        return loss/preds.shape[0]
