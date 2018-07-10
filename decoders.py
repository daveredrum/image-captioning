import math
import torch
import copy
import numpy as np
import torch.nn as nn
from collections import deque
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision.models as torchmodels
import torch.nn.functional as F
from torch.nn import Parameter
import configs


# decoder without attention
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, cuda_flag=True):
        super(Decoder, self).__init__()
        # the size of inputs and outputs should be equal to the size of the dictionary
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm_layer = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            # nn.Dropout(p=0.2)
        )
        self.cuda_flag = cuda_flag

    def init_hidden(self, visual_inputs):
        states = (
            Variable(torch.zeros(visual_inputs.size(0), self.hidden_size), requires_grad=False).cuda(),
            Variable(torch.zeros(visual_inputs.size(0), self.hidden_size), requires_grad=False).cuda()
        )

        return states

    def forward(self, features, caption_inputs, states):
        # feed
        seq_length = caption_inputs.size(1) + 1
        decoder_outputs = []
        for step in range(seq_length):
            if step == 0:
                embedded = features
            else:
                embedded = self.embedding(caption_inputs[:, step - 1])
            states = self.lstm_layer(embedded, states)
            lstm_outputs = states[0]
            outputs = self.output_layer(lstm_outputs).unsqueeze(1)
            decoder_outputs.append(outputs)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)

        return decoder_outputs


    def sample(self, embedded, states):
        new_states = self.lstm_layer(embedded, states)
        lstm_outputs = new_states[0]
        outputs = self.output_layer(lstm_outputs).unsqueeze(1)

        return outputs, new_states

    def beam_search(self, features, beam_size, max_length):
        batch_size = features.size(0)
        outputs = []
        for feat_id in range(batch_size):
            feature = features[feat_id].unsqueeze(0)
            states = self.init_hidden(feature)
            start, states = self.sample(feature, states)
            start = F.log_softmax(start, dim=2)
            start_scores, start_words = start.topk(beam_size, dim=2)[0].squeeze(), start.topk(beam_size, dim=2)[1].squeeze()
            # a queue containing all searched words and their log_prob
            searched = deque([([start_words[i].view(1)], start_scores[i].view(1), states) for i in range(beam_size)])
            done = []
            while True:
                candidate = searched.popleft()
                prev_word, prev_prob, prev_states = candidate
                if len(prev_word) <= max_length and int(prev_word[-1].item()) != 3:
                    embedded = self.embedding(prev_word[-1])
                    preds, new_states = self.sample(embedded, prev_states)
                    preds = F.log_softmax(preds, dim=2)
                    top_scores, top_words = preds.topk(beam_size, dim=2)[0].squeeze(), preds.topk(beam_size, dim=2)[1].squeeze()
                    for i in range(beam_size):
                        next_word, next_prob = copy.deepcopy(prev_word), prev_prob.clone()
                        next_word.append(top_words[i].view(1))
                        next_prob += top_scores[i].view(1)
                        searched.append((next_word, next_prob, new_states))
                    searched = deque(sorted(searched, reverse=True, key=lambda s: s[1])[:beam_size])    
                else:
                    done.append((prev_word, prev_prob))
                if not searched:
                    break       
            
            done = sorted(done, reverse=True, key=lambda s: s[1])
            best = [word[0].item() for word in done[0][0]]
            outputs.append(best)
        
        return outputs
                        

# attention module
class Attention2D(nn.Module):
    def __init__(self, visual_channels, hidden_size, visual_flat):
        super(Attention2D, self).__init__()
        # basic settings
        self.visual_channels = visual_channels
        self.hidden_size = hidden_size
        self.visual_flat = visual_flat
        # MLP
        self.comp_visual = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        self.comp_hidden = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 1, bias=False),
        )
        # initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            stdv = 1.0 / math.sqrt(weight.size(0))
            weight.data.uniform_(-stdv, stdv)
    
    def forward(self, visual_inputs, states):
        # visual_inputs = (batch_size, visual_flat, visual_channels)
        feature = visual_inputs.permute(0, 2, 1).contiguous()
        # get the hidden state
        hidden = states[0]
        # in = (batch_size, visual_flat, visual_channels)
        # out = (batch_size, visual_flat, hidden_size)
        V = self.comp_visual(feature)
        # in = (batch_size, hidden_size)
        # out = (batch_size, 1, hidden_size)
        H = self.comp_hidden(hidden).unsqueeze(1)
        # print("V", V.view(-1).min(0)[0].item(), V.view(-1).max(0)[0].item())
        # print("H", H.view(-1).min(0)[0].item(), H.view(-1).max(0)[0].item())
        # combine
        outputs = F.tanh(V + H)
        # outputs = (batch_size, visual_flat)
        outputs = self.output_layer(outputs).squeeze(2)
        outputs = F.softmax(outputs, dim=1)

        return outputs

# adaptive attention module
class AdaptiveAttention2D(nn.Module):
    def __init__(self, visual_channels, hidden_size, visual_flat):
        super(AdaptiveAttention2D, self).__init__()
        # basic settings
        self.visual_channels = visual_channels
        self.hidden_size = hidden_size
        self.visual_flat = visual_flat
        # MLP
        self.comp_visual = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        self.comp_hidden = nn.Sequential(
            nn.Linear(hidden_size, 1, bias=False),
        )
        self.comp_sentinel = nn.Sequential(
            nn.Linear(hidden_size, 1, bias=False),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 1, bias=False),
        )
        # initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            stdv = 1.0 / math.sqrt(weight.size(0))
            weight.data.uniform_(-stdv, stdv)
    
    def forward(self, visual_inputs, states, sentinel):
        # visual_inputs = (batch_size, visual_flat, visual_channels)
        feature = visual_inputs.permute(0, 2, 1).contiguous()
        # get the hidden state
        hidden = states[0]
        # in = (batch_size, visual_flat, visual_channels)
        # out = (batch_size, visual_flat, hidden_size)
        V = self.comp_visual(feature)
        # in = (batch_size, hidden_size)
        # out = (batch_size, 1, 1)
        H = self.comp_hidden(hidden).unsqueeze(1)
        # combine
        Z = F.tanh(V + H)
        # Z = (batch_size, visual_flat)
        Z = self.output_layer(Z).squeeze(2)
        # sentinel outputs
        # S = (batch_size, 1)
        S = F.tanh(self.comp_sentinel(sentinel) + self.comp_hidden(hidden))
        # concat
        # outputs = (batch_size, visual_flat + 1)
        outputs = torch.cat((Z, S), dim=1)
        outputs = F.softmax(outputs, dim=1)

        return outputs

# new LSTM with visual attention context
class Att2AllLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Att2AllLSTMCell, self).__init__()
        # basic settings
        self.input_size = input_size
        self.hidden_size = hidden_size
        # parameters
        for gate in ["i", "f", "c", "o"]:
            setattr(self, "w_{}".format(gate), Parameter(torch.Tensor(input_size, hidden_size)))
            setattr(self, "u_{}".format(gate), Parameter(torch.Tensor(hidden_size, hidden_size)))
            setattr(self, "z_{}".format(gate), Parameter(torch.Tensor(hidden_size, hidden_size)))
            setattr(self, "b_{}".format(gate), Parameter(torch.Tensor(hidden_size)))
        # initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            stdv = 1.0 / math.sqrt(weight.size(0))
            weight.data.uniform_(-stdv, stdv)
    
    # inputs = (batch, input_size)
    # states_h = (batch, hidden_size)
    # states_c = (batch, hidden_size)
    # atteded = (batch, input_size)
    # outputs = (states_h, states_c)
    def forward(self, embedded, states, atteded):
        # unpack states
        states_h, states_c = states
        # forward feed
        i = F.sigmoid(torch.matmul(embedded, self.w_i) + torch.matmul(states_h, self.u_i) + torch.matmul(atteded, self.z_i) + self.b_i)
        f = F.sigmoid(torch.matmul(embedded, self.w_f) + torch.matmul(states_h, self.u_f) + torch.matmul(atteded, self.z_f) + self.b_f)
        c_hat = F.tanh(torch.matmul(embedded, self.w_c) + torch.matmul(states_h, self.u_c) + torch.matmul(atteded, self.z_c) + self.b_c)
        states_c = f * states_c + i * c_hat
        o = F.sigmoid(torch.matmul(embedded, self.w_o) + torch.matmul(states_h, self.u_o) + torch.matmul(atteded, self.z_o) + self.b_o)
        states_h = o * F.tanh(states_c)
        # pack states
        states = (states_h, states_c)

        return states


# new LSTM with visual attention context
class Att2InLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Att2InLSTMCell, self).__init__()
        # basic settings
        self.input_size = input_size
        self.hidden_size = hidden_size
        # parameters
        for gate in ["i", "f", "c", "o"]:
            setattr(self, "w_{}".format(gate), Parameter(torch.Tensor(input_size, hidden_size)))
            setattr(self, "u_{}".format(gate), Parameter(torch.Tensor(hidden_size, hidden_size)))
            setattr(self, "b_{}".format(gate), Parameter(torch.Tensor(hidden_size)))
        setattr(self, "z_c", Parameter(torch.Tensor(hidden_size, hidden_size)))
        # initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            stdv = 1.0 / math.sqrt(weight.size(0))
            weight.data.uniform_(-stdv, stdv)
    
    # inputs = (batch, input_size)
    # states_h = (batch, hidden_size)
    # states_c = (batch, hidden_size)
    # atteded = (batch, input_size)
    # outputs = (states_h, states_c)
    def forward(self, embedded, states, atteded):
        # unpack states
        states_h, states_c = states
        # forward feed
        i = F.sigmoid(torch.matmul(embedded, self.w_i) + torch.matmul(states_h, self.u_i) + self.b_i)
        f = F.sigmoid(torch.matmul(embedded, self.w_f) + torch.matmul(states_h, self.u_f) + self.b_f)
        c_hat = F.tanh(torch.matmul(embedded, self.w_c) + torch.matmul(states_h, self.u_c) + torch.matmul(atteded, self.z_c) + self.b_c)
        states_c = f * states_c + i * c_hat
        o = F.sigmoid(torch.matmul(embedded, self.w_o) + torch.matmul(states_h, self.u_o) + self.b_o)
        states_h = o * F.tanh(states_c)
        # pack states
        states = (states_h, states_c)

        return states


# LSTM with sentinel gate
class AdaptiveLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AdaptiveLSTMCell, self).__init__()
        # basic settings
        self.input_size = input_size
        self.hidden_size = hidden_size
        # parameters
        for gate in ["i", "f", "c", "o", 's']:
            setattr(self, "w_{}".format(gate), Parameter(torch.Tensor(input_size, hidden_size)))
            setattr(self, "u_{}".format(gate), Parameter(torch.Tensor(hidden_size, hidden_size)))
            setattr(self, "b_{}".format(gate), Parameter(torch.Tensor(hidden_size)))
        # initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            stdv = 1.0 / math.sqrt(weight.size(0))
            weight.data.uniform_(-stdv, stdv)
    
    # inputs = (batch, input_size)
    # states_h = (batch, hidden_size)
    # states_c = (batch, hidden_size)
    # states = (states_h, states_c)
    # sentinel = (batch, hidden_size)
    def forward(self, embedded, states, atteded):
        # unpack states
        states_h, states_c = states
        # forward feed
        i = F.sigmoid(torch.matmul(embedded, self.w_i) + torch.matmul(states_h, self.u_i) + self.b_i)
        f = F.sigmoid(torch.matmul(embedded, self.w_f) + torch.matmul(states_h, self.u_f) + self.b_f)
        c_hat = F.tanh(torch.matmul(embedded, self.w_c) + torch.matmul(states_h, self.u_c) + self.b_c)
        states_c = f * states_c + i * c_hat
        o = F.sigmoid(torch.matmul(embedded, self.w_o) + torch.matmul(states_h, self.u_o) + self.b_o)
        states_h = o * F.tanh(states_c)
        # pack states
        states = (states_h, states_c)
        # sentinel gate
        s_hat = F.sigmoid(torch.matmul(embedded, self.w_s) + torch.matmul(states_h, self.u_s) + self.b_s)
        sentinel = s_hat * F.tanh(states_c)

        return states, sentinel

# decoder with attention
class AttentionDecoder2D(nn.Module):
    def __init__(self, attention, batch_size, input_size, hidden_size, visual_channels, visual_size, num_layers=1, cuda_flag=True):
        super(AttentionDecoder2D, self).__init__()
        # basic settings
        self.attention = attention
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.visual_channels = visual_channels
        self.visual_size = visual_size
        self.visual_flat = visual_size * visual_size
        self.visual_feature_size = visual_channels * visual_size * visual_size
        self.proj_size = 512
        self.feat_size = 512
        self.num_layers = num_layers
        self.cuda_flag = cuda_flag
        # layer settings
        # # initialize hidden states
        # self.init_h = nn.Sequential(
        #     nn.Linear(self.visual_channels, hidden_size),
        # )
        # self.init_c = nn.Sequential(
        #     nn.Linear(self.visual_channels, hidden_size),
        # )
        # embedding layer
        self.embedding = nn.Embedding(input_size, hidden_size)

        # LSTM
        # attention module
        if attention == 'att2all':
            self.attention = Attention2D(self.visual_channels, self.hidden_size, self.visual_flat)
            self.lstm_layer_1 = Att2AllLSTMCell(2 * self.hidden_size, self.hidden_size)
        elif attention == 'att2in':
            self.attention = Attention2D(self.visual_channels, self.hidden_size, self.visual_flat)
            self.lstm_layer_1 = Att2InLSTMCell(2 * self.hidden_size, self.hidden_size)
        elif attention == 'spatial':
            self.attention = Attention2D(self.visual_channels, self.hidden_size, self.visual_flat)
            self.lstm_layer_1 = nn.LSTMCell(2 * self.hidden_size, self.hidden_size)
        elif attention == 'adaptive':
            self.attention = AdaptiveAttention2D(self.visual_channels, self.hidden_size, self.visual_flat)
            self.lstm_layer_1 = AdaptiveLSTMCell(2 * self.hidden_size, self.hidden_size)
        else:
            raise Exception('invalid attention type, terminating...')
        
        # output layer
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_size + self.hidden_size, self.input_size),
            # nn.Dropout(p=0.2)
        )


    def init_hidden(self, visual_inputs):
        visual_flat = visual_inputs.view(visual_inputs.size(0), visual_inputs.size(1), visual_inputs.size(2) * visual_inputs.size(2))
        visual_flat = visual_flat.mean(2)
        # states = (
        #     self.init_h(visual_flat),
        #     self.init_c(visual_flat)
        # )
        states = (
            Variable(torch.zeros(visual_inputs.size(0), self.hidden_size)).cuda(),
            Variable(torch.zeros(visual_inputs.size(0), self.hidden_size)).cuda()
        ) 

        return states

    def forward(self, features, caption_inputs, states):
        _, global_features, area_features = features
        # feed
        seq_length = caption_inputs.size(1)
        decoder_outputs = []
        for step in range(seq_length):
            if self.attention == 'spatial':
                embedded = self.embedding(caption_inputs[:, step])
                lstm_input = torch.cat((embedded, global_features), dim=1)
                states = self.lstm_layer_1(lstm_input, states)
                lstm_outputs = states[0]
                attention_weights = self.attention(area_features, states)
                attended = torch.sum(area_features * attention_weights.unsqueeze(1), 2)
                outputs = torch.cat((attended, lstm_outputs), dim=1)
                outputs = self.output_layer(outputs).unsqueeze(1)
                decoder_outputs.append(outputs)
            elif self.attention == 'adaptive':
                embedded = self.embedding(caption_inputs[:, step])
                lstm_input = torch.cat((embedded, global_features), dim=1)
                states, sentinel = self.lstm_layer_1(lstm_input, states)
                lstm_outputs = states[0]
                attention_weights = self.attention(area_features, states, sentinel)
                attended = torch.sum(area_features * attention_weights[:, :-1].unsqueeze(1), 2)
                sentinel_scalar = attention_weights[:, -1]
                attended = sentinel_scalar * sentinel + (1 - sentinel_scalar) * attended
                outputs = torch.cat((attended, lstm_outputs), dim=1)
                outputs = self.output_layer(outputs).unsqueeze(1)
                decoder_outputs.append(outputs)
            else:
                embedded = self.embedding(caption_inputs[:, step])
                lstm_input = torch.cat((embedded, global_features), dim=1)
                attention_weights = self.attention(area_features, states)
                attended = torch.sum(area_features * attention_weights.unsqueeze(1), 2)
                states = self.lstm_layer_1(lstm_input, states, attended)
                lstm_outputs = states[0]
                outputs = torch.cat((attended, lstm_outputs), dim=1)
                outputs = self.output_layer(outputs).unsqueeze(1)
                decoder_outputs.append(outputs)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)

        return decoder_outputs 

    def sample(self, features, caption_inputs, states):
        _, global_features, area_features = features
        if self.attention == 'spatial':
            embedded = self.embedding(caption_inputs)
            lstm_input = torch.cat((embedded, global_features), dim=1)
            new_states = self.lstm_layer_1(lstm_input, states)
            lstm_outputs = new_states[0]
            attention_weights = self.attention(area_features, states)
            attended = torch.sum(area_features * attention_weights.unsqueeze(1), 2)
            outputs = torch.cat((attended, lstm_outputs), dim=1)
            outputs = self.output_layer(outputs).unsqueeze(1)
        elif self.attention == 'adaptive':
            embedded = self.embedding(caption_inputs)
            lstm_input = torch.cat((embedded, global_features), dim=1)
            new_states, sentinel = self.lstm_layer_1(lstm_input, states)
            lstm_outputs = new_states[0]
            attention_weights = self.attention(area_features, states, sentinel)
            attended = torch.sum(area_features * attention_weights[:, :-1].unsqueeze(1), 2)
            sentinel_scalar = attention_weights[:, -1]
            attended = sentinel_scalar * sentinel + (1 - sentinel_scalar) * attended
            outputs = torch.cat((attended, lstm_outputs), dim=1)
            outputs = self.output_layer(outputs).unsqueeze(1)
        else:
            embedded = self.embedding(caption_inputs)
            lstm_input = torch.cat((embedded, global_features), dim=1)
            attention_weights = self.attention(area_features, states)
            attended = torch.sum(area_features * attention_weights.unsqueeze(1), 2)
            new_states = self.lstm_layer_1(lstm_input, states, attended)
            lstm_outputs = new_states[0]
            outputs = torch.cat((attended, lstm_outputs), dim=1)
            outputs = self.output_layer(outputs).unsqueeze(1)

        return outputs, new_states, attention_weights

    def beam_search(self, features, caption_inputs, beam_size, max_length):
        batch_size = features[0].size(0)
        outputs = []
        for feat_id in range(batch_size):
            feats = (
                features[0][feat_id].unsqueeze(0),
                features[1][feat_id].unsqueeze(0),
                features[2][feat_id].unsqueeze(0),
            )
            states = self.init_hidden(feats[0])
            start, states, _ = self.sample(feats, caption_inputs[feat_id, 0].view(1), states)
            start = F.log_softmax(start, dim=2)
            start_scores, start_words = start.topk(beam_size, dim=2)[0].squeeze(), start.topk(beam_size, dim=2)[1].squeeze()
            # a queue containing all searched words and their log_prob
            searched = deque([([start_words[i].view(1)], start_scores[i].view(1), states) for i in range(beam_size)])
            done = []
            while True:
                candidate = searched.popleft()
                prev_word, prev_prob, prev_states = candidate
                if len(prev_word) <= max_length and int(prev_word[-1].item()) != 3:
                    preds, new_states, _ = self.sample(feats, prev_word[-1], prev_states)
                    preds = F.log_softmax(preds, dim=2)
                    top_scores, top_words = preds.topk(beam_size, dim=2)[0].squeeze(), preds.topk(beam_size, dim=2)[1].squeeze()
                    for i in range(beam_size):
                        next_word, next_prob = copy.deepcopy(prev_word), prev_prob.clone()
                        next_word.append(top_words[i].view(1))
                        next_prob += top_scores[i].view(1)
                        searched.append((next_word, next_prob, new_states))
                    searched = deque(sorted(searched, reverse=True, key=lambda s: s[1])[:beam_size])
                else:
                    done.append((prev_word, prev_prob))
                if not searched:
                    break           
            
            done = sorted(done, reverse=True, key=lambda s: s[1])
            best = [word[0].item() for word in done[0][0]]
            outputs.append(best)
        
        return outputs

# pipeline for pretrained encoder-decoder pipeline
# same pipeline for both 2d and 3d
class EncoderDecoder():
    def __init__(self, encoder_path, decoder_path, cuda_flag=True):
        if cuda_flag:
            self.encoder = torch.load(encoder_path).cuda()
            self.decoder = torch.load(decoder_path).cuda()
        else:
            self.encoder = torch.load(encoder_path)
            self.decoder = torch.load(decoder_path)
        # set mode
        self.encoder.eval()
        self.decoder.eval()

    def generate_text(self, image_inputs, dictionary, max_length):
        inputs = self.encoder.extract(image_inputs).unsqueeze(1)
        states = None
        # sample text indices via greedy search
        sampled = []
        for i in range(max_length):
            outputs, states = self.decoder.lstm_layer(inputs, states)
            outputs = self.decoder.output_layer(outputs[0])
            predicted = outputs.max(1)[1]
            sampled.append(predicted.view(-1, 1))
            inputs = self.decoder.embedding(predicted).unsqueeze(1)
        sampled = torch.cat(sampled, 1)
        # decoder indices to words
        captions = []
        for sequence in sampled.cpu().numpy():
            caption = []
            for index in sequence:
                word = dictionary[index]
                caption.append(word)
                if word == '<END>':
                    break
            captions.append(" ".join(caption))

        return captions

# for encoder-decoder pipeline with attention
class AttentionEncoderDecoder():
    def __init__(self, encoder_path, decoder_path, pretrained, cuda_flag=True):
        if cuda_flag:
            self.encoder = torch.load(encoder_path).cuda()
            self.decoder = torch.load(decoder_path).cuda()
        else:
            self.encoder = torch.load(encoder_path)
            self.decoder = torch.load(decoder_path)
        # set mode
        self.encoder.eval()
        self.decoder.eval()

        # set size
        if pretrained == 'vgg':
            self.size = 14
        elif pretrained == 'resnet':
            self.size = 7
        else:
            raise Exception('invalid name for pretrained model, terminating...')

    def generate_text(self, image_inputs, dict_word2idx, dict_idx2word, max_length):
        caption_inputs = Variable(torch.LongTensor(np.reshape(np.array(dict_word2idx["<START>"]), (1)))).cuda()
        visual_contexts = self.encoder(image_inputs)
        # sample text indices via greedy search
        sampled = []
        states = self.decoder.init_hidden(visual_contexts[0])
        for _ in range(max_length):
            outputs, states, _ = self.decoder.sample(visual_contexts, caption_inputs, states)
            # outputs = (1, 1, input_size)
            predicted = outputs.max(2)[1]
            # predicted = (1, 1)
            sampled.append(predicted)
            caption_inputs = predicted.view(1)
            if dict_idx2word[caption_inputs[-1].view(1).cpu().numpy()[0]] == '<END>':
                break
        sampled = torch.cat(sampled)
        # decoder indices to words
        caption = ['<START>']
        for index in sampled.cpu().numpy():
            word = dict_idx2word[index[0]]
            caption.append(word)
            if word == '<END>':
                break

        return caption

    # image_inputs = (1, visual_channels, visual_size, visual_size)
    # caption_inputs = (1)
    def visual_attention(self, image_inputs, dict_word2idx, dict_idx2word, max_length):
        caption_inputs = Variable(torch.LongTensor(np.reshape(np.array(dict_word2idx["<START>"]), (1)))).cuda()
        visual_contexts = self.encoder(image_inputs)
        # sample text indices via greedy search
        pairs = []
        states = self.decoder.init_hidden(visual_contexts[0])
        for _ in range(max_length):
            outputs, states, attention_weights = self.decoder.sample(visual_contexts, caption_inputs, states)
            # attentions = (visual_size, visual_size)
            predicted = outputs.max(2)[1]
            # predicted = (1, 1)
            caption_inputs = predicted.view(1)
            word = dict_idx2word[predicted.cpu().numpy()[0][0]]
            up_weights = F.upsample(attention_weights.view(1, 1, visual_contexts[0].size(2), visual_contexts[0].size(2)), size=(configs.COCO_SIZE, configs.COCO_SIZE), mode='bilinear')
            pairs.append((word, attention_weights.view(self.size, self.size), up_weights.view(configs.COCO_SIZE, configs.COCO_SIZE), states[0][0]))
            if word == '<END>':
                break

        return pairs