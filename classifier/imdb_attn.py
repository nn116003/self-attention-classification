from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from torch.autograd import Variable

import unicodedata
import string

import dill

from itertools import chain



class EncoderRNN(nn.Module):
    def __init__(self, emb_dim, h_dim, v_size, gpu=True, v_vec=None, batch_first=True):
        super(EncoderRNN, self).__init__()
        self.gpu = gpu
        self.h_dim = h_dim
        self.embed = nn.Embedding(v_size, emb_dim)
        if v_vec is not None:
            self.embed.weight.data.copy_(v_vec)
        self.lstm = nn.LSTM(emb_dim, h_dim, batch_first=batch_first,
                            bidirectional=True)

    def init_hidden(self, b_size):
        h0 = Variable(torch.zeros(1*2, b_size, self.h_dim))
        c0 = Variable(torch.zeros(1*2, b_size, self.h_dim))
        if self.gpu:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return (h0, c0)

    def forward(self, sentence, lengths=None):
        self.hidden = self.init_hidden(sentence.size(0))
        emb = self.embed(sentence)
        packed_emb = emb

        if lengths is not None:
            lengths = lengths.view(-1).tolist()
            packed_emb = nn.utils.rnn.pack_padded_sequence(emb, lengths)

        out, hidden = self.lstm(packed_emb, self.hidden)
        
        if lengths is not None:
            out = nn.utils.rnn.pad_packed_sequence(output)[0]
        
        out = out[:, :, :self.h_dim] + out[:, :, self.h_dim:]
        
        return out


class Attn(nn.Module):
    def __init__(self, h_dim):
        super(Attn, self).__init__()
        self.h_dim = h_dim
        self.main = nn.Sequential(
            nn.Linear(h_dim, 24),
            nn.ReLU(True),
            nn.Linear(24,1)
        )

    def forward(self, encoder_outputs):
        b_size = encoder_outputs.size(0)
        attn_ene = self.main(encoder_outputs.view(-1, self.h_dim)) # (b, s, h) -> (b * s, 1)
        return F.softmax(attn_ene.view(b_size, -1), dim=1).unsqueeze(2) # (b*s, 1) -> (b, s, 1)

class AttnClassifier(nn.Module):
    def __init__(self, h_dim, c_num):
        super(AttnClassifier, self).__init__()
        self.attn = Attn(h_dim)
        self.main = nn.Linear(h_dim, c_num)
        
    
    def forward(self, encoder_outputs):
        attns = self.attn(encoder_outputs) #(b, s, 1)
        feats = (encoder_outputs * attns).sum(dim=1) # (b, s, h) -> (b, h)
        return F.log_softmax(self.main(feats)), attns
        

def train_model(epoch, train_iter, optimizer, log_interval=10):
    encoder.train()
    classifier.train()
    correct = 0
    for idx, batch in enumerate(train_iter):
        (x, x_l), y = batch.text, batch.label - 1
        optimizer.zero_grad()
        encoder_outputs = encoder(x)
        output, attn = classifier(encoder_outputs)
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
    
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(y.data.view_as(pred)).cpu().sum()
        if idx % log_interval == 0:
            print('train epoch: {} [{}/{}], acc:{}, loss:{}'.format(
                epoch, idx*len(x), len(train_iter)*args.batch_size,
                correct/float(log_interval * len(x)),
                loss.data[0]))
            correct = 0

def test_model(epoch, test_iter):
    encoder.eval()
    classifier.eval()
    correct = 0
    for idx, batch in enumerate(test_iter):
        (x, x_l), y = batch.text, batch.label - 1
        encoder_outputs = encoder(x)
        output, attn = classifier(encoder_outputs)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(y.data.view_as(pred)).cpu().sum()
        
    print('test epoch:{}, acc:{}'.format(epoch, correct/float(len(test))))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch IMDB Example')
    parser.add_argument('--h-dim', type=int, default=32, metavar='N',
                        help='hidden state dim (default: 32)')
    parser.add_argument('--emb_dim', type=int, default=100, metavar='N',
                        help='word embedding dim (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 3)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(0)
    if args.cuda:
        torch.cuda.manual_seed(0)

    # define Field
    TEXT = data.ReversibleField(lower=True, include_lengths=True, batch_first=True)
    LABEL = data.Field(sequential=False)

    # make splits for data
    train, test = datasets.IMDB.splits(TEXT, LABEL)

    # build the vocabulary
    TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=args.emb_dim))
    LABEL.build_vocab(train)

    # save data field
    dill.dump(TEXT, open("TEXT.pkl",'wb'))
    dill.dump(LABEL, open("LABEL.pkl",'wb'))

    # make iterator for splits
    device = 0 if args.cuda else -1
    train_iter, test_iter = data.BucketIterator.splits(
        (train, test), 
        batch_size=args.batch_size, device=device, #sort_key=lambda x:len(x.text),
        #sort_within_batch=True, 
        repeat=False)


    # make model
    encoder = EncoderRNN(args.emb_dim, args.h_dim, len(TEXT.vocab), 
                         gpu=args.cuda, v_vec = TEXT.vocab.vectors)
    classifier = AttnClassifier(args.h_dim, 2)
    if args.cuda:
        encoder.cuda()
        classifier.cuda()

    # init model
    def weights_init(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Embedding') == -1):
            nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
            
    for m in encoder.modules():
        print(m.__class__.__name__)
        weights_init(m)

    for m in classifier.modules():
        print(m.__class__.__name__)
        weights_init(m)

    # optim
    optimizer = optim.Adam(chain(encoder.parameters(),classifier.parameters()), lr=args.lr)

    # train model 
    for epoch in range(args.epochs):
        train_model(epoch + 1, train_iter, optimizer)
        test_model(epoch + 1, test_iter)

    # save model
    dill.dump(encoder, open("encoder.pkl","wb"))
    dill.dump(classifier, open("classifier.pkl","wb"))
