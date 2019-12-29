import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F
import torchtext.data as data
from torchtext.datasets import text_classification
NGRAMS = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEXT = data.Field()
TARGETS = data.Field()
BATCH_SIZE = 16
train = data.TabularDataset(
path='E:\\Kaggle\\TweetDisaster\\train.csv', format='csv',
fields=[('text', data.Field()),
        ('target', data.Field())])
TEXT.build_vocab(train)
TARGETS.build_vocab(train)
class TrueOrFalse(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


VOCAB_SIZE = len(TEXT.vocab)
EMBED_DIM = 32
NUN_CLASS = len(TARGETS.vocab)
model = TrueOrFalse(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device)

from torch.utils.data import DataLoader
def generate_batch(batch):
    label = torch.tensor([getattr(entry,'target') for entry in batch])
    text = [getattr(entry,'text') for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    print(text)
    print(offsets)
    print(label)
    return text, offsets, label

def train_func(sub_train_):

    # Train the model
    train_loss = 0
    train_acc = 0
    print(sub_train_)
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn=generate_batch)

    for i, (text, offsets, cls) in enumerate(data):
        print(i)
        print(text)
        print(offsets)
        print(cls)
        optimizer.zero_grad()
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        output = model(text, offsets)
        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == cls).sum().item()

    # Adjust the learning rate
    scheduler.step()

    return train_loss / len(sub_train_), train_acc / len(sub_train_)

def test(data_):
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_batch)
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()

    return loss / len(data_), acc / len(data_)




import time
from torch.utils.data.dataset import random_split
N_EPOCHS = 5
min_valid_loss = float('inf')

criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
train_len = int(len(train) * 0.95)
sub_train_, sub_valid_ = random_split(train, [train_len, len(train) - train_len])


for epoch in range(N_EPOCHS):

    start_time = time.time()
    train_loss, train_acc = train_func(sub_train_)
    valid_loss, valid_acc = test(sub_valid_)

    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')