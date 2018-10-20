import torch
from torch.autograd import Variable
import torch.optim as optim
from multiCNN import MultiOutputCNN
from dataset import VerificationCodeDataset
import torch.nn.functional as F
from torchvision import transforms


def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    dset = VerificationCodeDataset('../data', 10, transform)
    loader = torch.utils.data.DataLoader(dset, batch_size=16, shuffle=True, num_workers=4)
    return loader


def categorical_crossentropy(pred, labels):
    ndigits = labels.size(1)
    loss = 0
    for i in range(ndigits):

        loss += F.cross_entropy(pred[i], labels[:, i].squeeze())
    return loss


def calculate_accuracy(pred, labels):

    ndigits = labels.size(1)
    count = Variable(torch.zeros(labels.size(0)).byte())
    for i in range(ndigits):
        _, p = torch.max(pred[i], 1)
        # print(p, labels[:, i])
        count += p.eq(labels[:, i].squeeze())
    print(count)
    acc = count.eq(ndigits).float().sum() / labels.size(0)
    return acc.data[0]


model = MultiOutputCNN(6, 10)
optimizer = optim.SGD(model.parameters(), lr=1e-3)

train_loader = load_data()


def train(model, optimizer, data_loader, epoch):
    model.train()
    acc = 0
    loss_ = 0
    for ib, (data, labels) in enumerate(data_loader):
        data = Variable(data)
        labels = Variable(labels)

        pred = model(data)
        optimizer.zero_grad()
        loss = categorical_crossentropy(pred, labels)
        loss.backward()

        optimizer.step()
        acc += calculate_accuracy(pred, labels)
        loss_ += loss.data[0]
    print('Epoch {}: Loss {:.4f}, Accuracy {:.4f}'.format(epoch, loss_ / ib, acc / ib))


for epoch in range(2):
    train(model, optimizer, train_loader, epoch)
