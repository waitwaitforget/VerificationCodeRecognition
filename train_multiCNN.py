import torch
from torch.autograd import Variable
import torch.optim as optim
from multiCNN import MultiOutputCNN
from dataset import VerificationCodeDataset
import torch.nn.functional as F
from torchvision import transforms


def load_data(mode):
    transform = transforms.Compose([transforms.ToTensor()])
    dset = VerificationCodeDataset('../data/'+mode, 10, transform)
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

train_loader = load_data('train')
test_loader = load_data('test')

class AverageMeter(object):
    def __init__(self,):
        self.reset()

    def reset(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0

    def update(self, val,n=1):
        self.val = val
        self.sum += val *n
        self.count += n
        self.avg= self.sum/self.count

def train(model, optimizer, data_loader, epoch):
    model.train()
    acc = AverageMeter()
    loss_ = AverageMeter()
    for ib, (data, labels) in enumerate(data_loader):
        data = Variable(data)
        labels = Variable(labels)

        pred = model(data)
        optimizer.zero_grad()
        loss = categorical_crossentropy(pred, labels)
        loss.backward()

        optimizer.step()
        acc.update(calculate_accuracy(pred, labels),data.size(0))
        loss_.update(loss.data[0],data.size(0))
    print('Epoch {}: Loss {:.4f}, Accuracy {:.4f}'.format(epoch, loss_.avg, acc.avg))

def test(model, data_loader, epoch):
    model.eval()
    acc = 0
    loss_ = 0
    for ib,(data, labels) in enumerate(data_loader):
        data = Variable(data)
        labels = Variable(labels)
        pred = model(data)
        loss = categorical_crossentropy(pred,labels)
        acc.update(calculate_accuracy(pred, labels),data.size(0))
        loss_.update(loss.data[0],data.size(0))
    print('Test epoch {}: Loss {:.4f}, Accuracy {:.4f}'.format(epoch, loss_.avg, acc.avg)

for epoch in range(120):
    train(model, optimizer, train_loader, epoch)
    test(model, test_loader, epoch)

checkpoint = {'epoch':epoch+1,'state_dict':model.state_dict()}
torch.save(checkpoint, 'model.pth')
