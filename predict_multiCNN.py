import torch
from torch.autograd import Variable
from multiCNN import MultiOutputCNN
import os
from skimage import io
from torchvision import transforms
import sys

def predict(imgpath):
    model = MultiOutputCNN(6,10)
    if not os.path.exists('model.pth'):
        print('No model exists, please check it!')
        return 0
    state = torch.load('model.pth')
    model.load_state_dict(state['state_dict'])

    img = io.imread(imgpath)
    img = img[:32,:112,:3]
    transform = transforms.Compose(transforms.ToTensor())
    img = transform(img)
    model.cpu()
    pred = model(Variable(img))
    res = []
    for i in range(len(pred)):
        _,p = torch.max(pred[i],1)
        res.append(p)
    print(''.join(map(str,res)))

if __name__=='__main__':
    predict(sys.argv[1])
