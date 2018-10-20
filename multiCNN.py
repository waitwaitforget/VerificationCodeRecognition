import torch.nn as nn


class MultiOutputCNN(nn.Module):
    def __init__(self, ndigits, nvocab):
        # inputsize 32*112
        super(MultiOutputCNN, self).__init__()

        feature_net = nn.Sequential()

        def Conv_Relu(depth, ni, no, nk):
            feature_net.add_module('layer_' + str(depth), nn.Conv2d(ni, no, kernel_size=nk))
            feature_net.add_module('bn_' + str(depth), nn.BatchNorm2d(no))
            feature_net.add_module('act' + str(depth), nn.ReLU())

        Conv_Relu(0, 3, 32, 3)  # 30 * 110 * 32
        Conv_Relu(1, 32, 32, 3)  # 28 * 108 * 32
        feature_net.add_module('max_pool1', nn.MaxPool2d(2))  # 14 * 54 * 32
        Conv_Relu(2, 32, 64, 3)  # 12 * 52 * 64
        Conv_Relu(3, 64, 64, 3)  # 10* 50 *64
        feature_net.add_module('max_pool2', nn.MaxPool2d(2))  # 5 * 25 * 64

        classifier = nn.ModuleList()
        for i in range(ndigits):
            classifier.append(nn.Sequential(nn.Linear(8000, 128), nn.ReLU(), nn.Linear(128, nvocab)))
        self.feature_net = feature_net
        self.classifier = classifier

    def forward(self, x):
        feats = self.feature_net(x)
        feats = feats.view(x.size(0), -1)

        preds = [classifier(feats) for classifier in self.classifier]
        return preds
