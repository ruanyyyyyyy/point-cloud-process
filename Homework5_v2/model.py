import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import pretty_errors

class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 40)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x): # 3, 3, 10000
        # TODO: use functions in __init__ to build network
        x = self.relu(self.bn1(self.conv1(x))) # 3, 64, 10000
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x))) # 3, 128, 10000
        x = self.relu(self.bn3(self.conv3(x))) # 3, 1024, 10000

        x, _ = torch.max(x, axis=-1) # 3, 1024

        x = self.relu(self.bn4(self.fc1(x))) # 3, 512
        x = self.dropout(x)
        x = self.relu(self.bn5(self.fc2(x))) # 3, 256
        x = self.fc3(x) # 3, 40
        x = self.logsoftmax(x)
        return x


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = PointNet().to(device)
    sim_data = (torch.rand(3, 3, 10000)).to(device)
    out = net(sim_data)
    print('gfn', out.size())