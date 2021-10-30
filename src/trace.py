import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        './data', train=True, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ])),  batch_size=64, shuffle=True)


network = Net()
optimizer = torch.optim.SGD(network.parameters(), lr=0.1,
                            momentum=0.9)
network.cuda()
network.train()
log_interval = 10
n_epochs = 50
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
def train(epoch):
    for data, target in tqdm(train_loader, leave=False):
        optimizer.zero_grad()
        target = target.cuda()
        output = network(data.cuda())
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % log_interval == 0:
        #     # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #     #     epoch, batch_idx * len(data), len(train_loader.dataset),
        #     #     100. * batch_idx / len(train_loader), loss.item()))
        #     train_losses.append(loss.item())
        #     train_counter.append(
        #         (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
        #     torch.save(network.state_dict(), './model.pth')
        #     torch.save(optimizer.state_dict(), './optimizer.pth')


for epoch in range(0,50):
    train(epoch)

network.eval()
trace = torch.jit.trace(network, torch.randn(64,1,28,28).cuda())
torch.jit.save(trace, 'model.pth')