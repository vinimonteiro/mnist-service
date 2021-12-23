import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from neural_network import NeuralNetwork
from torchvision import transforms, datasets

# Loading and transforming the dataset
train = datasets.MNIST("", train=True, download=True,
                      transform = transforms.Compose([transforms.ToTensor()]))
trainset = torch.utils.data.DataLoader(train, batch_size=15, shuffle=True)

# Initializating the neural network
model = NeuralNetwork()

# Training
optimizer = optim.Adam(model.parameters(), lr=0.001)
EPOCHS = 3
for epoch in range(EPOCHS):
    for data in trainset:
        X, y = data
        model.zero_grad()
        output = model(X.view(-1, 28 * 28))
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()

# Saving the model parameters
torch.save(model.state_dict(), "model.pth")
