import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import sys
import json
import requests

# Loading and transforming the dataset
test = datasets.MNIST("", train=False, download=True,
                      transform = transforms.Compose([transforms.ToTensor()]))

testset = torch.utils.data.DataLoader(test, batch_size=15, shuffle=True)

for data in testset:
    X, y = data

digit = X[6]
plt.imshow(digit.view(28,28))
plt.show()

# call guess service
input = json.dumps(digit.tolist())
response = requests.post("http://127.0.0.1:8888/guess", json=input).text
print(response)
