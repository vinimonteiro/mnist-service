import torch
import numpy as np
from neural_network import NeuralNetwork
from flask import Flask, request
import json

model = None
app = Flask(__name__)

def load():
    global model
    model = NeuralNetwork()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

@app.route('/guess', methods=['POST'])
def get_data():
    if request.method == 'POST':
        data_string = request.get_json()
        data_array = json.loads(data_string)
        data_tensor = torch.Tensor(data_array)
        data_tensor_flattened = data_tensor.view(-1, 784)
        guess = torch.argmax(model(data_tensor_flattened)[0])
    return str(guess)

if __name__ == '__main__':
    load()
    app.run(host='127.0.0.1', port=8888)
