from mimarsinan.test.test_models.simple_mlp import *

import torch

def get_xor_train_data():
    train_x = torch.Tensor([
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.]])
    train_y = torch.Tensor([
        [1.,0.],
        [0.,1.],
        [0.,1.],
        [1.,0.]
        ])
    
    return (train_x, train_y)

def train_xor_model():
    device = "cpu"
    num_epochs = 500
    
    train_x, train_y = get_xor_train_data()

    loss = 1.0
    ann = None
    while(loss > 0.01):
        del ann
        ann = SimpleMLP(3,1,2,2)
        optimizer = torch.optim.Adam(ann.parameters(), lr = 0.01)
        for _ in range(num_epochs):
            ann.train()
            train_y.to(device)
            outputs = ann.forward(train_x)
            loss = nn.MSELoss()(outputs.cpu(), train_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
    return ann
