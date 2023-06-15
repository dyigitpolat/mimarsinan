# source: https://github.com/VITA-Group/TENAS

import numpy as np
import torch


def recal_bn(network, xloader, recalbn, device):
    for m in network.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.data.fill_(0)
            m.running_var.data.fill_(0)
            m.num_batches_tracked.data.zero_()
            m.momentum = None
    network.train()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(xloader):
            if i >= recalbn: break
            inputs = inputs.cuda(device=device, non_blocking=True)
            _, _ = network(inputs)
    return network

def get_ntk(input_data, network, device, recalbn=0, train_mode=False, num_batch=-1):
    
    ntks = []

    network.eval()
    network.to(device)

    ######
    grads = [[]]
    network.zero_grad()
    logit = network(input_data)
    if isinstance(logit, tuple):
        logit = logit[1]  # 201 networks: return features and logits

    for _idx in range(len(input_data)):
        logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]).to(device), retain_graph=True)
        grad = []
        for name, W in network.named_parameters():
            if 'weight' in name and W.grad is not None:
                grad.append(W.grad.view(-1).to(device))
        grads[0].append(torch.cat(grad, -1).to(device))
        network.zero_grad()
    ######
    grads = [torch.stack(_grads, 0) for _grads in grads]
    ntks = [torch.einsum('nc,mc->nm', [_grads, _grads]) for _grads in grads]
    conds = []
    for ntk in ntks:
        eigenvalues = torch.linalg.eigvalsh(ntk, UPLO='U') # ascending
        conds.append(np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True, nan=100000.0))
    return conds