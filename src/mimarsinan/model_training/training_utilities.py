class AccuracyTracker:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def create_hook(self, y):
        def hook(module, input, output):
            _, predicted = output.max(1)
            self.total += float(y.size(0))
            self.correct += float(predicted.eq(y).sum().item())
        
        return hook

    def get_accuracy(self):
        return self.correct / self.total

    def reset(self):
        self.correct = 0
        self.total = 0

import torch.nn as nn
class BasicClassificationLoss:
    def __call__(self, model, x, y):
        return nn.CrossEntropyLoss()(model(x), y)
    

from mimarsinan.mapping.mapping_utils import get_fused_weights
import torch
class NormalizationAwareClassificationLoss:
    def __call__(self, model, x, y):

        norm_loss = 0.0
        for perceptron in model.get_perceptrons():
            w, b = get_fused_weights(perceptron.layer, perceptron.normalization)

            max_w = torch.max(torch.abs(w))
            max_b = torch.max(torch.abs(b))

            norm_loss += (torch.abs(1.0 - max_w) + torch.abs(1.0 - max_b))

        return nn.CrossEntropyLoss()(model(x), y) + norm_loss