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
        return nn.CrossEntropyLoss()(nn.Softmax(dim=1)(model(x)), y)
    
import torch

from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer
from mimarsinan.models.layers import SavedTensorDecorator

class CustomClassificationLoss:
    def __call__(self, model, x, y):

        for perceptron in model.get_perceptrons():
            perceptron.activation.decorate(SavedTensorDecorator())

        model_out = model(x)
        classification_loss = nn.CrossEntropyLoss()(model_out, y)

        act_loss = 0.0
        for perceptron in model.get_perceptrons():
            stats = perceptron.activation.pop_decorator()
            mean = stats.latest_input.mean()
            var = stats.latest_input.var()
            act_loss += torch.dist(mean, torch.zeros_like(mean)) + torch.dist(var, torch.ones_like(var) / 2)

        # param_loss = 0.0
        # for perceptron in model.get_perceptrons():
        #     w_mean = PerceptronTransformer().get_effective_weight(perceptron).mean()
        #     w_var = PerceptronTransformer().get_effective_weight(perceptron).var()
        #     b_mean = PerceptronTransformer().get_effective_bias(perceptron).mean()
        #     b_var = PerceptronTransformer().get_effective_bias(perceptron).var()

        #     perceptron_layer = perceptron.layer
        #     p_mean = (w_mean * perceptron_layer.weight.numel() + b_mean * perceptron_layer.bias.numel()) / (perceptron_layer.weight.numel() + perceptron_layer.bias.numel())
        #     p_var = (w_var * perceptron_layer.weight.numel() + b_var * perceptron_layer.bias.numel()) / (perceptron_layer.weight.numel() + perceptron_layer.bias.numel())

        #     param_loss += torch.dist(p_mean, torch.zeros_like(p_mean)) + torch.dist(p_var, 2 * torch.ones_like(p_var))

        classification_loss *= act_loss
        # classification_loss *= param_loss

        return classification_loss
