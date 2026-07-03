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
        return nn.CrossEntropyLoss(label_smoothing=0.1)(model(x), y)
    
import torch

from mimarsinan.models.nn.layers import SavedTensorDecorator

class CustomClassificationLoss:
    def __init__(self):
        self.main_loss_avg = None
        self.act_loss_avg = None

    def __call__(self, model, x, y):

        for perceptron in model.get_perceptrons():
            perceptron.activation.decorate(SavedTensorDecorator())

        model_out = model(x)
        classification_loss = nn.CrossEntropyLoss(label_smoothing=0.1)(model_out, y)

        act_losses = torch.zeros(len(model.get_perceptrons()), device=model_out.device)

        for idx, perceptron in enumerate(model.get_perceptrons()):
            saved_tensor = perceptron.activation.pop_decorator()
            flat_acts = saved_tensor.latest_output.view(-1)

            act_dist = nn.ReLU()(flat_acts - 1)
            act_losses[idx] = torch.sum(act_dist * nn.Softmax(dim=0)(act_dist))

        act_loss = torch.sum(act_losses * nn.Softmax(dim=0)(act_losses))

        classification_loss = classification_loss + act_loss

        return classification_loss
