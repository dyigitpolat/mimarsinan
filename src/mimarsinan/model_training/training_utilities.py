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

from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer
from mimarsinan.models.layers import SavedTensorDecorator

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
        # for perceptron in model.get_perceptrons():
        #     stats = perceptron.activation.pop_decorator()
        #     mean = stats.latest_input.mean()
        #     var = stats.latest_input.var()
        #     # act_loss += torch.dist(mean, torch.zeros_like(mean)) + torch.dist(var, torch.ones_like(var) / 2)
        #     act_loss += torch.dist(var, torch.ones_like(var) / 2)

        for idx, perceptron in enumerate(model.get_perceptrons()):
            saved_tensor = perceptron.activation.pop_decorator()
            flat_acts = saved_tensor.latest_output.view(-1)  # flatten to 1D
            # sorted_acts, _ = torch.sort(flat_acts)  # sort ascending
            # cumsum_acts = torch.cumsum(sorted_acts, dim=0)  # cumulative sum
            # norm_cumsum = cumsum_acts / cumsum_acts[-1]  # normalize by total sum
            # threshold_idx = torch.searchsorted(norm_cumsum, 0.99)  # index of first value >= 0.99
            # threshold = sorted_acts[threshold_idx].cpu()  # value at that index

            act_dist = nn.ReLU()(flat_acts - 1)
            act_losses[idx] = torch.sum(act_dist * nn.Softmax(dim=0)(act_dist))

        act_loss = torch.sum(act_losses * nn.Softmax(dim=0)(act_losses))


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

        # if self.main_loss_avg is None or self.act_loss_avg is None:
        #     self.main_loss_avg = classification_loss.item()
        #     self.act_loss_avg = act_loss.item()

        # self.main_loss_avg = 0.9 * self.main_loss_avg + 0.1 * classification_loss.item()
        # self.act_loss_avg = 0.9 * self.act_loss_avg + 0.1 * act_loss.item()

        # normalized_main_loss = 2 * (classification_loss / self.main_loss_avg)
        # normalized_act_loss = act_loss / self.act_loss_avg

        classification_loss = classification_loss + act_loss
        # classification_loss *= param_loss

        return classification_loss
