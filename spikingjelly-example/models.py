"""
ANN and SNN VGG9 for CIFAR-10.
  - ANN and SNN share identical state_dict keys (Conv+BN) for weight transfer.
  - SNN uses SpikingJelly LIFNode.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from spikingjelly.activation_based import neuron, surrogate, functional

VGG9_CFG = [64, 'M', 128, 128, 'M', 256, 256, 256, 'M']


def _fwd_T(x, module):
    """Apply stateless module across time: (T,B,...) -> flatten -> module -> unflatten."""
    T, B = x.shape[:2]
    return module(x.flatten(0, 1)).unflatten(0, (T, B))


# ======================== ANN ========================

class ANN_VGG9(nn.Module):
    def __init__(self, num_classes=10, act=None):
        super().__init__()
        def A(): return act() if act else nn.ReLU(inplace=True)
        ch = 3
        feats = []
        for v in VGG9_CFG:
            if v == 'M':
                feats.append(nn.MaxPool2d(2, 2))
            else:
                feats.append(nn.Sequential(
                    nn.Conv2d(ch, v, 3, 1, 1, bias=False), nn.BatchNorm2d(v), A()))
                ch = v
        self.features = nn.Sequential(*feats)
        self.head = nn.Sequential(
            nn.Conv2d(ch, 2048, 7, bias=False), nn.BatchNorm2d(2048), A(),
            nn.Conv2d(2048, 2048, 1, bias=False), nn.BatchNorm2d(2048), A(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(2048, num_classes))
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        if x.shape[-1] < 7:
            x = F.adaptive_avg_pool2d(x, 7)
        return self.head(x)


# ======================== SNN ========================

class SNN_VGG9(nn.Module):
    """Same Conv+BN structure as ANN_VGG9. LIF replaces ReLU. MaxPool wrapped for time dim."""
    def __init__(self, num_classes=10, T=4, tau=2.0, v_threshold=1.0):
        super().__init__()
        self.T = T
        def lif():
            return neuron.LIFNode(tau=tau, v_threshold=v_threshold, detach_reset=True,
                                  surrogate_function=surrogate.ATan(), step_mode='m')
        ch = 3
        feats = []
        for v in VGG9_CFG:
            if v == 'M':
                feats.append(nn.MaxPool2d(2, 2))   # wrapped in forward
            else:
                feats.append(nn.ModuleDict({
                    'conv': nn.Sequential(nn.Conv2d(ch, v, 3, 1, 1, bias=False), nn.BatchNorm2d(v)),
                    'sn': lif()}))
                ch = v
        self.features = nn.ModuleList(feats)
        self.head_conv = nn.ModuleDict({
            'fc1': nn.Sequential(nn.Conv2d(ch, 2048, 7, bias=False), nn.BatchNorm2d(2048)),
            'sn1': lif(),
            'fc2': nn.Sequential(nn.Conv2d(2048, 2048, 1, bias=False), nn.BatchNorm2d(2048)),
            'sn2': lif()})
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        """x: (T, B, C, H, W) -> (B, num_classes)"""
        functional.reset_net(self)
        for layer in self.features:
            if isinstance(layer, nn.MaxPool2d):
                x = _fwd_T(x, layer)
            else:
                x = layer['sn'](_fwd_T(x, layer['conv']))
        T, B, C, H, W = x.shape
        if H < 7:
            x = F.adaptive_avg_pool2d(x.flatten(0, 1), 7).unflatten(0, (T, B))
        x = self.head_conv['sn1'](_fwd_T(x, self.head_conv['fc1']))
        x = self.head_conv['sn2'](_fwd_T(x, self.head_conv['fc2']))
        x = _fwd_T(x, self.pool)
        return self.classifier(x.flatten(2).mean(0))  # mean over T


# ======================== ClampFloor ========================

class ClampFloor(nn.Module):
    """Quantize activation to {0, 1/T, 2/T, ..., 1}. STE backward."""
    def __init__(self, T=4):
        super().__init__()
        self.T = T
    def forward(self, x):
        x = x.clamp(0, 1)
        q = (x * self.T).floor().clamp(max=self.T) / self.T
        return x + (q - x).detach() if self.training else q


# ======================== Weight Transfer ========================

def transfer_weights(snn: SNN_VGG9, ann_sd: dict):
    """Load Conv+BN weights from ANN into SNN. Maps ANN keys -> SNN keys.
    Returns number of keys loaded."""
    key_map = {}
    # features: ANN features.{i}.{0=conv,1=bn,2=act} -> SNN features.{i}.conv.{0,1}
    for k in ann_sd:
        if k.startswith('features.'):
            parts = k.split('.')
            idx, sub = parts[1], '.'.join(parts[2:])
            # ANN: features.{idx}.0.weight (conv), features.{idx}.1.* (bn)
            # SNN: features.{idx}.conv.0.weight, features.{idx}.conv.1.*
            if sub.startswith('0.') or sub.startswith('1.'):
                key_map[k] = f'features.{idx}.conv.{sub}'
        # head: ANN head.{0,1}=fc1+bn, {3,4}=fc2+bn, {8}=linear
        elif k.startswith('head.0.'):
            key_map[k] = k.replace('head.0.', 'head_conv.fc1.0.')
        elif k.startswith('head.1.'):
            key_map[k] = k.replace('head.1.', 'head_conv.fc1.1.')
        elif k.startswith('head.3.'):
            key_map[k] = k.replace('head.3.', 'head_conv.fc2.0.')
        elif k.startswith('head.4.'):
            key_map[k] = k.replace('head.4.', 'head_conv.fc2.1.')
        elif k.startswith('head.8.'):
            key_map[k] = k.replace('head.8.', 'classifier.')

    mapped = {key_map[k]: v for k, v in ann_sd.items() if k in key_map}
    snn.load_state_dict(mapped, strict=False)
    return len(mapped)
