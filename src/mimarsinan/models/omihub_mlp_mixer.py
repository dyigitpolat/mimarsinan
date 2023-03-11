import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from mimarsinan.models.layers import *


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import warmup_scheduler
import numpy as np

import torchvision
import torchvision.transforms as transforms

import copy

q_max = 7
q_min = -8

def quantize_weight_tensor(weight_tensor):
    max_weight = torch.max(weight_tensor)
    min_weight = torch.min(weight_tensor)

    return torch.where(
        weight_tensor > 0,
        torch.round(((q_max) * (weight_tensor)) / (max_weight)) / (q_max / max_weight),
        torch.round(((q_min) * (weight_tensor)) / (min_weight)) / (q_min / min_weight))

def quantize_model(ann):
    for param in ann.parameters():
        param.data = nn.Parameter(quantize_weight_tensor(param)).data

def update_model_weights(ann, qnn):
    for param, q_param in zip(ann.parameters(), qnn.parameters()):
        q_param.data = nn.Parameter(param).data

def update_quantized_model(ann, qnn):
    update_model_weights(ann, qnn)
    quantize_model(qnn)

def transfer_gradients(a, b):
    for a_param, b_param in zip(a.parameters(), b.parameters()):
        a_param.grad = b_param.grad

class MLPMixer(nn.Module):
    def __init__(self,in_channels=3,img_size=32, patch_size=4, hidden_size=512, hidden_s=256, hidden_c=2048, num_layers=8, num_classes=10, drop_p=0., off_act=False, is_cls_token=False):
        super(MLPMixer, self).__init__()
        num_patches = img_size // patch_size * img_size // patch_size
        self.num_patches = num_patches
        # (b, c, h, w) -> (b, d, h//p, w//p) -> (b, h//p*w//p, d)
        self.is_cls_token = is_cls_token
        self.img_dim_h = img_size
        self.img_dim_w = img_size
        self.img_dim_c = in_channels
        self.num_classes = num_classes

        self.patch_emb = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size ,kernel_size=patch_size, stride=patch_size, bias=False),
            Rearrange('b d h w -> b (h w) d'), 
            nn.ReLU()
        )

        if self.is_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
            self.num_patches += 1

        self.num_layers = num_layers
        self.hidden_s = hidden_s
        self.mixer_layers = nn.Sequential(
            *[
                MixerLayer(self.num_patches, hidden_size, hidden_s, hidden_c, drop_p, off_act) 
            for _ in range(num_layers)
            ]
        )
        self.ln = nn.Identity() #nn.LayerNorm(hidden_size)

        self.clf = nn.Linear(hidden_size, num_classes, bias=False)
        self.debug = None


    def forward(self, x):
        out = self.patch_emb(x)
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0),1,1), out], dim=1)
        out = self.mixer_layers(out)
        out = self.ln(out)
        
        out = out[:, 0] if self.is_cls_token else out.mean(dim=1)

        out = self.clf(out)
        out = nn.ReLU()(out)
        self.debug = out

        return out


class MixerLayer(nn.Module):
    def __init__(self, num_patches, hidden_size, hidden_s, hidden_c, drop_p, off_act):
        super(MixerLayer, self).__init__()
        self.mlp1 = MLP1(num_patches, hidden_s, hidden_size, drop_p, off_act)
        self.mlp2 = MLP2(hidden_size, hidden_c, drop_p, off_act)

    def forward(self, x):
        out = self.mlp1(x)
        out = self.mlp2(out)
        return out

class MLP1(nn.Module):
    def __init__(self, num_patches, hidden_s, hidden_size, drop_p, off_act):
        super(MLP1, self).__init__()
        self.ln = Normalizer() #nn.LayerNorm(hidden_size)
        self.fc1 = nn.Conv1d(num_patches, hidden_s, kernel_size=1, bias=True)
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = nn.Conv1d(hidden_s, num_patches, kernel_size=1, bias=False)
        self.do2 = nn.Dropout(p=drop_p)
        self.act = nn.ReLU() #F.gelu if not off_act else lambda x:x
    def forward(self, x):
        out = self.do1(self.act(self.fc1(self.ln(x))))
        out = self.do2(self.act(self.fc2(out)))
        
        return out+x

class MLP2(nn.Module):
    def __init__(self, hidden_size, hidden_c, drop_p, off_act):
        super(MLP2, self).__init__()
        self.ln = Normalizer() #nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_c, bias=True)
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = nn.Linear(hidden_c, hidden_size, bias=False)
        self.do2 = nn.Dropout(p=drop_p)
        self.act = nn.ReLU() #F.gelu if not off_act else lambda x:x
    def forward(self, x):
        out = self.do1(self.act(self.fc1(self.ln(x))))
        out = self.do2(self.act(self.fc2(out)))
        return out+x

class Trainer(object):
    def __init__(self, model, args, num_steps = 0):
        self.device = args.device
        self.clip_grad = args.clip_grad
        self.cutmix_beta = args.cutmix_beta
        self.cutmix_prob = args.cutmix_prob
        self.model = model
        if args.optimizer=='sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        elif args.optimizer=='adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        else:
            raise ValueError(f"No such optimizer: {self.optimizer}")

        if args.scheduler=='step':
            self.base_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[args.epochs//2, 3*args.epochs//4], gamma=args.gamma)
        elif args.scheduler=='cosine':
            self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=args.min_lr)
        else:
            raise ValueError(f"No such scheduler: {self.scheduler}")


        if args.warmup_epoch:
            self.scheduler = warmup_scheduler.GradualWarmupScheduler(self.optimizer, multiplier=1., total_epoch=args.warmup_epoch, after_scheduler=self.base_scheduler)
        else:
            self.scheduler = self.base_scheduler
        self.scaler = torch.cuda.amp.GradScaler()

        self.epochs = args.epochs
        self.criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        self.num_steps = num_steps
        self.epoch_loss, self.epoch_corr, self.epoch_acc = 0., 0., 0.
    
    def _train_one_step(self, batch):
        self.model.train()
        img, label = batch
        self.num_steps += 1
        img, label = img.to(self.device), label.to(self.device)

        self.optimizer.zero_grad()
        r = np.random.rand(1)
        if self.cutmix_beta > 0 and r < self.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(self.cutmix_beta, self.cutmix_beta)
            rand_index = torch.randperm(img.size(0)).to(self.device)
            target_a = label
            target_b = label[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            img[:, :, bbx1:bbx2, bby1:bby2] = img[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
            # compute output
            with torch.cuda.amp.autocast():
                out = self.model(img)
                loss = self.criterion(out, target_a) * lam + self.criterion(out, target_b) * (1. - lam)
        else:
            # compute output
            with torch.cuda.amp.autocast():
                out = self.model(img)
                loss = self.criterion(out, label)

        self.scaler.scale(loss).backward()
        if self.clip_grad:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        acc = out.argmax(dim=-1).eq(label).sum(-1)/img.size(0)
        wandb.log({
            'loss':loss,
            'acc':acc
        }, step=self.num_steps)

    def _train_one_step_q(self, batch, qnn):
        update_quantized_model(self.model, qnn)

        qnn.train()
        self.model.train()
        img, label = batch
        self.num_steps += 1
        img, label = img.to(self.device), label.to(self.device)

        self.optimizer.zero_grad()
        r = np.random.rand(1)
        if self.cutmix_beta > 0 and r < self.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(self.cutmix_beta, self.cutmix_beta)
            rand_index = torch.randperm(img.size(0)).to(self.device)
            target_a = label
            target_b = label[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            img[:, :, bbx1:bbx2, bby1:bby2] = img[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
            # compute output
            with torch.cuda.amp.autocast():
                out = qnn(img)
                loss = self.criterion(out, target_a) * lam + self.criterion(out, target_b) * (1. - lam)
        else:
            # compute output
            with torch.cuda.amp.autocast():
                out = qnn(img)
                loss = self.criterion(out, label)

        self.scaler.scale(loss).backward()
        if self.clip_grad:
            nn.utils.clip_grad_norm_(qnn.parameters(), self.clip_grad)
        
        transfer_gradients(self.model, qnn)
        self.scaler.step(self.optimizer)
        self.scaler.update()


        acc = out.argmax(dim=-1).eq(label).sum(-1)/img.size(0)
        wandb.log({
            'loss':loss,
            'acc':acc
        }, step=self.num_steps)
        


    # @torch.no_grad
    def _test_one_step(self, batch):
        self.model.eval()
        img, label = batch
        img, label = img.to(self.device), label.to(self.device)

        with torch.no_grad():
            out = self.model(img)
            loss = self.criterion(out, label)

        self.epoch_loss += loss * img.size(0)
        self.epoch_corr += out.argmax(dim=-1).eq(label).sum(-1)


    def fit(self, train_dl, test_dl):
        for epoch in range(1, self.epochs+1):
            for batch in train_dl:
                self._train_one_step(batch)
            wandb.log({
                'epoch': epoch, 
                'lr': self.optimizer.param_groups[0]["lr"],
                }, step=self.num_steps
            )
            self.scheduler.step()

            
            num_imgs = 0.
            self.epoch_loss, self.epoch_corr, self.epoch_acc = 0., 0., 0.
            for batch in test_dl:
                self._test_one_step(batch)
                num_imgs += batch[0].size(0)
            self.epoch_loss /= num_imgs
            self.epoch_acc = self.epoch_corr / num_imgs
            wandb.log({
                'val_loss': self.epoch_loss,
                'val_acc': self.epoch_acc
                }, step=self.num_steps
            )

    def fit_q(self, train_dl, test_dl):
        qnn = copy.deepcopy(self.model)

        for epoch in range(1, self.epochs+1):
            for batch in train_dl:
                self._train_one_step_q(batch, qnn)
            wandb.log({
                'epoch': epoch, 
                'lr': self.optimizer.param_groups[0]["lr"],
                }, step=self.num_steps
            )
            self.scheduler.step()

            
            num_imgs = 0.
            self.epoch_loss, self.epoch_corr, self.epoch_acc = 0., 0., 0.
            for batch in test_dl:
                self._test_one_step(batch)
                num_imgs += batch[0].size(0)
            self.epoch_loss /= num_imgs
            self.epoch_acc = self.epoch_corr / num_imgs
            wandb.log({
                'val_loss': self.epoch_loss,
                'val_acc': self.epoch_acc
                }, step=self.num_steps
            )



def get_dataloaders(args):
    train_transform, test_transform = get_transform(args)

    if args.dataset == "c10":
        train_ds = torchvision.datasets.CIFAR10('./datasets', train=True, transform=train_transform, download=True)
        test_ds = torchvision.datasets.CIFAR10('./datasets', train=False, transform=test_transform, download=True)
        args.num_classes = 10
    elif args.dataset == "c100":
        train_ds = torchvision.datasets.CIFAR100('./datasets', train=True, transform=train_transform, download=True)
        test_ds = torchvision.datasets.CIFAR100('./datasets', train=False, transform=test_transform, download=True)
        args.num_classes = 100
    elif args.dataset == "svhn":
        train_ds = torchvision.datasets.SVHN('./datasets', split='train', transform=train_transform, download=True)
        test_ds = torchvision.datasets.SVHN('./datasets', split='test', transform=test_transform, download=True)
        args.num_classes = 10
    else:
        raise ValueError(f"No such dataset:{args.dataset}")

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return train_dl, test_dl

def get_transform(args):
    if args.dataset in ["c10", "c100", 'svhn']:
        args.padding=4
        args.size = 32
        if args.dataset=="c10":
            args.mean, args.std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        elif args.dataset=="c100":
            args.mean, args.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        elif args.dataset=="svhn":
            args.mean, args.std = [0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]
    else:
        args.padding=28
        args.size = 224
        args.mean, args.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_transform_list = [transforms.RandomCrop(size=(args.size,args.size), padding=args.padding)]
    if args.dataset!="svhn":
        train_transform_list.append(transforms.RandomCrop(size=(args.size,args.size), padding=args.padding))

    if args.autoaugment:
        if args.dataset == 'c10' or args.dataset=='c100':
            train_transform_list.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
        elif args.dataset == 'svhn':
            train_transform_list.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
        else:
            print(f"No AutoAugment for {args.dataset}")   
        

    train_transform = transforms.Compose(
        train_transform_list+[
            transforms.ToTensor(),
            transforms.Normalize(
                mean=args.mean,
                std = args.std
            )
        ]
    )
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=args.mean,
            std = args.std
        )
    ])

    return train_transform, test_transform

def get_omihub_mlp_mixer(args):
    model = None
    if args.model=='mlp_mixer':
        model = MLPMixer(
            in_channels=3,
            img_size=args.size,
            hidden_size=args.hidden_size,
            patch_size = args.patch_size,
            hidden_c = args.hidden_c,
            hidden_s = args.hidden_s,
            num_layers = args.num_layers,
            num_classes=args.num_classes,
            drop_p=args.drop_p,
            off_act=args.off_act,
            is_cls_token=args.is_cls_token
        )
    else:
        raise ValueError(f"No such model: {args.model}")

    return model.to(args.device)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class Args:
    def __init__(self):
        self.dataset = 'c100'
        self.model = 'mlp_mixer'
        self.batch_size = 128
        self.eval_batch_size = 128
        self.num_workers = 8
        self.seed = 9806
        self.epochs = 1000
        self.patch_size = 8 # choice
        self.hidden_size = 256 # choice
        self.hidden_c = 256 # choice
        self.hidden_s = 128 # choice
        self.num_layers = 16 # choice
        self.drop_p = 0.
        self.off_act = False
        self.is_cls_token = False
        self.lr = 1e-3
        self.min_lr = 1e-6
        self.momentum = 0.9
        self.optimizer = 'adam'
        self.scheduler = 'cosine'
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.weight_decay = 5e-5
        self.off_nesterov = False
        self.label_smoothing = 0.1
        self.gamma = 0.1
        self.warmup_epoch = 5
        self.autoaugment = True
        self.clip_grad = 0
        self.cutmix_beta = 1.0
        self.cutmix_prob = 0.5
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_custom_omihub_mlp_mixer(
    patch_size, hidden_size, hidden_c, hidden_s, num_layers, h, w, c, outs):
    
    print("model: ", 
        patch_size, hidden_size, hidden_c, hidden_s, num_layers, h, w, c, outs)

    return MLPMixer(
        in_channels = c,
        img_size = max(h, w),
        hidden_size = hidden_size,
        patch_size = patch_size,
        hidden_c = hidden_c,
        hidden_s = hidden_s,
        num_layers = num_layers,
        num_classes = outs,
        drop_p = 0.0,
        off_act = False,
        is_cls_token = False)

def get_omihub_mlp_mixer_search_space():
    return {
        'patch_size' : {'_type': 'choice', '_value': [
            2, 4, 8, 16, 32]},
        'hidden_size' : {'_type': 'choice', '_value': [
            16, 32, 48, 64, 96, 128, 192, 256]},
        'hidden_c' : {'_type': 'choice', '_value': [
            16, 32, 48, 64, 96, 128, 192, 256]},
        'hidden_s' : {'_type': 'choice', '_value': [
            16, 32, 48, 64, 96, 128, 192, 256]},
        'num_layers' : {'_type': 'choice', '_value': [
            2, 4, 6, 8, 12, 16, 24, 32]}
    }

if __name__=='__main__':
    args = Args()
    wandb.login()
    
    # seed exploration run
    # import random 
    # seed = random.randint(1000,10000)
    # print(seed)
    # torch.random.manual_seed(seed)
    # args.epochs = 10

    experiment_name = f"{args.seed}_{args.model}_{args.dataset}_{args.optimizer}_{args.scheduler}"
    if args.autoaugment:
        experiment_name += "_aa"
    if args.clip_grad:
        experiment_name += f"_cg{args.clip_grad}"
    if args.off_act:
        experiment_name += f"_noact"
    if args.cutmix_prob>0.:
        experiment_name += f'_cm'
    if args.is_cls_token:
        experiment_name += f"_cls"

    with wandb.init(project='mlp_mixer_1000_run', config=args, name=experiment_name):
        train_dl, test_dl = get_dataloaders(args)
        model = get_omihub_mlp_mixer(args)
        model.load_state_dict(torch.load("./saved_models/model.state_dict"))
        trainer = Trainer(model, args)
        trainer.fit(train_dl, test_dl)

        torch.save(model.state_dict(), f"./saved_models/{experiment_name}")
