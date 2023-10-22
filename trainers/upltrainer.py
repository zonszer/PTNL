from random import sample
from dassl.engine import TRAINER_REGISTRY, TrainerX
import os.path as osp
import os
import time
import datetime
import numpy as np
from tqdm import tqdm
import json
from dassl.utils import read_json, write_json

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data import DataManager
from copy import deepcopy

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
    
from datasets.data_manager import UPLDataManager, ElevaterDataManager
from evaluation.evaluator import UPLClassification
from .hhzsclip import ZeroshotCLIP
from .utils import (select_top_k_similarity_per_class, caculate_noise_rate, save_outputs,
select_top_k_similarity, select_top_by_value, caculate_noise_rate_analyze, 
select_top_k_similarity_per_class_with_noisy_label,
add_partial_labels, generate_uniform_cv_candidate_labels,
select_top_k_certainty_per_class,
)

from utils_temp.utils_ import dict_add, get_regular_weight
_tokenizer = _Tokenizer()
from trainers.loss import GeneralizedCrossEntropy, PLL_loss


CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    # semi-supervised templates
    "SSOxfordPets": "a photo of a {}, a type of pet.",
    "SSOxfordFlowers": "a photo of a {}, a type of flower.",
    "SSFGVCAircraft": "a photo of a {}, a type of aircraft.",
    "SSDescribableTextures": "{} texture.",
    "SSEuroSAT": "a centered satellite photo of {}.",
    "SSStanfordCars": "a photo of a {}.",
    "SSFood101": "a photo of {}, a type of food.",
    "SSSUN397": "a photo of a {}.",
    "SSCaltech101": "a photo of a {}.",
    "SSUCF101": "a photo of a person doing {}.",
    "SSImageNet": "a photo of a {}.",
    # ElevaterDatasets:
    'cifar-100': "a photo of a {}.",
}

OriginDatasets = {
    "SSImageNet":"imagenet",
    "SSCaltech101":"caltech-101",
    "SSOxfordPets":"oxford_pets",
    "SSUCF101":"ucf101",
    "SSOxfordFlowers":"oxford_flowers",
    "SSStanfordCars":"stanford_cars",
    "SSFGVCAircraft":"fgvc_aircraft",
    "SSDescribableTextures":"dtd",
    "SSEuroSAT":"eurosat",
    "SSFood101":"food-101",
    "SSSUN397":"sun397",          
}

ElevaterDatasets = {
# 'hateful-memes',  'patch-camelyon', 'kitti-distance', 'fgvc-aircraft-2013b-variants102', 
# 'fer-2013', 'caltech-101', 'eurosat_clip', 'oxford-iiit-pets', 'resisc45_clip', 'oxford-flower-102', 'food-101', 
# 'voc-2007-classification', 'rendered-sst2', 'country211', 'mnist', 'gtsrb', 'dtd', 'stanford-cars',
'cifar-10': 'cifar_10_20211007',
'cifar-100': 'cifar100_20200721',
}

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)       #x.shape=torch.Size([100, 77, 512])

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection    #proj.shape=torch.Size([512, 1024])

        return x    #shape=torch.Size([100, 1024])


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.UPLTrainer.N_CTX
        ctx_init = cfg.TRAINER.UPLTrainer.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.UPLTrainer.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            # nn.init.zeros_(ctx_vectors)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS    #上面prompt_prefix的作用只是用来创建相应的token_prefix和token_suffix
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS and EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.UPLTrainer.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.clip = clip_model
        self.classnames = classnames
        self.cfg = cfg

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype)) #torch.Size([32, 1024])

        prompts = self.prompt_learner()      # compose prompt according to the position
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)   #torch.Size([100, 1024])

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        if self.cfg.TRAINER.PLL.USE_REGULAR and self.training:
            if hasattr(self, 'text_regular_feat'):
                self.regular = logit_scale * self.text_regular_feat @ text_features.t()
            else:
                temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
                prompts = [temp.format(c.replace("_", " ")) for c in self.classnames]
                prompts = torch.cat([clip.tokenize(p) for p in prompts])
                prompts = prompts.to(logits.device)    #shape=torch.Size([100, 77])
                with torch.no_grad():
                    text_features_fixed = self.clip.encode_text(prompts)
                    text_features_fixed = text_features_fixed / text_features_fixed.norm(dim=-1, keepdim=True)
                self.text_regular_feat = text_features_fixed        
                self.regular_label = torch.arange(0, len(self.classnames)).to(logits.device)
                # self.regular_dict = {self.regular_label[i]: self.text_regular_feat[i] for i in range(len(self.classnames))}
                self.regular = logit_scale * self.text_regular_feat @ text_features.t()         #TODO 这里搞一个random sample self.text_regular_feat等价于 batch

        return logits, image_features, text_features

    def zero_shot_forward(self, image, device):
        temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in self.classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(device)    #shape=torch.Size([100, 77])
        with torch.no_grad():
            text_features = self.clip.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features = self.clip.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits, image_features, text_features





@TRAINER_REGISTRY.register()
class UPLTrainer(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)
        # self.GCE_loss = GeneralizedCrossEntropy(q=0.5)
        self.gt_label_dict = self.get_gt_label(cfg)

    def check_cfg(self, cfg):
        assert cfg.TRAINER.UPLTrainer.PREC in ["fp16", "fp32", "amp"]

    def build_loss(self):
        if self.cfg.TRAINER.LOSS_TYPE == 'CE':
            criterion = torch.nn.CrossEntropyLoss()
            criterion.cfg = self.cfg.TRAINER.PLL
        else:
            if hasattr(self, 'partialY') and self.partialY != None:
                criterion = PLL_loss(type=self.cfg.TRAINER.LOSS_TYPE, cfg=self.cfg.TRAINER.PLL,
                                     PartialY=deepcopy(self.partialY))
            else:
                criterion = PLL_loss(type=self.cfg.TRAINER.LOSS_TYPE, cfg=self.cfg.TRAINER.PLL)
        self.criterion = criterion

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.UPLTrainer.PREC == "fp32" or cfg.TRAINER.UPLTrainer.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.UPLTrainer.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def get_gt_label_(self, cfg):
        if self.cfg.DATASET.NAME in ElevaterDatasets.keys():
            dataset_map = ElevaterDatasets
            dataset_dir = dataset_map[self.cfg.DATASET.NAME]
            root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
            dataset_dir = os.path.join(root, 'classification', dataset_dir)
            gt_labels = os.path.join(dataset_dir, "train_images.txt")
        elif self.cfg.DATASET.NAME in OriginDatasets.keys():
            dataset_map = OriginDatasets
            dataset_dir = dataset_map[self.cfg.DATASET.NAME]
            root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
            dataset_dir = os.path.join(root, dataset_dir)
            gt_labels = os.path.join(dataset_dir, "{}_GTlabels.json".format(self.cfg.DATASET.NAME))
        else:
            raise ValueError
        
        gt_label_dict = {}
        if os.path.exists(gt_labels):
            with open(gt_labels, "r") as f:         #TOorg 读json和 txt文件的不同代码使用（txt必须用 'r' mode while json can be 'r' or 'rb')
                if gt_labels.endswith('.txt'):
                    for i, line in enumerate(f):
                        img_path, value = line.strip().split()
                        gt_label_dict[img_path] = int(value)
                else:
                    gt_label_dict = json.load(f)
            print("Loaded training GT labels from {}".format(gt_labels))
        else:
            print("Generating training GT labels to {}".format(gt_labels))
            gt_label_dict = {}
            for batch_idx, batch in enumerate(self.train_loader_x):
                input, label, impath = self.parse_batch_test_with_impath(batch)
                for l, ip in zip(label, impath):
                    ip = './data/' + ip.split('/data/')[1]
                    gt_label_dict[ip] = l.item()
            with open(gt_labels, "w") as outfile:
                json.dump(gt_label_dict, outfile)
        return gt_label_dict
    
    def get_gt_label(self, cfg):
        gt_label_dict = {}
        if self.cfg.DATASET.NAME in OriginDatasets.keys():
            dataset_map = OriginDatasets
            dataset_dir = dataset_map[self.cfg.DATASET.NAME]
            root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
            dataset_dir = os.path.join(root, dataset_dir)
            gt_labels = os.path.join(dataset_dir, "{}_GTlabels.json".format(self.cfg.DATASET.NAME))

            if os.path.exists(gt_labels):
                with open(gt_labels, "r") as f:         
                    gt_label_dict = json.load(f)
                print("Loaded training GT labels from {}".format(gt_labels))
            else:
                print("Generating training GT labels to {}".format(gt_labels))
                gt_label_dict = {}
                for batch_idx, batch in enumerate(self.train_loader_x):
                    input, label, impath = self.parse_batch_test_with_impath(batch)
                    for l, ip in zip(label, impath):
                        ip = './data/' + ip.split('/data/')[1]
                        gt_label_dict[ip] = l.item()
                with open(gt_labels, "w") as outfile:
                    json.dump(gt_label_dict, outfile)
        elif self.cfg.DATASET.NAME in ElevaterDatasets.keys():
            pass
        return gt_label_dict

    def _get_gt_label(self, impath, dtype):
        gt_label_list = []
        if '/' in impath[0]:
            for ip in impath:
                ip = './data/' + ip.split('/data/')[1]
                gt_label = self.gt_label_dict[ip]
                gt_label_list.append(gt_label)
        else:
            for idx_str in impath:
                gt_label_list.append(self.dm.gt_labels_sstrain[idx_str])
        gt_label = torch.tensor(gt_label_list, dtype=dtype).to(self.device)
        return gt_label
    
    def forward_backward(self, batch):
        _, _, impath = self.parse_batch_test_with_impath(batch)
        image, label, index = self.parse_batch_train(batch)     #When PLL: labels_true == self.labels_true[index]
        gt_label = self._get_gt_label(impath, dtype=label.dtype)
        prec = self.cfg.TRAINER.UPLTrainer.PREC
        if prec == "amp":
            with autocast():
                output, image_features, text_features = self.model(image)
                # loss = F.cross_entropy(output, label, self.class_weights)
                # loss = F.cross_entropy(output, label)
                loss = self.criterion(output, label, index)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            notuse_mask = torch.zeros_like(index, dtype=torch.bool).to(self.device)
            halfuse_mask = torch.zeros_like(index, dtype=torch.bool).to(self.device)
            if self.criterion.losstype == 'rc_refine' and hasattr(self, 'feat_idxs_unsafe'):
                for i, idx in enumerate(index):
                    notuse_mask[i] = self.feat_idxs_unsafe.get(idx.item(), False)
                    halfuse_mask[i] = self.feat_idxs_halfsafe.get(idx.item(), False)

            weight = (~halfuse_mask) * (~notuse_mask) + self.cfg.TRAINER.PLL.HALF_USE_W * halfuse_mask       #full use examples weight are 1.0 and half use is 0.5, not use is 0.0
            output, image_features, text_features = self.model(image)             # 0.5 is weight for half use examples
            # loss = self.GCE_loss(output, label)
            loss = self.criterion(output, label, index, reduce=False)
            loss = (loss * weight).mean()

            if self.cfg.TRAINER.PLL.USE_REGULAR:
                loss_regular = F.cross_entropy(self.model.regular, self.model.regular_label, )
                if hasattr(self, 'pool_unc_norm'):
                    loss = loss + get_regular_weight(beta_median=self.cfg.TRAINER.PLL.BETA,
                                                        class_acc=self.pool_certn_norm,
                                                    ) * loss_regular

            self.model_backward_and_update(loss)
            # if hasattr(self.criterion, 'check_update'):
                # update_dict = dict(zip(index.tolist(), gt_label.cpu().long().tolist()))
                # self.criterion.gt_label_dict.update(update_dict)
                # self.criterion.check_conf_update(image, label, index)   

        # gradients compare:
        # if self.criterion.num % 2 == 0:
        #     grad_ratios_dict = self.compare_gradients(image, index, label, gt_label)
        #     for key, value in grad_ratios_dict.items():
        #         # assert len(grad_ratios_dict[key]) == 1
        #         if not hasattr(self, 'grad_ratios_dict'):
        #             self.grad_ratios_dict = {}
        #         for i in range(len(grad_ratios_dict[key])):
        #             dict_add(self.grad_ratios_dict, key, value[i])

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, gt_label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def compare_gradients(self, image, index, label, gt_label):
        def seprate_partial_labels(label, gt_label):
            gt_label = gt_label.long()
            gt_label_onehot = torch.zeros_like(label)
            other_label = deepcopy(label)
            for i in range(label.shape[0]): #label is shape of (bs, classes) on-hot metrix
                gt_label_onehot[i, gt_label[i]] = 1.
                other_label[i, gt_label[i]] = 0.
            return gt_label_onehot, other_label
        
        gt_label_onehot, other_label = seprate_partial_labels(label, gt_label)
        ratios = {}

        # 1. Calculate the forward pass for the gt_labels
        # HACK this func assert loss_type == 'cc'
        output, image_features, text_features = self.model(image)
        if hasattr(self.criterion, 'confidence'):
            conf_tmp = deepcopy(self.criterion.conf)
        loss = self.criterion(output, label, index)
        if hasattr(self.criterion, 'confidence'):
            self.criterion.conf = conf_tmp

        if not torch.isfinite(loss).all():
            return ratios
        # Zero any existing gradients first
        self.model_zero_grad()
        self.model_backward(loss)
        names = self.get_model_names()  

        grad_gt = {}
        for name in names:
            # Fetch the gradients of the parameters
            grad_gt[name] = [param.grad.clone() for param in self._models[name].parameters()]
        
        # Reset gradients in the model parameters
        self.model_zero_grad()

        # 2. Repeat the process for non-gt labels and calculate the ratio
        output, i_features, t_features = self.model(image)
        if hasattr(self.criterion, 'confidence'):
            conf_tmp = deepcopy(self.criterion.conf)
        loss = self.criterion(output, other_label, index)
        if hasattr(self.criterion, 'confidence'):
            self.criterion.conf = conf_tmp
        
        if not torch.isfinite(loss).all():
            return ratios
        # Zero any existing gradients first
        self.model_backward(loss)
        names = self.get_model_names()  

        grad_non_gt = {}
        for name in names:
            # Fetch the gradients of the parameters
            grad_non_gt[name] = [param.grad.clone() for param in self._models[name].parameters()]
        
        # Reset gradients in the model parameters
        self.model_zero_grad()
        
        for name in names:
            ratio_list = []
            for grad_gt_param, grad_non_gt_param in zip(grad_gt[name], grad_non_gt[name]):
                if grad_gt_param is not None and grad_non_gt_param is not None:
                    ratio = torch.norm(grad_non_gt_param) / torch.norm(grad_gt_param)
                    ratio_list.append(ratio.item())
            ratios[name] = ratio_list
        return ratios


    def parse_batch_train(self, batch):
        if isinstance(batch, dict):
            input = batch["img"]
            label = batch["label"]
            index = batch["index"]
        elif isinstance(batch, list):
            input = batch[0]
            label = batch[1]
            index = [eval(i) for i in batch[2]]

        input = input.to(self.device)
        label = label.to(self.device)
        return input, label, index
    
    def parse_batch_train_with_impath(self, batch):
        if isinstance(batch, dict):
            input = batch["img"]
            label = batch["label"]
            index = batch["index"]
            impath = batch["impath"]
        elif isinstance(batch, list):
            input = batch[0]
            label = batch[1]
            index = [eval(i) for i in batch[2]]
            impath = None

        input = input.to(self.device)
        label = label.to(self.device)
        return input, label, index, impath
    
    def parse_batch_test(self, batch):
        if isinstance(batch, dict):
            input = batch["img"]
            label = batch["label"]
        elif isinstance(batch, list):
            input = batch[0]
            label = batch[1]

        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def parse_batch_test_with_impath(self, batch):
        impath, idx = None, None
        if isinstance(batch, dict):
            input = batch["img"]
            label = batch["label"]
            impath = batch["impath"]
        elif isinstance(batch, list):
            input = batch[0]
            label = batch[1]
            idx = batch[2]

        input = input.to(self.device)
        label = label.to(self.device)
        return input, label, impath or idx
    

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"   #or model-last.pth.tar

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))
            try:
                checkpoint = load_checkpoint(model_path)
            except:
                checkpoint = load_checkpoint(model_path.replace('best', 'last'))

            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def load_model_by_id(self, directory, model_id, epoch=None):
        if not directory:
            print(
                'Note that load_model() is skipped as no pretrained model is given'
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = 'model-best-{}.pth.tar'.format(model_id)

        if epoch is not None:
            model_file = 'model.pth.tar-' + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']

            # Ignore fixed token vectors
            if 'token_prefix' in state_dict:
                del state_dict['token_prefix']

            if 'token_suffix' in state_dict:
                del state_dict['token_suffix']

            print(
                'Loading weights to {} '
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def test(self, split=None, trainer_list=None):                      
        """A generic testing pipeline."""

        self.set_model_mode("eval")
        self.evaluator.reset()
        save_path = os.path.join(self.cfg.TEST.Analyze_Result_Path, self.cfg.DATASET.NAME, 'fp'+str(self.cfg.TRAINER.UPLTrainer.NUM_FP),
        str(self.cfg.OPTIM.MAX_EPOCH)+'_'+str(self.cfg.SEED)+'_'+str(self.cfg.DATASET.NUM_SHOTS)+'_random_init'+str(self.cfg.TRAINER.UPLTrainer.CLASS_TOKEN_POSITION))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        results_id = 0
        while os.path.exists(os.path.join(save_path, 'per_image_results_{}_{}.txt'.format(split, results_id))):     #TOorg: how to save a result file without overwritten (also used in dassl)
            results_id += 1 
        self.per_image_txt_writer = open(os.path.join(save_path, 'per_image_results_{}_{}.txt'.format(split, results_id)), 'w') if split=='test' else None
        self.per_class_txt_writer = open(os.path.join(save_path, 'per_class_results_{}_{}.txt'.format(split, results_id)), 'w') if split=='test' else None

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
            print("Do evaluation on {} set".format(split))
        elif split=="novel":
            data_loader = self.test_novel_loader
            print("Do evaluation on test novel set")
        elif split=="base":
            data_loader = self.test_base_loader
            print("Do evaluation on test base set")
        elif split=="all":
            data_loader = self.test_loader
            print("Do evaluation on test set")
        else:
            data_loader = self.test_loader
            print("Do evaluation on test set")

        outputs_all = []
        label_all = []
        image_features_all = []
        text_features_all = []
        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            if trainer_list is None or len(trainer_list)==1:
                output, image_features, text_features = self.model_inference(input)
                image_features_all.append(image_features)
                text_features_all.append(text_features)
            else:
                # ensemble
                outputs = [t.model_inference(input)[0] for t in trainer_list]
                output = sum(outputs) / len(outputs)
            self.evaluator.process(output, label, self.per_image_txt_writer, self.per_class_txt_writer)
            outputs_all.append(output)
            label_all.append(label)
        results = self.evaluator.evaluate()
        #0. save loged logits and conf: 
        # if True or log_conf == True:
            # self.criterion.log_conf(all_logits=torch.cat(outputs_all, dim=0), all_labels=torch.cat(label_all, dim=0))
        if False and split == 'test':
            #1. save class_acc_sumlist and evalset_acc_sumlist:        #NOTE before uncomment remember to changed the name, otherwise the original file will be overwritten
            filename = f'analyze_result_temp/class_acc_sumlist/{self.cfg.DATASET.NAME}-{self.cfg.DATASET.NUM_SHOTS}-{self.cfg.TRAINER.UPLTrainer.NUM_FP}-{self.cfg.SEED}-PLL{self.cfg.TRAINER.PLL.PARTIAL_RATE}_{self.cfg.TRAINER.LOSS_TYPE}_beta{self.cfg.TRAINER.PLL.BETA}.json'
            with open(filename, "w") as file:
                json.dump(self.evaluator.class_acc_sumlist, file)
            filename = f'analyze_result_temp/evalset_acc_sumlist/{self.cfg.DATASET.NAME}-{self.cfg.DATASET.NUM_SHOTS}-{self.cfg.TRAINER.UPLTrainer.NUM_FP}-{self.cfg.SEED}-PLL{self.cfg.TRAINER.PLL.PARTIAL_RATE}_{self.cfg.TRAINER.LOSS_TYPE}_beta{self.cfg.TRAINER.PLL.BETA}.json'
            with open(filename, "w") as file:
                json.dump(self.evaluator.evalset_acc_sumlist, file)
        #     #2. save grad_ratios_dict:
        #     filename = f'analyze_result_temp/grad_ratios_dict/{self.cfg.DATASET.NAME}-{self.cfg.DATASET.NUM_SHOTS}-{self.cfg.TRAINER.UPLTrainer.NUM_FP}-{self.cfg.SEED}-PLL{self.criterion.cfg.PARTIAL_RATE}_{self.criterion.losstype}.json'
        #     with open(filename, "w") as file:
        #         json.dump(self.grad_ratios_dict, file)
        # if split in ['all', 'train', 'test', 'novel', 'base']:
        #     if len(outputs_all) != 0:
        #         outputs_all = torch.cat(outputs_all, dim=0)
        #         label_all = torch.cat(label_all, dim=0)
        #         image_features_all = torch.cat(image_features_all, dim=0)
        #         text_features_all = text_features_all[0]
        #         torch.save(image_features_all, os.path.join(save_path, '{}_v_features.pt'.format(split)))
        #         torch.save(image_features_all, os.path.join(save_path, '{}_targets.pt'.format(split)))
        #         torch.save(outputs_all, os.path.join(save_path, '{}_logits.pt'.format(split)))
        #         torch.save(text_features_all, os.path.join(save_path, '{}_l_features.pt'.format(split)))
        #         torch.save(label_all, os.path.join(save_path, '{}_labels.pt'.format(split)))

        self.per_image_txt_writer.close() if self.per_image_txt_writer != None else None
        self.per_class_txt_writer.close() if self.per_class_txt_writer != None else None

        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    @torch.no_grad()
    def zero_shot_analyze(self, trainer_list=None):
        """A generic predicting pipeline."""
        self.set_model_mode("eval")
        self.model.eval()
        self.evaluator.reset()

        # outputs_all = []
        # label_all = []
        # image_features_all = []
        # text_features_all = []
        # for batch_idx, batch in enumerate(self.test_loader):
        #     input, label = self.parse_batch_test(batch)
        #     if trainer_list is None or len(trainer_list)==1:
        #         output, image_features, text_features = self.model_inference(input)
        #         image_features_all.append(image_features)
        #         text_features_all.append(text_features)
        #     else:
        #         # ensemble
        #         outputs = [t.model_inference(input)[0] for t in trainer_list]
        #         output = sum(outputs) / len(outputs)
        #     self.evaluator.process(output, label)
        #     outputs_all.append(output)
        #     label_all.append(label)
        # results = self.evaluator.evaluate()

        data_loader = self.train_loader_sstrain
        # data_loader = self.test_loader
        outputs = []
        image_features_list = []
        img_paths = []
        from tqdm import tqdm
        for batch_idx, batch in tqdm(enumerate(data_loader)):
            input, label, impath = self.parse_batch_test_with_impath(batch)
            if trainer_list is None or len(trainer_list)==1:    #output is model logits
                output, image_features, text_features = self.model.zero_shot_forward(input, self.device)
            else:
                # ensemble
                outputs = [t.model.zero_shot_forward(input, self.device)[0] for t in trainer_list]
                output = sum(outputs) / len(outputs)
            outputs.append(output)
            image_features_list.append(image_features)
            img_paths.append(impath)
        sstrain_outputs = torch.cat(outputs, dim=0)         #torch.Size([4128, 100])
        sstrain_img_paths = np.concatenate(img_paths, axis=0)   #(4128,)
        image_features = torch.cat(image_features_list, axis=0) #torch.Size([4128, 1024])
        # text_features = torch.cat(text_features, axis=0)
        print('image_features', image_features.shape)
        print('text_features', text_features.shape)             #↓ outputs, img_paths, K=1, image_features=None, is_softmax=True
        predict_label_dict, _ = select_top_k_similarity_per_class(sstrain_outputs, sstrain_img_paths, -1, image_features, True)     #选择每个类别visual emb和text emb最相似的K个样本，对每个样本取预测的vector，最后加到info dict中（k>=0时）。 对每个样本取预测的vector，然后加到所有训练样本的info dict中（k=-1时）
        if data_loader is self.train_loader_sstrain:
            save_outputs(self.train_loader_x, self, predict_label_dict, self.cfg.DATASET.NAME, text_features, backbone_name=self.cfg.MODEL.BACKBONE.NAME)
        # caculate_noise_rate_analyze(predict_label_dict, train_loader=self.test_loader, trainer=self)
        caculate_noise_rate_analyze(predict_label_dict, train_loader=self.train_loader_x, trainer=self)
        return predict_label_dict


    def load_from_exist_file(self, file_path, model_names):
        '''load logits and label from saved PSEUDO_LABEL_MODELS'''
        logits, partialY = None, None

        if isinstance(self.dm, ElevaterDataManager):
            Datums = self.train_loader_sstrain.dataset.dataset.dataset.dataset_manifest.images      #len(Datums) = 3200
            convert_datum_to_dict = lambda Datums: {str(i): datum.labels[0] for i, datum in enumerate(Datums)}
            predict_label_dict = convert_datum_to_dict(Datums)  #predict_label_dict -> k: idxs in the train+val dataset, v: label  
           
            if self.cfg.TRAINER.PLL.USE_PLL:                 #HACK only suppprot PLL now:
                predict_label_dict, partialY, labels_true = add_partial_labels(label_dict=predict_label_dict,
                                                        partial_rate=self.cfg.TRAINER.PLL.PARTIAL_RATE)
        else:
            for model in model_names:
                model_path = os.path.join(file_path, model)
                logist_path = os.path.join(model_path, '{}_logits.pt'.format(self.cfg.DATASET.NAME))    #sequential order
                if logits is None:
                    logits = torch.load(logist_path)
                else:
                    logits += torch.load(logist_path)

                info_path = os.path.join(model_path, '{}.json'.format(self.cfg.DATASET.NAME))       #sequential order
                info = json.load(open(info_path))       
                items = []
                for label in info:
                    for img_path in info[label]:
                        item = info[label][img_path]
                        items.append([img_path, int(item[3])]) # 路径 pred_label    #sequential order
                _ = sorted(items, key=(lambda x:x[1]))
                sstrain_img_paths = np.array(items)[:,0]        #shape is (4128,), why not directly use existing loaded dataset?

            logits /= len(model_names)          #NOTE extract Pseudo-labels here 
            predict_label_dict = select_top_k_similarity_per_class_with_noisy_label(img_paths=sstrain_img_paths,
                                                                                    K=self.cfg.DATASET.NUM_SHOTS,
                                                                                    random_seed=self.cfg.SEED, 
                                                                                    gt_label_dict=self.gt_label_dict,
                                                                                    num_fp=self.cfg.TRAINER.UPLTrainer.NUM_FP)
            if self.cfg.TRAINER.PLL.USE_PLL:
                predict_label_dict, partialY, labels_true = add_partial_labels(label_dict=predict_label_dict,
                                                        partial_rate=self.cfg.TRAINER.PLL.PARTIAL_RATE)

            if self.cfg.TRAINER.PLL.USE_LABEL_FILTER:
                partialY_ = []; T = 1                       #TODO change T and do Label Smoothing
                for i, (ip, label_pl) in enumerate(predict_label_dict.items()):
                    labels_dict = info[str(labels_true[i].item())]
                    sequential_idx = labels_dict[ip][0]             #assert ip in labels_dict.keys()
                    zeroshot_label_plb = F.softmax(logits[sequential_idx].float() / T, dim=-1) * label_pl
                    base_value = zeroshot_label_plb.sum(dim=0)           #base_value can use for sth
                    zeroshot_label_plb = zeroshot_label_plb/base_value    

                    #assign attributes:
                    predict_label_dict[ip] = zeroshot_label_plb
                for k, v in predict_label_dict.items():
                    partialY_.append(v.unsqueeze(0))
                partialY = torch.cat(partialY_, dim=0)

        # Attributes:
        self.partialY = partialY
        return predict_label_dict 

    @torch.no_grad()
    def zero_shot_predict(self, trainer_list=None):
        """A generic predicting pipeline."""
        self.set_model_mode("eval")
        self.model.eval()
        self.evaluator.reset()

        save_path = os.path.join(self.cfg.TEST.Analyze_Result_Path, self.cfg.DATASET.NAME,
        str(self.cfg.OPTIM.MAX_EPOCH)+'_'+str(self.cfg.SEED)+'_'+str(self.cfg.DATASET.NUM_SHOTS))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        data_loader = self.train_loader_sstrain

        outputs = []
        img_paths = []


        for batch_idx, batch in tqdm(enumerate(data_loader)):
            input, label, impath = self.parse_batch_test_with_impath(batch)
            if trainer_list is None or len(trainer_list)==1:
                output, image_features, text_features = self.model.zero_shot_forward(input, self.device)
            else:
                # ensemble
                outputs = [t.model.zero_shot_forward(input, self.device)[0] for t in trainer_list]
                output = sum(outputs) / len(outputs)
            outputs.append(output)
            img_paths.append(impath)


        outputs = torch.cat(outputs, dim=0)
        img_paths = np.concatenate(img_paths, axis=0)


        if self.cfg.DATASET.CLASS_EQULE is True:
            if self.cfg.DATASET.CONF_THRESHOLD > 0:
                predict_label_dict_1, predict_conf_dict_1 = select_top_k_similarity_per_class(outputs, img_paths, K=self.cfg.DATASET.NUM_SHOTS)
                predict_label_dict_2, predict_conf_dict_2 = select_top_by_value(outputs, img_paths, conf_threshold=self.cfg.DATASET.CONF_THRESHOLD)

                print(len(predict_label_dict_1), 'predict_label_dict_1')
                print(len(predict_label_dict_2), 'predict_label_dict_2')

                predict_label_dict = dict(predict_label_dict_1, **predict_label_dict_2)
                predict_conf_dict = dict(predict_conf_dict_1, **predict_conf_dict_2)
                caculate_noise_rate(predict_label_dict, train_loader=self.train_loader_x, trainer=self)
                print('select {} samples'.format(len(predict_label_dict)))

            else:
                print("K {} shots".format(self.cfg.DATASET.NUM_SHOTS))
                predict_label_dict, predict_conf_dict = select_top_k_similarity_per_class(outputs, img_paths, K=self.cfg.DATASET.NUM_SHOTS)
                caculate_noise_rate(predict_label_dict,  train_loader=self.train_loader_x, trainer=self)
                print('select {} samples'.format(len(predict_label_dict)))

        else:
            print("K", self.cfg.DATASET.NUM_SHOTS*text_features.shape[0])
            predict_label_dict, predict_conf_dict = select_top_k_similarity(outputs, img_paths, K=self.cfg.DATASET.NUM_SHOTS*text_features.shape[0])
            caculate_noise_rate(predict_label_dict, train_loader=self.train_loader_x, trainer=self)
            print('select {} samples'.format(len(predict_label_dict)))
        return predict_label_dict, predict_conf_dict

    @torch.no_grad()
    def zero_shot_test(self, split=None, trainer_list=None):
        """A generic predicting pipeline."""

        self.set_model_mode("eval")
        self.evaluator.reset()

        save_path = os.path.join(self.cfg.TEST.Analyze_Result_Path, self.cfg.DATASET.NAME,
        str(self.cfg.OPTIM.MAX_EPOCH)+'_'+str(self.cfg.SEED)+'_'+str(self.cfg.DATASET.NUM_SHOTS))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        results_id = 0
        while os.path.exists(os.path.join(save_path, 'per_image_results_{}_{}.txt'.format(split, results_id))):
            results_id += 1
        self.per_image_txt_writer = open(os.path.join(save_path, 'per_image_results_{}_{}.txt'.format(split, results_id)), 'w')
        self.per_class_txt_writer = open(os.path.join(save_path, 'per_class_results_{}_{}.txt'.format(split, results_id)), 'w')

        if split is None:
            split = self.cfg.TEST.SPLIT

        save_path = os.path.join(self.cfg.TEST.Analyze_Result_Path, self.cfg.DATASET.NAME,
        str(self.cfg.OPTIM.MAX_EPOCH)+'_'+str(self.cfg.SEED)+'_'+str(self.cfg.DATASET.NUM_SHOTS))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
            print("Do evaluation on {} set".format(split))
        elif split=="novel":
            data_loader = self.test_novel_loader
            print("Do evaluation on test novel set")
        elif split=="base":
            data_loader = self.test_base_loader
            print("Do evaluation on test base set")
        elif split=="all":
            data_loader = self.test_loader
            print("Do evaluation on test set")
        elif split=="train":
            data_loader = self.train_loader_x
            print("Do evaluation on train set")
        else:
            data_loader = self.test_loader
            print("Do evaluation on test set")

        for batch_idx, batch in enumerate(data_loader):
            input, label, impath = self.parse_batch_test_with_impath(batch)
            if trainer_list is None or len(trainer_list)==1:
                output, image_features, text_features = self.model.zero_shot_forward(input, self.device)
            else:
                # ensemble
                outputs = [t.model.zero_shot_forward(input, self.device)[0] for t in trainer_list]
                output = sum(outputs) / len(outputs)
            self.evaluator.process(output, label, self.per_image_txt_writer, self.per_class_txt_writer)
        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        self.per_image_txt_writer.close()
        self.per_class_txt_writer.close()

        return list(results.values())[0]


    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (except self.dm).
        """
        _, preprocess = clip.load(self.cfg.MODEL.BACKBONE.NAME)
        if self.cfg.DATASET.NAME in ElevaterDatasets:
            dm = ElevaterDataManager(self.cfg, custom_tfm_test=preprocess)      
        else:
            dm = UPLDataManager(self.cfg, custom_tfm_test=preprocess)
        # _, preprocess = clip.load(self.cfg.MODEL.BACKBONE.NAME)
        # dm = UPLDataManager(self.cfg, custom_tfm_test=preprocess)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader
        self.train_loader_sstrain = dm.train_loader_sstrain
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        if self.cfg.DATALOADER.OPEN_SETTING:
            self.test_novel_loader = dm.test_novel_loader
            self.test_base_loader = dm.test_base_loader

        self.dm = dm

    def sstrain_with_id(self, model_id):
        self.sstrain(self.start_epoch, self.max_epoch, model_id)

    def sstrain(self, start_epoch, max_epoch, model_id):
        """Generic training loops."""
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch

        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch_with_sstrain()
            self.after_epoch(model_id)
        self.after_train(model_id)

    def run_epoch_with_sstrain(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_sstrain)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_sstrain):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (
                self.batch_idx + 1
            ) % self.cfg.TRAIN.PRINT_FREQ == 0 or self.num_batches < self.cfg.TRAIN.PRINT_FREQ:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    "epoch [{0}/{1}][{2}/{3}]\t"
                    "time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "eta {eta}\t"
                    "{losses}\t"
                    "lr {lr:.6e}".format(
                        self.epoch + 1,
                        self.max_epoch,
                        self.batch_idx + 1,
                        self.num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        eta=eta,
                        losses=losses,
                        lr=self.get_current_lr(),
                    )
                )

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def after_epoch(self, model_id):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )
        # if ((self.epoch + 1) % 5) == 0 and self.cfg.DATASET.NAME!="SSImageNet":
        #     curr_result = self.test(split="test")
        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    model_name="model-best-{}.pth.tar".format(model_id)
                )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(
                    self.epoch,
                    self.output_dir,
                    model_name="model-last-{}.pth.tar".format(model_id)
                )
        

    def forward_get_conf(self, batch):
        _, _, impath = self.parse_batch_test_with_impath(batch)
        image, label, index = self.parse_batch_train(batch)     #When PLL: labels_true == self.labels_true[index]
        gt_label = self._get_gt_label(impath, dtype=label.dtype)

        output, image_features, text_features = self.model(image)
        self.criterion.check_conf_update(image, label, index, output=output)   

        summary = {
            "acc": compute_accuracy(output, gt_label)[0].item(),
            'index': index,
            'output': output,
            # 'label': label,
            # 'image_features': image_features,
            # 'text_features': text_features,
        }
        return summary

    @torch.no_grad()
    def before_epoch(self):
        if self.epoch == 0:            #self.epoch start from 0
            if 'refine' in self.criterion.losstype:
                self.criterion.cls_pools_dict = self.init_cls_pools(split="train")

        elif self.epoch > 0:
            self.set_model_mode("eval")
            self.model.eval()
            output_all = []; indexs_all = []

            for batch_idx, batch in enumerate(self.dm.train_loader_sstrain_notfm):
                summary = self.forward_get_conf(batch)
                indexs_all.append(summary['index'])
                output_all.append(summary['output'])
                if (
                    batch_idx + 1
                ) % self.cfg.TRAIN.PRINT_FREQ == 0 or self.num_batches < self.cfg.TRAIN.PRINT_FREQ:
                    print(f'batch_idx: {batch_idx}, pred_acc: {summary["acc"]}')

            indexs_all = torch.cat(indexs_all, dim=0)
            output_all = torch.cat(output_all, dim=0)
            if hasattr(self.criterion, 'cls_pools_dict'):
                pool_cur_capacity = self.cfg.TRAINER.PLL.MAX_POOLNUM * self.pool_certn_norm
            else:
                pool_cur_capacity = None
            pool_certn_norm, info_dict = self.criterion.update_conf_epochend(indexs_all, output_all, 
                                                                             pool_cur_capacity)
            self.pool_certn_norm = pool_certn_norm
            self.feat_idxs_halfsafe = info_dict.get('popped_idxs_safe', None)
            self.feat_idxs_unsafe = info_dict.get('popped_idxs_unsafe', None)

            if hasattr(self.criterion, 'cls_pools_dict'):
                pool_next_capacity = self.cfg.TRAINER.PLL.MAX_POOLNUM * pool_certn_norm
                for cls_idx, pool in self.criterion.cls_pools_dict.items():
                    pool.scale_pool(next_capacity=round(pool_next_capacity[cls_idx].item()))
                    pool.reset()
                    
        if self.epoch > 0:            #self.epoch start from 0
            if self.cfg.TRAINER.PLL.USE_PLL:
                self.criterion.clean_conf()


    @torch.no_grad()                           
    def init_cls_pools(self, split="train"):
        pools_dict = select_top_k_certainty_per_class(class_ids=torch.arange(0, len(self.lab2cname)), 
                                                      K=int(max(self.cfg.TRAINER.PLL.MAX_POOLNUM*0.4, 3)))     #选择每个类别visual emb和text emb最相似的K个样本，对每个样本取预测的vector，最后加到info dict中（k>=0时）。 对每个样本取预测的vector，然后加到所有训练样本的info dict中（k=-1时）
        return pools_dict
    

    def after_train(self, model_id):
        print("Finished training")

        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model_by_id(self.output_dir, model_id)
            # self.test(split='novel')
            # self.test(split='base')
            # self.test(split='train')
            self.test(split='test')

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

    @torch.no_grad()
    def test_with_existing_logits(self, logits, split='test'):
        self.set_model_mode("eval")
        self.evaluator.reset()

        save_path = os.path.join(self.cfg.TEST.Analyze_Result_Path, self.cfg.DATASET.NAME, 'fp'+str(self.cfg.TRAINER.UPLTrainer.NUM_FP),
        str(self.cfg.OPTIM.MAX_EPOCH)+'_'+str(self.cfg.SEED)+'_'+str(self.cfg.DATASET.NUM_SHOTS)+'_random_init'+str(self.cfg.TRAINER.UPLTrainer.CLASS_TOKEN_POSITION))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        results_id = 0
        while os.path.exists(os.path.join(save_path, 'per_image_results_{}_{}.txt'.format(split, results_id))):
            results_id += 1
        self.per_image_txt_writer = open(os.path.join(save_path, 'per_image_results_{}_{}.txt'.format(split, results_id)), 'w')
        self.per_class_txt_writer = open(os.path.join(save_path, 'per_class_results_{}_{}.txt'.format(split, results_id)), 'w')

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
            print("Do evaluation on {} set".format(split))
        elif split=="novel":
            data_loader = self.test_novel_loader
            print("Do evaluation on test novel set")
        elif split=="base":
            data_loader = self.test_base_loader
            print("Do evaluation on test base set")
        elif split=="all":
            data_loader = self.test_loader
            print("Do evaluation on test set")
        else:
            data_loader = self.test_loader
            print("Do evaluation on test set")

        label_all = []
        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            label_all.append(label)
        label_all = torch.hstack(label_all)
        print(label_all.shape)

        self.evaluator.process(logits, label_all, self.per_image_txt_writer, self.per_class_txt_writer)
        results = self.evaluator.evaluate()
        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        return results
