import math
import torch.nn.functional as F
import torch
import torch.nn as nn
from collections import defaultdict
from copy import deepcopy

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    if torch.cuda.device_count() > 1:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cuda')
else:
    device = torch.device('cpu')


class PLL_loss(nn.Module):
    softmax = nn.Softmax(dim=1)
    # opt params (assigned during code running)
    model: nn.Module
    pred_label_dict = defaultdict(list)
    gt_label_dict: dict = {}

    def __init__(self, type=None, PartialY=None,
                 eps=1e-7, cfg=None):
        super(PLL_loss, self).__init__()
        self.eps = eps
        self.losstype = type
        self.device = device
        self.cfg = cfg
        self.num = 0
        #PLL items: 
        self.loss_min = self.cfg.LOSS_MIN
        self.T = self.cfg.TEMPERATURE
        if '_' in type or '_' in self.cfg.CONF_LOSS_TYPE:   #means need to update conf
            self.conf = self.init_confidence(PartialY)
            self.conf_momn = self.cfg.CONF_MOMN
            if type == 'rc+':
                self.beta = self.cfg.BETA

        if 'refine' in self.losstype:
            self.not_inpool_idxs = torch.LongTensor([]).to(self.device)
            self.origin_labels = deepcopy(self.conf)
            self.cls_pools_dict = {}

        if 'gce' in type or 'gce' in self.cfg.CONF_LOSS_TYPE:
            self.q = 0.7

    def init_confidence(self, PartialY):        #TODO: remove this init for convience
        tempY = PartialY.sum(dim=1, keepdim=True).repeat(1, PartialY.shape[1])   #repeat train_givenY.shape[1] times in dim 1
        confidence = (PartialY/tempY).float()
        confidence = confidence.to(self.device)
        return confidence
    
    def forward(self, *args):
        """"
        x: outputs logits
        y: targets (multi-label binarized vector)
        """
        if self.losstype == 'cc' or 'epoch' in self.losstype:
            loss = self.forward_cc(*args)
        elif self.losstype == 'ce':
            loss = self.forward_ce(*args)
        elif self.losstype == 'gce':
            loss = self.forward_gce(*args)
        elif self.losstype in ['rc_rc', 'rc_cav', 'rc_refine']:      #can add gce_rc
            loss = self.forward_rc(*args)
        elif self.losstype in ['cc_rc', 'cc_refine']:      
            loss = self.forward_cc_plus(*args)
        elif self.losstype == 'rc+':
            loss = self.forward_rc_plus(*args)
        else:
            raise ValueError
        # loss = torch.clamp(loss - self.loss_min, min=0.0)       #TODO: remove here 
        return loss.mean()

    def check_update(self, images, y, index):
        if '_' in self.cfg.CONF_LOSS_TYPE or '_' in self.losstype:
            if '_' in self.cfg.CONF_LOSS_TYPE:
                conf_type = self.cfg.CONF_LOSS_TYPE.split('_')[-1]
            else:
                conf_type = self.losstype.split('_')[-1]
            assert conf_type in ['rc', 'cav', 'refine'], 'conf_type not supported'
            self.update_confidence(self.model, self.conf, images, y, index, conf_type)
        else:
            return
    

    def forward_gce(self, x, y, index=None):
        """y is shape of (batch_size, num_classes (0 ~ 1.)), one-hot vector"""
        p = F.softmax(x, dim=1)      #outputs are logits
        # Create a tensor filled with a very small number to represent 'masked' positions
        masked_p = p.new_full(p.size(), float('-inf'))
        # Apply the mask
        masked_p[y.bool()] = p[y.bool()] + self.eps         
        # Adjust masked positions to avoid undefined gradients by adding epsilon
        masked_p[y.bool()] = (1 - masked_p[y.bool()] ** self.q) / self.q
        masked_p[~y.bool()] = self.eps 
        loss = masked_p.sum(dim=1)
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is infinite or NaN!")
        return loss
    
    def forward_gce_rc(self, x, y, index):
        """y is shape of (batch_size, num_classes (0 ~ 1.)), one-hot vector"""
        p = F.softmax(x, dim=1)      #outputs are logits
        # Create a tensor filled with a very small number to represent 'masked' positions
        masked_p = p.new_full(p.size(), float('-inf'))

        # p = p.float16()                                               #HACK solution here
        # masked_p = masked_p * self.confidence[index, :]       #NOTE add multiple conf here    
        # Apply the mask
        masked_p[y.bool()] = p[y.bool()] + self.eps      
        # Adjust masked positions to avoid undefined gradients by adding epsilon
        masked_p[y.bool()] = (1 - masked_p[y.bool()] ** self.q) / self.q        
        masked_p[~y.bool()] = self.eps 
        loss = (masked_p * self.conf[index, :]).sum(dim=1)    #NOTE add multiple conf here   
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is infinite or NaN!")
        return loss
    
    def forward_cc(self, x, y, index=None):
        sm_outputs = F.softmax(x, dim=1)      #outputs are logits
        final_outputs = sm_outputs * y
        loss = - torch.log(final_outputs.sum(dim=1))     #NOTE: add / y.sum(dim=1)
        return loss
    
    def forward_ce(self, x, y, index=None):
        sm_outputs = F.log_softmax(x, dim=1)
        final_outputs = sm_outputs * y
        loss = - final_outputs.sum(dim=1)        #NOTE: add y.sum(dim=1)
        return loss
    
    def forward_rc(self, x, y, index):
        logsm_outputs = F.log_softmax(x, dim=1)         #x is the model ouputs
        final_outputs = logsm_outputs * (self.conf[index, :] + self.eps)
        loss = - final_outputs.sum(dim=1)    #final_outputs=torch.Size([256, 10]) -> Q: why use negative? A:  
        return loss     

    def forward_cc_plus(self, x, y, index):
        logsm_outputs = F.softmax(x / self.T, dim=1)         #x is the model ouputs
        final_outputs = logsm_outputs * self.conf[index, :]
        loss = - torch.log((final_outputs).sum(dim=1))    #final_outputs=torch.Size([256, 10]) -> Q: why use negative? A:  
        # if torch.where(torch.isnan(final_outputs))[0].shape[0] > 0:
        #     raise FloatingPointError("Loss is infinite or NaN!")
        return loss 
        

    def forward_rc_plus(self, x, y, index):
        logsm_outputs = F.softmax(x, dim=1)         #x is the model ouputs
        # y_new = self.update_partial_labels(y, self.confidence, index)
        final_outputs = logsm_outputs * y
        cc_loss = - torch.log(final_outputs.sum(dim=1)) 

        conf_loss = self.get_conf_loss(x, y, index, type=self.cfg.CONF_LOSS_TYPE)

        loss = cc_loss + self.beta*conf_loss
        # if not torch.isfinite(loss).all():
        #     raise FloatingPointError("Loss is infinite or NaN!")
        return loss     

    def get_conf_loss(self, x, y, index, type=None):
        #non-conf loss below:
        if type=='ce':
            conf_loss = self.forward_ce(x, y, index)
        elif type=='gce':
            conf_loss = self.forward_gce(x, y, index)
        #conf loss below:
        elif type in ['rc_rc', 'rc_cav']:               
            conf_loss = self.forward_rc(x, y, index)
        elif type=='gce_rc':
            conf_loss = self.forward_gce_rc(x, y, index)    
        else:
            raise ValueError('conf_loss type not supported')
        return conf_loss  

    @torch.no_grad()
    def update_confidence(self, model, confidence, images, labels, batch_idxs, conf_type):
        outputs, image_features, text_features = model(images)
        if conf_type == 'rc':
            temp_un_conf = F.softmax(outputs / self.T, dim=1)
            conf_selected = temp_un_conf * labels # un_confidence stores the weight of each example
            base_value = conf_selected.sum(dim=1).unsqueeze(1).repeat(1, conf_selected.shape[1])
            self.conf[batch_idxs, :] = conf_selected / base_value  # use maticx for element-wise division
        elif conf_type == 'cav':
            cav = (outputs * torch.abs(1 - outputs)) * labels
            cav_pred = torch.max(cav, dim=1)[1]
            gt_label = F.one_hot(cav_pred, labels.shape[1]) # label_smoothing() could be used to further improve the performance for some datasets
            self.conf[batch_idxs, :] = gt_label.float()
        elif conf_type == 'refine':
            batch_idxs = batch_idxs.to(self.device)
            cav = (outputs * torch.abs(1 - outputs)) * labels
            max_idx, cav_pred = torch.max(cav, dim=1)
            unc = self.cal_uncertainty(outputs, cav_pred)      

            in_pool = torch.empty(0, dtype=torch.bool).to(self.device)

            for i, cls_idx in enumerate(cav_pred):
                pool = self.cls_pools_dict[cls_idx.item()]
                in_pool_ = pool.update(feat_idxs=batch_idxs[i].unsqueeze(0), feat_unc=unc[i].unsqueeze(0))
                in_pool = torch.cat((in_pool, in_pool_))
                # if in_pool_.item():
                #     self.pred_label_dict.update({batch_idxs[i].item(): [cls_idx.cpu().item(), unc[i].cpu().item()]})

            self.not_inpool_idxs = torch.cat((self.not_inpool_idxs, batch_idxs[~in_pool]))

            # if self.cfg.TOP_POOLS != 1:
            #     pass                        #TODO add recursion here
            # else:
            #     # self.conf[batch_idxs[~in_pool], :] = labels[~in_pool]     #TODO can ehance here, if not in pool how to change the conf
            #     pass

    def clean_conf(self):
        if hasattr(self, 'origin_labels'):
            self.conf[self.not_inpool_idxs, :] = self.origin_labels[self.not_inpool_idxs, :]
        if hasattr(self, 'conf'):
            self.conf = torch.where(self.conf < self.eps, torch.zeros_like(self.conf), self.conf)


    def update_conf_epochend(self, pool_id, shrink_f=0.1):      # shrink_f larger means val of unc_norm has larger effect on momn
        assert 'refine' in self.losstype
        cur_pool = self.cls_pools_dict[pool_id]

        pred_conf_value = self.conf[cur_pool.pool_idx, pool_id]        
        if cur_pool.pool_capacity > 1:
            unc_norm = (cur_pool.pool_unc - cur_pool.pool_unc.min()) / (cur_pool.pool_unc.max() - cur_pool.pool_unc.min())
            conf_increment = (1 - self.conf_momn) * (1 - unc_norm * shrink_f) 
            pred_conf_value_ = pred_conf_value * self.conf_momn + conf_increment
        else:
            pred_conf_value_ = pred_conf_value * self.conf_momn + (1 - self.conf_momn)
        base_value = pred_conf_value_ - pred_conf_value + 1

        # assert (base_value >= 1).all(), 'base_value should larger than 1'     #TODO double check
        self.conf[cur_pool.pool_idx, pool_id] = pred_conf_value_
        self.conf[cur_pool.pool_idx, :] = self.conf[cur_pool.pool_idx, :] / base_value.unsqueeze(1).repeat(1, self.conf.shape[1])
        # if torch.where(torch.isnan(self.conf[cur_pool.pool_idx, :]))[0].shape[0] > 0:
        #     raise FloatingPointError("Loss is infinite or NaN!")

    def search_pools(self):
        pass
    

    @torch.no_grad()
    def cal_uncertainty(self, output, label, index=None):
        """calculate uncertainty for each sample in batch"""
        unc = F.cross_entropy(output, label, reduction='none')
        return unc

    def log_conf(self, all_logits=None, all_labels=None):
        log_id = 'PLL' + str(self.cfg.PARTIAL_RATE)
        if self.num % 1 == 0 and self.num != 50:        #50 is the last epoch (test dataset)
            print(f'save logits -> losstype: {self.losstype}, save id: {self.num}')
            if self.losstype == 'cc_refine' or 'epoch' in self.losstype:      # need to run 2 times for getting conf
                if not hasattr(self, 'save_type'):
                    self.save_type = f'cc_refine_{self.losstype.split("-")[1]}epoch'

                torch.save(self.conf, f'analyze_result_temp/logits&labels_10.14/conf-{self.save_type}_{log_id}-{self.num}.pt')
                torch.save(self.pred_label_dict, f'analyze_result_temp/logits&labels_10.14_pool/pred_label-{self.save_type}_{log_id}-{self.num}.pt')
                torch.save(self.gt_label_dict, f'analyze_result_temp/logits&labels_10.14_pool/gt_label-{self.save_type}_{log_id}-{self.num}.pt')
                self.save_label_pools()
                self.pred_label_dict = defaultdict(list)
                self.gt_label_dict = {}

            elif all_logits != None:
                all_logits = F.softmax(all_logits, dim=1)    
                all_labels = F.one_hot(all_labels)
                torch.save(all_logits,  f'analyze_result_temp/logits&labels/outputs_{self.losstype.upper()+self.cfg.CONF_LOSS_TYPE}_{log_id}-{self.num}.pt')
                torch.save(all_labels,  f'analyze_result_temp/logits&labels/labels_{self.losstype.upper()+self.cfg.CONF_LOSS_TYPE}_{log_id}-{self.num}.pt')
        self.num += 1

    def save_label_pools(self):
        log_id = 'PLL' + str(self.cfg.PARTIAL_RATE)
        torch.save(self.cls_pools_dict, f'analyze_result_temp/logits&labels_10.14_pool/pools_dict-{self.save_type}_{log_id}-{self.num}.pt')

class GeneralizedCrossEntropy(nn.Module):
    """Computes the generalized cross-entropy loss, from `
    "Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels"
    <https://arxiv.org/abs/1805.07836>`_
    Args:
        q: Box-Cox transformation parameter, :math:`\in (0,1]`
    Shape:
        - Input: the raw, unnormalized score for each class.
                tensor of size :math:`(minibatch, C)`, with C the number of classes
        - Target: the labels, tensor of size :math:`(minibatch)`, where each value
                is :math:`0 \leq targets[i] \leq C-1`
        - Output: scalar
    """

    def __init__(self, q: float = 0.7) -> None:
        super().__init__()
        self.q = q
        self.epsilon = 1e-6
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        '''target.shape is (batch_size,)'''
        p = self.softmax(input)
        p = p[torch.arange(p.shape[0]), target]
        # Avoid undefined gradient for p == 0 by adding epsilon
        p += self.epsilon
        loss = (1 - p ** self.q) / self.q
        return torch.mean(loss)

class Unhinged(nn.Module):
    """Computes the Unhinged (linear) loss, from
    `"Learning with Symmetric Label Noise: The Importance of Being Unhinged"
    <https://arxiv.org/abs/1505.07634>`_
    Shape:
        - Input: the raw, unnormalized score for each class.
                tensor of size :math:`(minibatch, C)`, with C the number of classes
        - Target: the labels, tensor of size :math:`(minibatch)`, where each value
                is :math:`0 \leq targets[i] \leq C-1`
        - Output: scalar
    """

    def __init__(self) -> None:
        super().__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.softmax(input)
        p = p[torch.arange(p.shape[0]), target]

        loss = 1 - p

        return torch.mean(loss)


class PHuberCrossEntropy(nn.Module):
    """Computes the partially Huberised (PHuber) cross-entropy loss, from
    `"Can gradient clipping mitigate label noise?"
    <https://openreview.net/pdf?id=rklB76EKPr>`_
    Args:
        tau: clipping threshold, must be > 1
    Shape:
        - Input: the raw, unnormalized score for each class.
                tensor of size :math:`(minibatch, C)`, with C the number of classes
        - Target: the labels, tensor of size :math:`(minibatch)`, where each value
                is :math:`0 \leq targets[i] \leq C-1`
        - Output: scalar
    """

    def __init__(self, tau: float = 10) -> None:
        super().__init__()
        self.tau = tau

        # Probability threshold for the clipping
        self.prob_thresh = 1 / self.tau
        # Negative of the Fenchel conjugate of base loss at tau
        self.boundary_term = math.log(self.tau) + 1

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.softmax(input)
        p = p[torch.arange(p.shape[0]), target]

        loss = torch.empty_like(p)
        clip = p <= self.prob_thresh
        loss[clip] = -self.tau * p[clip] + self.boundary_term
        loss[~clip] = -torch.log(p[~clip])

        return torch.mean(loss)


class PHuberGeneralizedCrossEntropy(nn.Module):
    """Computes the partially Huberised (PHuber) generalized cross-entropy loss, from
    `"Can gradient clipping mitigate label noise?"
    <https://openreview.net/pdf?id=rklB76EKPr>`_
    Args:
        q: Box-Cox transformation parameter, :math:`\in (0,1]`
        tau: clipping threshold, must be > 1
    Shape:
        - Input: the raw, unnormalized score for each class.
                tensor of size :math:`(minibatch, C)`, with C the number of classes
        - Target: the labels, tensor of size :math:`(minibatch)`, where each value
                is :math:`0 \leq targets[i] \leq C-1`
        - Output: scalar
    """

    def __init__(self, q: float = 0.7, tau: float = 10) -> None:
        super().__init__()
        self.q = q
        self.tau = tau

        # Probability threshold for the clipping
        self.prob_thresh = tau ** (1 / (q - 1))
        # Negative of the Fenchel conjugate of base loss at tau
        self.boundary_term = tau * self.prob_thresh + (1 - self.prob_thresh ** q) / q

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.softmax(input)
        p = p[torch.arange(p.shape[0]), target]

        loss = torch.empty_like(p)
        clip = p <= self.prob_thresh
        loss[clip] = -self.tau * p[clip] + self.boundary_term
        loss[~clip] = (1 - p[~clip] ** self.q) / self.q

        return torch.mean(loss)


class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss
    
    
class NormalizedCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(NormalizedCrossEntropy, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (- pred.sum(dim=1))
        return self.scale * nce.mean()

class ReverseCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(ReverseCrossEntropy, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        return self.scale * rce.mean()


class NCEandRCE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes):
        super(NCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.nce = NormalizedCrossEntropy(scale=alpha, num_classes=num_classes)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.rce(pred, labels)
