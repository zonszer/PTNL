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
        self.T = self.cfg.TEMPERATURE
        if '_' in type or '_' in self.cfg.CONF_LOSS_TYPE:   #means need to update conf
            self.conf = self.init_confidence(PartialY)
            self.conf_momn = self.cfg.CONF_MOMN
            if type == 'rc+':
                self.beta = self.cfg.BETA

        if 'refine' in self.losstype:
            self.origin_labels = deepcopy(self.conf)
            self.cls_pools_dict = {}
            self.safe_f = self.cfg.SAFE_FACTOR

        if 'gce' in type or 'gce' in self.cfg.CONF_LOSS_TYPE:
            self.q = 0.7

    def init_confidence(self, PartialY):        #TODO: remove this init for convience
        tempY = PartialY.sum(dim=1, keepdim=True).repeat(1, PartialY.shape[1])   #repeat train_givenY.shape[1] times in dim 1
        confidence = (PartialY/tempY).float()
        confidence = confidence.to(self.device)
        return confidence
    
    def forward(self, *args, reduce=True):
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
        if reduce:
            loss = loss.mean()
        return loss

    def check_conf_update(self, images, y, index, output=None):
        if '_' in self.cfg.CONF_LOSS_TYPE or '_' in self.losstype:
            if '_' in self.cfg.CONF_LOSS_TYPE:
                conf_type = self.cfg.CONF_LOSS_TYPE.split('_')[-1]
            else:
                conf_type = self.losstype.split('_')[-1]
            assert conf_type in ['rc', 'cav', 'refine'], 'conf_type not supported'
            self.update_confidence(y, index, conf_type, outputs=output)
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
        final_outputs = logsm_outputs * self.conf[index, :]
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
    def update_confidence(self, PL_labels, batch_idxs, conf_type, outputs):
        # outputs, image_features, text_features = model(images)
        if conf_type == 'rc':
            rc_conf = self.cal_pred_conf(outputs, PL_labels, conf_type)
            base_value = rc_conf.sum(dim=1).unsqueeze(1).repeat(1, rc_conf.shape[1])
            self.conf[batch_idxs, :] = rc_conf / base_value  # use maticx for element-wise division

        elif conf_type == 'cav':
            cav_conf = self.cal_pred_conf(outputs, PL_labels, conf_type)
            cav_pred = torch.max(cav_conf, dim=1)[1]
            gt_label = F.one_hot(cav_pred, PL_labels.shape[1]) # label_smoothing() could be used to further improve the performance for some datasets
            self.conf[batch_idxs, :] = gt_label.float()

        elif conf_type == 'refine':
            conf_type = 'cav'
            conf = self.cal_pred_conf(outputs, PL_labels, conf_type)
            self.fill_pools(conf, outputs, batch_idxs, top_pools=1, record_notinpool=True)


    @torch.no_grad()
    def cal_pred_conf(self, logits, PL_labels, conf_type):
        if conf_type == 'rc':
            conf = F.softmax(logits / self.T, dim=1)
        elif conf_type == 'cav':
            conf = (logits * torch.abs(1 - logits)) 
        return conf * PL_labels

    @torch.no_grad()
    def cal_uncertainty(self, output, label, index=None):
        """calculate uncertainty for each sample in batch"""
        unc = F.cross_entropy(output, label, reduction='none')
        return unc
    

    def fill_pools(self, conf, outputs, batch_idxs, top_pools=1, record_notinpool=True, pool_idxs=None):
        """fill pools with top_pools samples for each class"""
        if pool_idxs is None:
            pool_idxs = torch.arange(0, len(self.cls_pools_dict))
        if conf.shape[1] == 0 or outputs.shape[1] == 0:
            return 
        not_in_pool_init = torch.ones(outputs.shape[0], dtype=torch.bool)
        all_idxs = torch.arange(0, outputs.shape[0])

        def recursion(num_top_pools, output, cav_logits, not_in_pool, all_idxs):
            if num_top_pools == 0 or (not_in_pool==False).all():
                return 
            else:
                this_loop_idxs = all_idxs[not_in_pool]        #torch.arange(0, output.shape[0])[not_in_pool]
                # try:
                max_val, cav_pred = torch.max(cav_logits[this_loop_idxs], dim=1)
                # except:
                #     print('this_loop_idxs.shape: ', this_loop_idxs.shape)
                #     print('this_loop_idxs: ', this_loop_idxs)
                    # print('not_in_pool: ', not_in_pool)
                    # print('cav_logits.shape: ', cav_logits.shape)
                    # raise ValueError
                cls_ids = pool_idxs[cav_pred]
                unc = self.cal_uncertainty(output[this_loop_idxs], cav_pred)      

                for i, (cls_id, idx) in enumerate(zip(cls_ids, this_loop_idxs)):
                    pool = self.cls_pools_dict[cls_id.item()]
                    in_pool = pool.update(feat_idx=batch_idxs[idx], feat_unc=unc[i], 
                                          record_popped = record_notinpool)
                    not_in_pool[idx] = not in_pool
                    # if in_pool.item():    #HACK： should change the position
                        # self.pred_label_dict.update({batch_idxs[i].item(): [cls_idx.cpu().item(), unc[i].cpu().item()]})

            cav_logits[this_loop_idxs, cav_pred] = -torch.inf
            recursion(num_top_pools-1, output, cav_logits, not_in_pool, all_idxs)
            return 
        
        # call recursion:
        recursion(top_pools, outputs, conf, not_in_pool_init, all_idxs)


    def refill_pools(self, indexs_all, output_all):
        popped_idxs = []
        pool_not_full = torch.zeros(len(self.cls_pools_dict), dtype=torch.bool)

        for pool_id, cur_pool in self.cls_pools_dict.items():
            if cur_pool.popped_idx.shape[0] > 0:
                popped_idx, popped_unc = cur_pool.pop_notinpool_items()
                popped_idxs.append(popped_idx)

            if cur_pool.pool_capacity < cur_pool.pool_max_capacity:
                pool_not_full[pool_id] = True
            
        popped_idxs = torch.cat(popped_idxs, dim=0)
        sort_idxs = torch.argsort(indexs_all)       #output_all[sort_idxs] is the data original order
        not_full_idxs = torch.where(pool_not_full == True)[0]
        popped_output = output_all[sort_idxs][popped_idxs.unsqueeze(1), not_full_idxs.unsqueeze(0)]
        popped_labels = self.origin_labels[popped_idxs.unsqueeze(1), not_full_idxs.unsqueeze(0)]
        conf = self.cal_pred_conf(popped_output, popped_labels, conf_type='cav')

        for pool_id in not_full_idxs.tolist():
            cur_pool = self.cls_pools_dict[pool_id]
            cur_pool.freeze_stored_items()

        self.fill_pools(conf, popped_output, popped_idxs, 
                                    top_pools=self.cfg.TOP_POOLS, 
                                    record_notinpool=True, 
                                    pool_idxs=not_full_idxs)
        
        for pool_id in not_full_idxs.tolist():
            cur_pool = self.cls_pools_dict[pool_id]
            cur_pool.unfreeze_stored_items()
            cur_pool.recalculate_unc(logits_all=output_all[sort_idxs], criterion=self.cal_uncertainty)


    @torch.no_grad()
    def update_conf_epochend(self, indexs_all, output_all):      # shrink_f larger means val of unc_norm has larger effect on momn
        info_dict = {}
        pool_unc_avgs = None
        if 'refine' in self.losstype:
            print('update conf_refine at epoch end:')
            # pop items not in pools and fill the remained pools:
            self.refill_pools(indexs_all, output_all)
            
            # clean pool and calculate conf increament:
            (info_dict, pool_unc_avgs) = self.update_conf_refine(shrink_f=0.5)
            
            print(f'<{info_dict["not_inpool_num"]}> samples are not in pool:,'
                    f'<{info_dict["safe_range_num"]}> samples are in safe range,'
                    f'<{info_dict["clean_num"]}> samples are cleaned')   
            
        return pool_unc_avgs, info_dict

    def update_conf_refine(self, shrink_f=0.5):
        not_inpool_num = 0
        safe_range_num = 0
        clean_num = 0
        pool_unc_avgs = []
        popped_idxs_safe = []; popped_idxs_unsafe = []
        if self.losstype.split('_')[0] == 'rc':
            shrink_f = 0.75

        for pool_id, cur_pool in self.cls_pools_dict.items():
            print(f'pool_id: {pool_id}, pool_capacity: {cur_pool.pool_capacity}/{cur_pool.pool_max_capacity}')
            if cur_pool.pool_capacity == 0:
                pool_unc_avgs.append(torch.full((1,), torch.nan, dtype=torch.float16))
                continue
            if cur_pool.pool_capacity == cur_pool.pool_max_capacity: 
                unc_min = cur_pool.pool_unc.min()
                unc_norm = (cur_pool.pool_unc - unc_min) / (cur_pool.unc_max - unc_min)
            else:
                unc_norm = 0
            # if isinstance(unc_norm, torch.Tensor) and (torch.isnan(unc_norm)).any():
            #     unc_norm = 0
            pool_unc_avgs.append(cur_pool.pool_unc.mean().unsqueeze(0).cpu())
            
            # 1. cal increament and update conf:
            pred_conf_value = self.conf[cur_pool.pool_idx, pool_id]       
            conf_increment = (1 - self.conf_momn) * (1 - unc_norm * shrink_f) 
            pred_conf_value_ = pred_conf_value * self.conf_momn + conf_increment
            base_value = pred_conf_value_ - pred_conf_value + 1
            # assert (base_value >= 1).all(), 'base_value should larger than 1'     #TODO double check
            self.conf[cur_pool.pool_idx, pool_id] = pred_conf_value_

            # 2. clean poped items which are not in safe range:
            if cur_pool.popped_idx_past != None and cur_pool.popped_idx_past.shape[0] > 0:
                safe_range = ((cur_pool.popped_unc_past - self.safe_f*cur_pool.unc_max) <= 0)
                self.conf[cur_pool.popped_idx_past[~safe_range], :] = \
                            self.origin_labels[cur_pool.popped_idx_past[~safe_range], :]
                safe_range_num += safe_range.sum().item()
                clean_num += (~safe_range).sum().item()
                if self.losstype.split('_')[0] == 'rc':
                    popped_idxs_unsafe.extend(cur_pool.popped_idx_past[~safe_range].tolist())
                    popped_idxs_safe.extend(cur_pool.popped_idx_past[safe_range].tolist())

        pool_unc_avgs = torch.cat(pool_unc_avgs, dim=0)
        nan_mask = torch.isnan(pool_unc_avgs)
        idx = torch.nonzero(nan_mask)
        pool_unc_avgs[idx] = pool_unc_avgs[~nan_mask].max()

        max_unc = pool_unc_avgs.max()
        min_unc = pool_unc_avgs.min()
        pool_unc_norm =  - (pool_unc_avgs - min_unc) / (max_unc - min_unc) + 1.0

        not_inpool_num = safe_range_num + clean_num
        info = {
            'not_inpool_num': not_inpool_num,
            'safe_range_num': safe_range_num,
            'clean_num': clean_num,
            'popped_idxs_unsafe': {idx: True for idx in popped_idxs_unsafe} if len(popped_idxs_unsafe) > 0 else {}, 
            'popped_idxs_safe': {idx: True for idx in popped_idxs_safe} if len(popped_idxs_safe) > 0 else {},
        }
        return info, pool_unc_norm, 


    def clean_conf(self):
        if hasattr(self, 'conf'):
            self.conf = torch.where(self.conf < (1/self.conf.shape[1]), torch.zeros_like(self.conf), self.conf)
            base_value = self.conf.sum(dim=1).unsqueeze(1).repeat(1, self.conf.shape[1])
            self.conf = self.conf / base_value


    def log_conf(self, all_logits=None, all_labels=None):
        log_id = 'PLL' + str(self.cfg.PARTIAL_RATE)
        if self.num % 1 == 0 and self.num != 50:        #50 is the last epoch (test dataset)
            print(f'save logits -> losstype: {self.losstype}, save id: {self.num}')
            if self.losstype == 'cc_refine' or 'epoch' in self.losstype:      # need to run 2 times for getting conf
                if not hasattr(self, 'save_type'):
                    self.save_type = f'cc_refine'

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
