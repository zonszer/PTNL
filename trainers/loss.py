import math
import torch.nn.functional as F
import torch
import torch.nn as nn
from collections import defaultdict
from copy import deepcopy
from utils_temp.utils_ import find_elem_idx_BinA
from sklearn.mixture import GaussianMixture

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
        # self.origin_labels = self.init_confidence(PartialY)
        if '_' in type or '_' in self.cfg.CONF_LOSS_TYPE:   #means need to update conf
            self.conf = self.init_confidence(PartialY)
            self.conf_momn = self.cfg.CONF_MOMN
            if type == 'rc+':
                self.beta = self.cfg.BETA

        if 'refine' in self.losstype:
            self.origin_labels = deepcopy(self.conf)
            # self.cls_pools_dict = {}
            self.safe_f = self.cfg.SAFE_FACTOR

        if 'gce' in type or 'gce' in self.cfg.CONF_LOSS_TYPE:
            self.q = 0.7
        if 'lw' in type:
            self.lw_weight = 2; self.lw_weight0 = 1
            if 'refine' in type:
                self.conf_true = torch.zeros(self.conf.shape, dtype=self.conf.dtype).to(self.device)

    def init_confidence(self, PartialY):       
        tempY = PartialY.sum(dim=1, keepdim=True).repeat(1, PartialY.shape[1])   #repeat train_givenY.shape[1] times in dim 1
        confidence = (PartialY/tempY).float()
        confidence = confidence.to(self.device)
        return confidence
    
    def forward(self, *args, reduce=False):
        """"
        x: output logits
        y: partialY (multi-label binarized vector)
        """
        if self.losstype == 'cc' or 'epoch' in self.losstype:
            loss = self.forward_cc(*args)
        elif self.losstype == 'ce':
            loss = self.forward_ce(*args)
        elif self.losstype == 'gce':
            loss = self.forward_gce(*args)
        elif self.losstype in ['rc_rc', 'rc_cav', 'rc_refine', 'cav_refine']:      #can add gce_rc
            loss = self.forward_rc(*args)
        elif self.losstype in ['cc_rc', 'cc_refine']:      
            loss = self.forward_cc_plus(*args)
        elif self.losstype in ['lw_lw', 'lw_refine']:
            loss = self.forward_lw(*args)
        elif self.losstype == 'rc+':
            loss = self.forward_rc_plus(*args)
        else:
            raise ValueError
        if reduce:
            loss = loss.mean()
        return loss

    def check_conf_update(self, images, y, index, output=None):
        # self.origin_labels[index, :] = output.float()
        if '_' in self.cfg.CONF_LOSS_TYPE or '_' in self.losstype:
            if '_' in self.cfg.CONF_LOSS_TYPE:
                conf_type = self.cfg.CONF_LOSS_TYPE.split('_')[-1]
            else:
                conf_type = self.losstype.split('_')[-1]
            assert conf_type in ['rc', 'cav', 'refine', 'lw'], 'conf_type not supported'
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
        logsm_outputs = F.softmax(x, dim=1)         #x is the model ouputs
        final_outputs = logsm_outputs * self.conf[index, :]
        loss = - torch.log((final_outputs).sum(dim=1))    #final_outputs=torch.Size([256, 10]) -> Q: why use negative? A:  
        # if torch.where(torch.isnan(final_outputs))[0].shape[0] > 0:
        #     raise FloatingPointError("Loss is infinite or NaN!")
        return loss 

    def forward_lw(self, x, y, index):
        onezero = torch.zeros(x.shape[0], x.shape[1])
        onezero[y > 0] = 1
        counter_onezero = 1 - onezero
        onezero = onezero.to(self.device)
        counter_onezero = counter_onezero.to(self.device)

        sm_outputs = F.softmax(x, dim=1)

        sig_loss1 = - torch.log(sm_outputs + self.eps)
        l1 = self.conf[index, :] * onezero * sig_loss1
        average_loss1 = l1.sum(dim=1) 

        sig_loss2 = - torch.log(1 - sm_outputs + self.eps)
        l2 = self.conf[index, :] * counter_onezero * sig_loss2      
        average_loss2 = l2.sum(dim=1) 

        average_loss = self.lw_weight0 * average_loss1 + self.lw_weight * average_loss2
        return average_loss

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

        elif conf_type == 'lw':
            lw_conf = self.cal_pred_conf(outputs, PL_labels, conf_type)
            new_weight1, new_weight2 = lw_conf
            new_weight1 = new_weight1 / (new_weight1 + self.eps).sum(dim=1).repeat(
                    self.conf.shape[1], 1).transpose(0, 1)
            new_weight2 = new_weight2 / (new_weight2 + self.eps).sum(dim=1).repeat(
                    self.conf.shape[1], 1).transpose(0, 1)
            new_weight = (new_weight1 + new_weight2) 
            self.conf[batch_idxs, :] = new_weight

        elif conf_type == 'refine':
            conf_type_ = self.losstype.split('_')[0]
            conf = self.cal_pred_conf(outputs, PL_labels, conf_type_)
            if conf_type_ == 'cav':
                cav_pred = torch.max(conf, dim=1)[1]
                gt_label = F.one_hot(cav_pred, PL_labels.shape[1]) 
                conf_ = gt_label.float()
                self.conf[batch_idxs, :] = conf_
            elif conf_type_ == 'rc':                                               #NOTE when add method, each one need to / base_value because of zero-shot lables (to maintain containce with original PL_label)
                base_value = conf.sum(dim=1).unsqueeze(1).repeat(1, conf.shape[1])
                conf_ = conf / base_value  # use maticx for element-wise division
                self.conf[batch_idxs, :] = conf_  # use maticx for element-wise division
            elif conf_type_ == 'cc':
                base_value = conf.sum(dim=1).unsqueeze(1).repeat(1, conf.shape[1])      
                conf_ = conf / base_value  # use maticx for element-wise division
                # self.conf[batch_idxs, :] = self.origin_labels[batch_idxs, :]  # use maticx for element-wise division
            elif conf_type_ == 'lw':
                new_weight1, new_weight2 = conf
                new_weight1 = new_weight1 / (new_weight1 + self.eps).sum(dim=1).repeat(
                        self.conf.shape[1], 1).transpose(0, 1)
                new_weight2 = new_weight2 / (new_weight2 + self.eps).sum(dim=1).repeat(
                        self.conf.shape[1], 1).transpose(0, 1)
                new_weight = (new_weight1 + new_weight2) 
                self.conf_true[batch_idxs, :] = new_weight1       
                self.conf[batch_idxs, :] = new_weight
                conf_ = new_weight1         
            else:
                raise ValueError('conf_type not supported')
            # if hasattr(self, 'cls_pools_dict'):
                # self.fill_pools(conf_, outputs, batch_idxs, max_iter_num=1, record_notinpool=True)


    @torch.no_grad()
    def cal_pred_conf(self, logits, PL_labels, conf_type, lw_return2=True):
        if conf_type == 'rc':
            conf = F.softmax(logits, dim=1)
            conf = conf * PL_labels
        elif conf_type == 'cav':
            conf = (logits * torch.abs(1 - logits)) 
            conf = conf * PL_labels
        elif conf_type == 'cc':
            conf = F.softmax(logits, dim=1)
            conf = conf * PL_labels
        elif conf_type == 'lw':
            sm_outputs = F.softmax(logits, dim=1)
            onezero = torch.zeros(sm_outputs.shape[0], sm_outputs.shape[1])
            onezero[PL_labels > 0] = 1
            counter_onezero = 1 - onezero
            onezero = onezero.to(self.device)
            counter_onezero = counter_onezero.to(self.device)

            new_weight1 = sm_outputs * onezero
            new_weight2 = sm_outputs * counter_onezero
            if lw_return2:
                conf = (new_weight1, new_weight2)
            else:
                conf = new_weight1

        return conf


    @torch.no_grad()
    def cal_uncertainty(self, output, label, index=None):
        """calculate uncertainty for each sample in batch"""
        unc = F.cross_entropy(output, label, reduction='none')
        return unc
    

    def fill_pools(self, labels, uncs, feat_idxs, max_iter_num=1, record_notinpool=True, pool_idxs=None):
        """fill pools with top_pools samples for each class"""
        if pool_idxs is None:
            pool_idxs = torch.arange(0, len(self.cls_pools_dict)).to(self.device)
        if labels.shape[1] == 0:
            return 
        assert max_iter_num >= 1
        not_in_pool_init = torch.ones(labels.shape[0], dtype=torch.bool)
        del_elems_init = torch.zeros(labels.shape[0], dtype=torch.bool)
        all_idxs = torch.arange(0, labels.shape[0])
        quary_num = torch.zeros(labels.shape[0], dtype=torch.long)
        
        def recursion(top_uncs, top_labels, not_in_pool, del_elems):
            in_itering = (not_in_pool & ~del_elems)
            if (in_itering).sum() == 0 or (not_in_pool==False).all():
                return 
            else:
                this_loop_idxs = all_idxs[in_itering]        #torch.arange(0, output.shape[0])[not_in_pool]
                this_loop_uncs = top_uncs[this_loop_idxs, quary_num[this_loop_idxs]]
                this_loop_labels = top_labels[this_loop_idxs, quary_num[this_loop_idxs]]
                quary_num[this_loop_idxs] += 1
                assert labels.shape[0] == (this_loop_idxs.shape[0] + self.Pools.cal_pool_sum_num() + del_elems.sum()), "All_samples = not_in + in_pool + not_in_but_enough_iter"
                # unc[mask] = torch.inf
                this_loop_fill_num = 0
                for i, (cls_id, idx) in enumerate(zip(this_loop_labels, this_loop_idxs)):
                    pool = self.cls_pools_dict[cls_id.item()]
                    in_pool = pool.update(feat_idx=feat_idxs[idx], feat_unc=this_loop_uncs[i], 
                                            record_popped = record_notinpool)
                    not_in_pool[idx] = not in_pool
                    if in_pool:
                        this_loop_fill_num += 1
                    # if in_pool.item():    #HACK： should change the position
                        # self.pred_label_dict.update({batch_idxs[i].item(): [cls_idx.cpu().item(), unc[i].cpu().item()]})

                popped_feat_idxs, _, popped_unc = self.collect_popped_items(pool_range=pool_idxs.tolist(), 
                                                                             retain=False)
                elem_idxs = find_elem_idx_BinA(A=feat_idxs, B=popped_feat_idxs)  
                # not_found_idxs_.append(popped_feat_idxs[not_found_idxs]); not_found_uncs_.append(popped_unc[not_found_idxs])
                not_in_pool[elem_idxs] = True
                del_elems = (quary_num[all_idxs] == max_iter_num) & (not_in_pool)
                # del_elems = quary_num[all_idxs] == max_iter_num
                # not_in_pool[del_elem_idxs] = False
                recursion(top_uncs, top_labels, not_in_pool, del_elems)
        
        # call recursion:
        recursion(uncs, labels, not_in_pool_init, del_elems_init)
        return feat_idxs[not_in_pool_init], uncs[not_in_pool_init, :][:,0]    #get top 1 uncertainty for each sample
        

    def collect_popped_items(self, pool_range=None, retain=False):
        popped_idxs = []
        popped_uncs = []
        pools_not_full = torch.zeros(len(self.cls_pools_dict), dtype=torch.bool)
        if pool_range == None:
            pool_range = self.cls_pools_dict
        for pool_id in pool_range:
            cur_pool = self.cls_pools_dict[pool_id]
            if cur_pool.popped_idx.shape[0] > 0:
                popped_idx, popped_unc = cur_pool.pop_notinpool_items(retain=retain)
                popped_idxs.append(popped_idx)
                popped_uncs.append(popped_unc)

            if cur_pool.pool_capacity < cur_pool.pool_max_capacity:
                pools_not_full[pool_id] = True
        
        if popped_idxs != []:
            popped_idxs = torch.cat(popped_idxs, dim=0)
            popped_uncs = torch.cat(popped_uncs, dim=0)
        else:
            popped_idxs = torch.tensor([], dtype=torch.long)
            popped_uncs = torch.tensor([], dtype=torch.float16).to(self.device)
        return popped_idxs, pools_not_full, popped_uncs


    def collect_uncs_byCls(self, outputs, indexs, min_num=1):
        """collect uncs for all items by their PL_labels"""
        sort_idxs = torch.argsort(indexs)       #output_all[sort_idxs] is the data original order
        feat_idxs = indexs[sort_idxs]
        outputs = outputs[sort_idxs]
        # get the idxs of where PL_labels > eps and collect the corresponding outputs by idx order:
        items_collected = {}
        idxs_collected = {}
        PL_labels = self.conf > self.eps
        idxs = torch.nonzero(PL_labels)     # idxs is shape of (num, 2), 2 means (batch_idx, class_idx)
        self.conf_weight = deepcopy(self.conf)

        for i in range(idxs.shape[0]):
            cls = idxs[i][1].item()
            if cls not in items_collected:
                items_collected[cls] = []                       #HACK if UPL and use CLIP as lable (not uniform) one cls may be zero
                idxs_collected[cls] = []
            items_collected[cls].append(idxs[i][0].unsqueeze(0))
            idxs_collected[cls].append(idxs[i].unsqueeze(0))
        
        inpool_num, notinpool_num = {}, {}
        for cls in items_collected:
            cls_feat_idxs = torch.cat(items_collected[cls], dim=0)
            cls_origin_idxs = torch.cat(idxs_collected[cls], dim=0)
            cls_outputs = outputs[cls_feat_idxs]
            cls_labels = torch.full((cls_outputs.shape[0],), cls, dtype=torch.long).to(self.device)
            cls_uncs = self.cal_uncertainty(cls_outputs, cls_labels)
            in_pool, prob = self.fit_GMM(cls_uncs.cpu())
            base_value = prob.sum()
            weight = (prob / base_value).to(self.device)  # use maticx for element-wise division

            self.conf_weight[cls_origin_idxs[:, 0], cls_origin_idxs[:, 1]] = weight
            inpool_num[cls] = in_pool.shape[0]
            notinpool_num[cls] = prob.shape[0] - in_pool.shape[0]

        labels, uncs = self.prepare_items_attrs(outputs, feat_idxs, 
                                                max_num=1)
        not_inpool_feat_idxs, notinpool_uncs = self.fill_pools(labels, uncs, feat_idxs, 
                                                        max_iter_num=1,
                                                        record_notinpool=True)
        return inpool_num, notinpool_num, not_inpool_feat_idxs, notinpool_uncs

    def fit_GMM(self, uncs, min_num=1):
        """fit GMM for class items"""
        # Normalize the losses
        if uncs.shape[0] > 1:
            uncs = (uncs - uncs.min()) / (uncs.max() - uncs.min())
        input_uncs = uncs.reshape(-1, 1)  # Reshape for GMM fitting
        # Fit a Gaussian Mixture Model (GMM) on the input losses
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(input_uncs)
        prob = gmm.predict_proba(input_uncs)  # Predict the probabilities of the input being in one of the GMM components
        prob = prob[:, gmm.means_.argmin()]  # Select the probabilities corresponding to the component with the smaller mean loss
        pred_idx = (prob > 0.5).nonzero()[0]
        # not_in_idxs = (prob < 0.5).nonzero()[0]        
        prob = torch.from_numpy(prob).float()
        return pred_idx, prob, #not_in_idxs


    def prepare_items_attrs(self, outputs, indexs, max_num):    #TODO max_num should adjust according to partial ratio
        """prepare labels and uncs for all items"""
        PL_labels = self.origin_labels[indexs, :]
        conf = self.cal_pred_conf(outputs, PL_labels,     #TODO we can norm pred_conf all in cal_pred_conf()
                                conf_type=self.losstype.split('_')[0],
                                lw_return2=False)  
        labels_ = []; uncs_ = []
        for i in range(max_num):
            max_val, max_idx = torch.max(conf, dim=1)       # get max val in each row
            mask = (max_val < self.eps)                     
            uncs = self.cal_uncertainty(outputs, max_idx)     #uncs is shape of (batch_size, max_num)
            uncs[mask] = torch.inf
            conf[torch.arange(0, conf.shape[0]), max_idx] = -torch.inf

            labels_.append(max_idx.unsqueeze(1))
            uncs_.append(uncs.unsqueeze(1))
        labels_ = torch.cat(labels_, dim=1)
        uncs_ = torch.cat(uncs_, dim=1)

        return labels_, uncs_

    def refill_pools(self, indexs_all, output_all):
        sort_idxs = torch.argsort(indexs_all)       #output_all[sort_idxs] is the data original order
        feat_idxs = indexs_all[sort_idxs]
        outputs = output_all[sort_idxs]

        pred_labels = torch.argmax(outputs, dim=1)
        # class_counts = Counter(pred_labels.cpu().numpy()) 
        # get the predicted class differet count number from class_id = 0:
        self.class_counts = torch.ones(self.conf.shape[1], dtype=torch.long)
        # for i in range(self.conf.shape[1]):
        #     self.class_counts[i] = class_counts.get(i, 0)
        # # normalize the class_counts to [0, 1] by min max norm:
        # self.class_counts = (self.class_counts - self.class_counts.min()) / (self.class_counts.max() - self.class_counts.min())

        #prepare uncessay attrs for all items:
        labels, uncs = self.prepare_items_attrs(outputs, feat_idxs, 
                                                max_num=self.cfg.TOP_POOLS)
        not_inpool_feat_idxs, notinpool_uncs = self.fill_pools(labels, uncs, feat_idxs, 
                                                        max_iter_num=self.cfg.TOP_POOLS, 
                                                        record_notinpool=True)

        return not_inpool_feat_idxs, notinpool_uncs


    @torch.no_grad()
    def update_conf_epochend(self, indexs_all, output_all):      # shrink_f larger means val of unc_norm has larger effect on momn
        info_dict = {}
        pools_certainty_norm = None
        if 'refine' in self.losstype and hasattr(self, 'cls_pools_dict'):
            print('-----------update conf_refine at epoch end:-----------')
            # pop items not in pools and fill the remained pools:
            # notinpool_idxs, notinpool_uncs = self.refill_pools(indexs_all, output_all)
            (inpool_num, notinpool_num, notinpool_idxs, notinpool_uncs,
             ) = self.collect_uncs_byCls(output_all, indexs_all)
            
            #clean pool and calculate conf increament:
            info_dict = self.update_conf_refine(notinpool_idxs, notinpool_uncs)

            # print info:
            inpool_all, notin_all = 0, 0
            for cls in sorted(inpool_num):
                print(f'cls: {cls}, inpool_num: {inpool_num[cls]}, notinpool_num: {notinpool_num[cls]}')
                inpool_all += inpool_num[cls]
                notin_all += notinpool_num[cls]
            print(f'<sum {inpool_all}> samples are in pools,'
                  f'<sum {notin_all}> samples are not in pools,')
            self.Pools.print()
            self.Pools.cal_pool_ACC()
            
        return pools_certainty_norm, info_dict, torch.ones(self.conf.shape[1], dtype=torch.long)

    def update_conf_refine(self, notinpool_idxs, notinpool_uncs, shrink_f=1.0):
        not_inpool_num = 0
        safe_range_num = 0
        clean_num = 0
        pool_unc_avgs = []
        unsafe_feat_weight = torch.ones(self.conf.shape[0], dtype=torch.float16, device=self.device)
        conf_type_ = self.losstype.split('_')[0]
        if conf_type_ == 'lw':
            conf_torevise = self.conf_true
        else:
            conf_torevise = self.conf

        for pool_id, cur_pool in self.cls_pools_dict.items():
            pass
            # if cur_pool.pool_capacity == 0:
            #     pool_unc_avgs.append(torch.full((1,), torch.nan, dtype=torch.float16))
            #     continue
            # elif cur_pool.pool_capacity <= self.cfg.MAX_POOLNUM * self.cfg.POOL_INITRATIO: 
            #     unc_norm = 0
            # else:
            #     unc_min = cur_pool.pool_unc.min()
            #     unc_norm = (cur_pool.pool_unc - unc_min) / (cur_pool.unc_max - unc_min)
            # # if isinstance(unc_norm, torch.Tensor) and (torch.isnan(unc_norm)).any():
            # #     unc_norm = 0
            # pool_unc_avgs.append(cur_pool.pool_unc.mean().unsqueeze(0).cpu())
            
            # # 1. cal increament and update conf:
            # # if conf_type_ == 'cc':
            # #     pred_conf_value = conf_torevise[cur_pool.pool_idx, pool_id]       
            # #     conf_increment = (1 - self.conf_momn) * (1 - unc_norm * shrink_f)       #(1-unc_norm) is cern norm
            # #     pred_conf_value_ = pred_conf_value * self.conf_momn + conf_increment
            # #     base_value = pred_conf_value_ - pred_conf_value + 1
            # #     # assert (base_value >= 1).all(), 'base_value should larger than 1'    
            # #     conf_torevise[cur_pool.pool_idx, pool_id] = pred_conf_value_
            # max_val, max_idx = torch.max(conf_torevise[cur_pool.pool_idx, :], dim=1)
            # current_val = conf_torevise[cur_pool.pool_idx, pool_id]
            # # assert (current_val > self.eps).all(), 'current_val should not be 0'
            # if conf_type_ == 'cav':
            #     revised_conf = 1.0
            #     revised_max_conf = 0.           #self.conf_momn = 0.1 0.2
            # elif conf_type_ == 'cc':
            #     revised_conf = (current_val+max_val)/2 + (self.conf_momn * (unc_norm * shrink_f))
            #     revised_max_conf = (current_val+max_val)/2     

            # elif conf_type_ in ['rc', 'lw']:
            #     revised_conf = (current_val+max_val)/2 * (1 + self.conf_momn * (unc_norm * shrink_f)) #NOTE remember change back here
            #     revised_max_conf = (current_val+max_val)/2 
            # else:
            #     raise ValueError('conf_type_ not supported')
            # # if torch.isnan(conf_torevise[cur_pool.pool_idx, :]).any():
            # #     print(f'nan in conf, pool_id: {pool_id}')
            # #     print(f'current_val: {current_val}, max_val: {max_val}, unc_norm: {unc_norm}, revised_conf: {revised_conf}')
            # #     print(f'conf_torevise[cur_pool.pool_idx, :]: {conf_torevise[cur_pool.pool_idx, :]}')
            # #     raise ValueError('nan in conf')
            # conf_torevise[cur_pool.pool_idx, max_idx] = revised_max_conf
            # conf_torevise[cur_pool.pool_idx, pool_id] = revised_conf

        # 2. clean poped items which are not in safe range:
        is_inf = torch.isinf(notinpool_uncs)
        if notinpool_idxs.shape[0] - is_inf.sum() > 1:
            # safe_range = ((cur_pool.popped_unc - self.safe_f*cur_pool.unc_max) <= 0)
            # conf_torevise[cur_pool.popped_idx[~safe_range], :] = \
            #             self.origin_labels[cur_pool.popped_idx[~safe_range], :]
            # safe_range_num += safe_range.sum().item()
            # clean_num += (~safe_range).sum().item()
            # if conf_type_ == 'rc' or conf_type_ == 'cav' or conf_type_ == 'cc':  
            unc_notinpool_min = notinpool_uncs.min()
            unc_notinpool_max = notinpool_uncs[~is_inf].max()
            notinpool_uncs[is_inf] = unc_notinpool_max
            cern_norm = -(notinpool_uncs - unc_notinpool_min) / (unc_notinpool_max - unc_notinpool_min) + 1.0        #TOorg: 0/torch.inf == 0
            # if torch.isnan(cern_norm).any() or torch.isinf(cern_norm).any():
            #     print(f'notinpool_uncs: {notinpool_uncs}')
            #     print(f'unc_notinpool_min: {unc_notinpool_min}, unc_notinpool_max: {notinpool_uncs.max()}')
            #     print(f'cern_norm: {cern_norm}')
            #     raise ValueError('nan in cern_norm') 
            
            unsafe_feat_weight[notinpool_idxs] = cern_norm * self.cfg.HALF_USE_W 
            if conf_type_ == 'cc':   
                sorted_cern_norm, sorted_indices = torch.sort(cern_norm)
                num_select = int((1-self.cfg.HALF_USE_W) * notinpool_idxs.shape[0])
                selected_indices = sorted_indices[:num_select:]
                reduce_idxs = notinpool_idxs[selected_indices]
                conf_torevise[reduce_idxs, :] = self.origin_labels[reduce_idxs, :]
                # conf_torevise[notinpool_idxs, :] = self.origin_labels[notinpool_idxs, :]
        else:
            unsafe_feat_weight[notinpool_idxs] = 0.0

        # pool_unc_avgs = torch.cat(pool_unc_avgs, dim=0)
        # nan_mask = torch.isnan(pool_unc_avgs)
        # idx = torch.nonzero(nan_mask)
        # pool_unc_avgs[idx] = pool_unc_avgs[~nan_mask].max()

        # max_unc = pool_unc_avgs.max()
        # min_unc = pool_unc_avgs.min()
        # pool_unc_norm =  - (pool_unc_avgs - min_unc) / (max_unc - min_unc) + 1.0

        not_inpool_num = len(notinpool_idxs) #safe_range_num + clean_num
        info = {
            'not_inpool_num': not_inpool_num,
            'safe_range_num': safe_range_num,
            'clean_num': clean_num,
            'unsafe_feat_weight': unsafe_feat_weight,
        }
        return info #pool_unc_norm


    def clean_conf(self):
        if hasattr(self, 'conf') and 'refine' in self.losstype:
            if self.losstype.split('_')[0] == 'lw':
                mask = ((0 < self.conf_true) & (self.conf_true < 1/self.conf_true.shape[1]))        # TODO change the position of clean conf and make it correct with self.conf_weight
                self.conf_true = torch.where(mask, 
                                            torch.zeros_like(self.conf_true), 
                                            self.conf_true)
                base_value = self.conf_true.sum(dim=1).unsqueeze(1).repeat(1, self.conf_true.shape[1])
                self.conf_true = self.conf_true / base_value
                self.conf = torch.where((self.conf_true < self.eps) & (~mask), self.conf, self.conf_true) 
            else:
                self.conf = torch.where(self.conf < (1/self.conf.shape[1]), 
                                        torch.zeros_like(self.conf), 
                                        self.conf)
                base_value = self.conf.sum(dim=1).unsqueeze(1).repeat(1, self.conf.shape[1])
                self.conf = self.conf / base_value
            base_value = self.conf_weight.sum(dim=1).unsqueeze(1).repeat(1, self.conf.shape[1])
            self.conf_weight = self.conf_weight / base_value
            
            self.conf = (1 - self.conf_momn)*self.conf + self.conf_momn*self.conf_weight        #TODO check here, and lw_loss may be hacked


    def log_conf(self, all_logits=None, all_labels=None):
        log_id = 'PLL' + str(self.cfg.PARTIAL_RATE)
        if self.num % 1 == 0 and self.num != 50:        #50 is the last epoch (test dataset)
            print(f'save logits -> losstype: {self.losstype}, save id: {self.num}')
            if self.losstype == '_refine':      # need to run 2 times for getting conf
                if not hasattr(self, 'save_type'):
                    self.save_type = f'cc_refine'

                torch.save(self.conf, f'analyze_result_temp/logits&labels_10.14/conf-{self.save_type}_{log_id}-{self.num}.pt')
                torch.save(self.pred_label_dict, f'analyze_result_temp/logits&labels_10.14_pool/pred_label-{self.save_type}_{log_id}-{self.num}.pt')
                torch.save(self.gt_label_dict, f'analyze_result_temp/logits&labels_10.14_pool/gt_label-{self.save_type}_{log_id}-{self.num}.pt')
                self.save_label_pools()
                self.pred_label_dict = defaultdict(list)
                self.gt_label_dict = {}

            elif True:
                # assert self.losstype == 'rc_rc' or self.losstype == 'ce' or self.losstype == 'rc_refine'
                all_logits = self.origin_labels
                # labels_true = self.cls_pools_dict[0].labels_true
                torch.save(all_logits,  f'analyze_result_temp/logits&labels_11.11_ssucf101/outputs_{self.losstype.upper()+self.cfg.CONF_LOSS_TYPE}_{log_id}-{self.num}.pt')
                # torch.save(labels_true,  f'analyze_result_temp/logits&labels_11.11_ssucf101/labels_true_{self.losstype.upper()+self.cfg.CONF_LOSS_TYPE}_{log_id}-0.pt')
       
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
