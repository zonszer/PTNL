import math
import torch.nn.functional as F
import torch
import torch.nn as nn

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    if torch.cuda.device_count() > 1:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cuda')
else:
    device = torch.device('cpu')


class PLL_loss(nn.Module):
    def __init__(self, type=None, PartialY=None,
                 eps=1e-6):
        super(PLL_loss, self).__init__()
        self.eps = eps
        self.losstype = type
        self.device = device
        self.softmax = nn.Softmax(dim=1)
        #PLL items: 
        if type == 'rc':
            self.confidence = self.init_confidence(PartialY)
            self.num = 0
        if type == 'gce':
            self.q = 0.7

    def init_confidence(self, PartialY):
        tempY = PartialY.sum(dim=1, keepdim=True).repeat(1, PartialY.shape[1])   #repeat train_givenY.shape[1] times in dim 1
        confidence = PartialY.float()/tempY
        confidence = confidence.to(self.device)
        return confidence
    
    def forward(self, *args):
        """"
        x: outputs logits
        y: targets (multi-label binarized vector)
        """
        if self.losstype == 'cc':
            loss = self.forward_cc(*args)
        elif self.losstype == 'ce':
            loss = self.forward_ce(*args)
        elif self.losstype == 'gce':
            loss = self.forward_gce(*args)
        elif self.losstype == 'rc':
            loss = self.forward_rc(*args)
        else:
            raise ValueError
        return loss

    def forward_gce(self, x, y, index):
        """y is shape of (batch_size, num_classes (0 ~ 1.)), one-hot vector"""
        p = F.softmax(x, dim=1)      #outputs are logits
        # Create a tensor filled with a very small number to represent 'masked' positions
        masked_p = p.new_full(p.size(), float('-inf'))
        # Apply the mask
        masked_p[y.bool()] = p[y.bool()] + self.eps         #TOorg : 这行代码保证矩阵对应位置的赋值
        # Adjust masked positions to avoid undefined gradients by adding epsilon
        masked_p[y.bool()] = (1 - masked_p[y.bool()] ** self.q) / self.q
        masked_p[~y.bool()] = self.eps  #TOorg : 类似torch.nonzero
        loss = (masked_p.sum(dim=1) / y.sum(dim=1)).mean()
        return loss
    
    def forward_cc(self, x, y, index):
        sm_outputs = F.softmax(x, dim=1)      #outputs are logits
        final_outputs = sm_outputs * y
        average_loss = - torch.log(final_outputs.sum(dim=1) / y.sum(dim=1)).mean()     #NOTE: add y.sum(dim=1)
        return average_loss
    
    def forward_ce(self, x, y, index):
        sm_outputs = F.log_softmax(x, dim=1)
        final_outputs = sm_outputs * y
        average_loss = - (final_outputs.sum(dim=1) / y.sum(dim=1)).mean()  #NOTE: add y.sum(dim=1)
        return average_loss
    
    def forward_rc(self, x, y, index):
        logsm_outputs = F.log_softmax(x, dim=1)         #x is the model ouputs
        final_outputs = logsm_outputs * self.confidence[index, :]
        average_loss = - ((final_outputs).sum(dim=1)).mean()    #final_outputs=torch.Size([256, 10]) -> Q: why use negative? A:  
        self.confidence_update(self.confidence, x, y, index)
        return average_loss     

    def forward_rc_(self, x, y, index):
        logsm_outputs = F.softmax(x, dim=1)         #x is the model ouputs
        final_outputs = logsm_outputs * self.confidence[index, :]
        average_loss = - torch.log((final_outputs).sum(dim=1) / y.sum(dim=1)).mean()    #final_outputs=torch.Size([256, 10]) -> Q: why use negative? A:  
        self.confidence_update(self.confidence, x, y, index)
        return average_loss     

    def confidence_update(self, confidence, batch_outputs, batchY, batch_index):
        with torch.no_grad():
            temp_un_conf = F.softmax(batch_outputs, dim=1)
            confidence[batch_index, :] = temp_un_conf * batchY # un_confidence stores the weight of each example
            #weight[batch_index] = 1.0/confidence[batch_index, :].sum(dim=1)
            base_value = confidence.sum(dim=1).unsqueeze(1).repeat(1, confidence.shape[1])
            self.confidence = confidence/base_value  # use maticx for element-wise division

            if self.num % 100 == 0:
                torch.save(self.confidence, f'analyze_result_temp/confidence_RC-{self.num}_init.pt')
                # torch.load('analyze_result_temp/confidence_RC-0.pt')
            self.num += 1


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