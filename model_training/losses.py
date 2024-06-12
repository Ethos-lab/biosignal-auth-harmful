"""
Defining additional contrastive loss function(s).
Final model version ended up using SupCon and not SimCLR
SupCon loss taken from Supervised Contrastive Learning at https://github.com/HobbitLong/SupContrast/blob/master/losses.py#L11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

    

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.


        """
        device = features.device
        features = features.unsqueeze(1)

        assert len(features.shape) == 3

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # computer log prob
        exp_logits = torch.exp(logits) * logits_mask
        
        # Prev definition of log_prob problematic because exp_logits.sum(1) could be 0 where an ex doesn't have a pair 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        

        # compute mean of log-likelihood over positive
        # Some of this ends up nan because mask.sum(1) can be 0. Ie there can be samples that have no match in the entire batch
        # Fuzz the denom; shouldnt affect the result because num would still be 0
        ix_zeros = torch.where(mask.sum(1) == 0)[0]
        denom = mask.sum(1).index_fill_(0, ix_zeros, 1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / denom  # average per num cases we see


        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # So need to nanmean instead of mean, and hope we never get a batch with none at all 
        loss = loss.view(anchor_count, batch_size).mean()


        return loss



# Taken from SimCLR github repo...
class SimCLRLoss(nn.Module):

    def __init__(self, device, temperature=0.07):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.temperature = temperature

    def forward(self, features, labels):

        # Labels is a batch_size array with pat numbers 
        labels = F.one_hot(labels)
        labels = torch.matmul(labels, labels.T)

        device = features.device
        features = F.normalize(features, dim=1)  

        similarity_matrix = torch.matmul(features, features.T)
        similarity_matrix = similarity_matrix.squeeze(1).squeeze(1)  

        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        import pdb; pdb.set_trace()

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        logits = logits / self.temperature

        loss = self.criterion(logits, labels)
        
        return loss
