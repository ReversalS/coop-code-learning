import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelMoCo(nn.Module):

    # def __init__(self, base_encoder,
    def __init__(self, encoder_q, encoder_k,
        dim=128, K=4096, m=0.99, T=0.1, arch='ConvS2S', 
        bn_splits=8, symmetric=True):

        super(ModelMoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.symmetric = symmetric

        # create the encoder
        # self.encoder_q = base_encoder(feature_dim=dim, arch=arch, bn_splits=bn_splits)
        # self.encoder_k = base_encoder(feature_dim=dim, arch=arch, bn_splits=bn_splits)
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)    # initialize
            param_k.requires_grad = False       # not update by gradient
        
        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        "Momentum update of the key encoder"
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0   # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.t() # transpose
        ptr = (ptr + batch_size) % self.K   # move pointer

        self.queue_ptr[0] = ptr

    # @torch.no_grad()
    # def _batch_shuffle_single_gpu(self, x):
    #     "Batch shuffle, for making use of BatchNorm"
    #     # random shuffle index
    #     idx_shuffle = torch.randperm(x.shape[0]).cuda()

    #     # index for restoring
    #     idx_unshuffle = torch.argsort(idx_shuffle)

    #     return x[idx_shuffle], idx_unshuffle

    # @torch.no_grad()
    # def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
    #     "Undo batch shuffle."
    #     reutrn x[idx_unshuffle]
    
    # def contrastive_loss(self, x_q, x_k):
    #     # compute query features
    #     q = self.encoder_q(x_q) # queries: N x C
    #     q = F.normalize(q, dim=1)   # already normalized

    #     # compute key features
    #     with torch.no_grad():   # no gradient to keys
    #         # shuffle for making use of BN
    #         x_k_, idx_unshuffle = self._batch_shuffle_single_gpu(x_k)

    #         k = self.encoder_k(x_k_)
    #         k = F.normalize(k, dim=1)   # already normalized

    #         # undo shuffle
    #         k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)
        
    #     # compute loss
    #     # Einstein sum is more intuitive (expression)
    #     # positive logits: N x 1
    #     l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
    #     # negative logits: N x K
    #     l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()]).unsqueeze(-1)

    #     # logits: N x (1 + K)
    #     logits = torch.cat([l_pos, l_neg], dim=1)

    #     # apply temperature
    #     logits /= self.T

    #     # labels: positive key indicators
    #     labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

    #     loss = nn.CrossEntropyLoss().cuda()(logits, labels)

    #     return loss, q, k

    def contrastive_loss(self, x_q, x_k):
        "without BN"
        # compute query features
        q = self.encoder_q(x_q) # queries: N x C
        q = F.normalize(q, dim=1)   # already normalized

        # compute key features
        with torch.no_grad():   # no gradient to keys
            k = self.encoder_k(x_k)
            k = F.normalize(k, dim=1)   # already normalized
        
        # compute loss
        # Einstein sum is more intuitive (expression)
        # positive logits: N x 1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: N x K
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: N x (1 + K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        loss = nn.CrossEntropyLoss().cuda()(logits, labels)

        return loss, q, k


    def forward(self, x1, x2):
        """
        Input:
            x1: a batch of query ?
            x2: abtch of key ?
        Output:
            loss
        """

        # update the key encoder
        with torch.no_grad():
            self._momentum_update_key_encoder()
        
        # compute loss
        if self.symmetric:  # symmetric loss
            loss_12, q1, k2 = self.contrastive_loss(x1, x2)
            loss_21, q2, k1 = self.contrastive_loss(x2, x1)
            loss = loss_12 + loss_21
            k = torch.cat([k1, k2], dim=0)
        else:   # asymmetric loss
            loss, q, k = self.contrastive_loss(x1, x2)
        
        self._dequeue_and_enqueue(k)

        return loss