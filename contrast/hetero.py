import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class ModelHeteroMoCo(nn.Module):
    "A completely separate version"

    def __init__(self, encoder_q0, encoder_k0, encoder_q1, encoder_k1,
        dim=128, K=4096, m=0.99, T=0.1,
        bn_splits=8):

        super(ModelHeteroMoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        self.encoder_q0 = encoder_q0
        self.encoder_k0 = encoder_k0
        for param_q0, param_k0 in zip(self.encoder_q0.parameters(), self.encoder_k0.parameters()):
            param_k0.data.copy_(param_q0.data)    # initialize
            param_k0.requires_grad = False       # not update by gradient
        self.encoder_q1 = encoder_q1
        self.encoder_k1 = encoder_k1
        for param_q1, param_k1 in zip(self.encoder_q1.parameters(), self.encoder_k1.parameters()):
            param_k1.data.copy_(param_q1.data)    # initialize
            param_k1.requires_grad = False       # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        "Momentum update of the key encoder"
        for param_q0, param_k0 in zip(self.encoder_q0.parameters(), self.encoder_k0.parameters()):
            param_k0.data = param_k0.data * self.m + param_q0.data * (1 - self.m)
        for param_q1, param_k1 in zip(self.encoder_q1.parameters(), self.encoder_k1.parameters()):
            param_k1.data = param_k1.data * self.m + param_q1.data * (1 - self.m)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0   # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.t() # transpose
        ptr = (ptr + batch_size) % self.K   # move pointer

        self.queue_ptr[0] = ptr

    def contrastive_loss(self, x_q, encoder_q, x_k, encoder_k):
        "without BN"
        # compute query features
        q = encoder_q(x_q) # queries: N x C
        q = F.normalize(q, dim=1)   # already normalized

        # compute key features
        with torch.no_grad():   # no gradient to keys
            k = encoder_k(x_k)
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
            x1: a batch of data from one view
            x2: a batch of data from another view
        Output:
            loss
        """

        # update the key encoder
        with torch.no_grad():
            self._momentum_update_key_encoder()
        
        # compute loss
        loss_12, q1, k2 = self.contrastive_loss(x1, self.encoder_q0, x2, self.encoder_q1)
        loss_21, q2, k1 = self.contrastive_loss(x2, self.encoder_q1, x1, self.encoder_q0)
        loss = loss_12 + loss_21
        k = torch.cat([k1, k2], dim=0)
        
        self._dequeue_and_enqueue(k)

        return loss


class ModelPartHeteroMoCo(nn.Module):
    "A partly sharing version"
    pass
