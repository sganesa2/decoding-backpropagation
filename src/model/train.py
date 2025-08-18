from typing import Literal

import torch
import torch.nn.functional as F

class BatchNormalizedMLP:
    """
    This is a 3-layer MLP with the following required arguments:
        h: Total number of neurons in the hidden layer
        n: Number of characters provided as context
    """
    def __init__(self, h:int, n:int, feature_dims:int)->None:
        self.h = h
        self.n = n
        self.feature_dims = feature_dims
        self.vocab_size = 27 #26 alphabets + 1 special token(".")
        self.generator = torch.Generator().manual_seed(6385189022)

        self.C = torch.randn((self.vocab_size, feature_dims), generator=self.generator, requires_grad= True)
        self.H = torch.randn((n*feature_dims,h), generator=self.generator, requires_grad=True)
        self.b1 = torch.randn(h, generator=self.generator, requires_grad=True)
        self.W1 = torch.randn((h,self.vocab_size), generator=self.generator, requires_grad=True)
        self.W2 = torch.randn((n*feature_dims,self.vocab_size), generator=self.generator, requires_grad=True)
        self.b2 = torch.randn(self.vocab_size, generator=self.generator, requires_grad=True)

        self.bn_epsilon = 1e-5
        self.H_bngain = torch.ones(h, requires_grad=True)
        self.H_bnbias = torch.zeros(h, requires_grad= True)
        self.H_bnmean_running = torch.zeros(h)
        self.H_bnstd_running = torch.ones(h)
        self.momentum:float = 0.999

        self.pytorch_computed_loss = torch.tensor(0)
        self.manually_computed_loss = torch.tensor(0)

    @property
    def params(self)->list[torch.Tensor]:
        return [self.C, self.H, self.W1, self.W2, self.b2, self.H_bngain, self.H_bnbias]

    def _kaiming_init_all_weights(self)->None:
        """
        IF NEEDED
        Since I'm using tanh() act func across all hidden layer neurons my kaiming init factor is: (5/3)*1/sqrt(fan_in)
        """
        self.H.data *= (5/3)*(1/self.n*self.feature_dims)
    
    def _squash_op_layer_params(self)->None:
        """
        Minimize the output layer parameters by a factor b/w [0,1]
        to ensure that the logits produced are close to zero(at initialiation our network makes no assumptions)
        """
        self.W1.data *= 0.01
        self.W2.data *= 0.01
        self.b2.data *= 0

    def _batchnormalized(self, hprebn:torch.Tensor, flag:bool, run_type:Literal['train','inference'])->None:
        """
        LESSON LEARNT: With a NoneType return you will definitely lose the child nodes of hpreact
        that are added within this function. if you want to retain child nodes, return hpreact.
        """
        if not flag: return

        self.hprebn = hprebn

        if run_type=='train':
            self.bnmean = 1/self.n*self.hprebn.sum(0, keepdim=True)
            self.bndiff = self.hprebn - self.bnmean
            self.bndiff2 = self.bndiff**2
            self.bnvar = 1/(self.n-1)*(self.bndiff2).sum(0, keepdim=True) # note: Bessel's correction (dividing by n-1, not n)
            self.bnvar_inv = (self.bnvar + self.bn_epsilon)**-0.5
            with torch.no_grad():
                self.H_bnmean_running = self.momentum*self.H_bnmean_running + (1-self.momentum)*self.bnmean
                self.H_bnstd_running = self.momentum*self.H_bnstd_running + (1-self.momentum)*(self.bnvar_inv)**-1
        else:
            self.bnmean, self.bnstd = self.H_bnmean_running, self.H_bnstd_running
            self.bndiff, self.bnvar_inv = self.hprebn - self.bnmean, self.bnstd**-1

        self.bnraw = self.bndiff * self.bnvar_inv
        self.hpreact = self.H_bngain * self.bnraw + self.H_bnbias

        return self.hpreact
    
    def forward(self, xs:torch.Tensor, optim_type:str, run_type:Literal['train','inference'])->torch.Tensor:
        flag = True if optim_type=='minibatch_gradient_descent' else False
            
        self.emb = self.C[xs]
        self.embcat = self.emb.view(-1,self.n*self.feature_dims)
        hprebn = self.embcat@self.H
        hpreact = self._batchnormalized(hprebn, flag, run_type)
        self.h = hpreact.tanh()

        self.l1 = self.h@self.W1
        self.l2 = self.embcat@self.W2 + self.b2
        self.logits = self.l1+self.l2
        return self.logits

    def cmp(self, name:str, dt:torch.tensor, t:torch.Tensor)->None:
        """
        Utility function for comparing manual gradients to PyTorch gradients
        """
        ex = torch.all(dt == t.grad).item()
        app = torch.allclose(dt, t.grad)
        maxdiff = (dt - t.grad).abs().max().item()
        assert ex or app, f"Gradients for {name} do not match! Max difference: {maxdiff}"

    # def _manual_backward_pass(self, x:torch.Tensor, y:torch.Tensor)->None:
    #     dlogprobs = torch.zeros([*logprobs.shape])
    #     dlogprobs[range(n), Yb] += -1*(n**-1)
    #     dprobs = dlogprobs*(1/probs)
    #     dcounts_sum_inv = (dprobs*counts).sum(1, keepdims=True)
    #     dcounts = dprobs*counts_sum_inv
    #     dcounts_sum = dcounts_sum_inv*(-1/counts_sum**2)
    #     dcounts = dcounts+ (dcounts_sum*1)
    #     dnorm_logits = dcounts*(norm_logits.exp())
    #     dlogit_maxes = dnorm_logits.sum(1, keepdim=True)*(-1)
    #     dlogits = torch.zeros([*logits.shape])
    #     dlogits[torch.arange(n), logits.argmax(1)] += dlogit_maxes.view(-1)*1
    #     dlogits += dnorm_logits

    #     db2 = (dlogits*1).sum(0) 
    #     dW2 = torch.zeros([*W2.shape])
    #     for i in range(n):
    #     dW2 += dlogits[i].view(1,-1)*h[i].view(-1,1)
    #     dh = dlogits@W2.T
    #     dhpreact = dh*(1-h**2)
    #     dbngain = (dhpreact*bnraw).sum(0)
    #     dbnbias = (dhpreact*1).sum(0)
    #     dbnraw = dhpreact*bngain
    #     dbnvar_inv = (dbnraw*bndiff).sum(0, keepdim=True)
    #     dbndiff = dbnraw*bnvar_inv
    #     dbnvar = dbnvar_inv*(-0.5)*((bnvar + 1e-5)**-1.5)
    #     dbndiff2 = (1.0/(n-1))*torch.ones_like(bndiff2)*dbnvar
    #     dbndiff += 2*bndiff*dbndiff2
    #     dhprebn = dbndiff.clone()
    #     dbnmeani = -dbndiff.sum(0)
    #     dhprebn += (1/n)*torch.ones_like(hprebn)*dbnmeani
    #     dembcat = dhprebn @ W1.T
    #     dW1 = embcat.T @ dhprebn
    #     db1 = dhprebn.sum(0)
    #     demb = dembcat.view(emb.shape)
    #     dC = torch.zeros_like(C)
    #     for i in range(len(Xb)):
    #     for j in range(block_size):
    #         dC[Xb[i,j]] += demb[i,j]



    #     self.cmp('logprobs', dlogprobs, logprobs)
    #     self.cmp('probs', dprobs, probs)
    #     self.cmp('counts_sum_inv', dcounts_sum_inv, counts_sum_inv)
    #     self.cmp('counts_sum', dcounts_sum, counts_sum)
    #     self.cmp('counts', dcounts, counts)
    #     self.cmp('norm_logits', dnorm_logits, norm_logits)
    #     self.cmp('logit_maxes', dlogit_maxes, logit_maxes)
    #     self.cmp('logits', dlogits, logits)
    #     self.cmp('h', dh, h)
    #     self.cmp('W2', dW2, W2)
    #     self.cmp('b2', db2, b2)
    #     self.cmp('hpreact', dhpreact, hpreact)
    #     self.cmp('bngain', dbngain, bngain)
    #     self.cmp('bnbias', dbnbias, bnbias)
    #     self.cmp('bnraw', dbnraw, bnraw)
    #     self.cmp('bnvar_inv', dbnvar_inv, bnvar_inv)
    #     self.cmp('bnvar', dbnvar, bnvar)
    #     self.cmp('bndiff2', dbndiff2, bndiff2)
    #     self.cmp('bndiff', dbndiff, bndiff)
    #     self.cmp('bnmeani', dbnmeani, bnmeani)
    #     self.cmp('hprebn', dhprebn, hprebn)
    #     self.cmp('embcat', dembcat, embcat)
    #     self.cmp('W1', dW1, W1)
    #     self.cmp('b1', db1, b1)
    #     self.cmp('emb', demb, emb)
    #     self.cmp('C', dC, C)

    def _cross_entropy_loss_manual(self, logits:torch.Tensor, y:torch.Tensor)->torch.Tensor:
        self.logit_maxes = logits.max(1, keepdim=True).values
        self.norm_logits = logits - self.logit_maxes # subtract max for numerical stability
        self.counts = self.norm_logits.exp()
        self.counts_sum = self.counts.sum(1, keepdims=True)
        self.counts_sum_inv = self.counts_sum**-1 # if I use (1.0 / counts_sum) instead then I can't get backprop to be bit exact...
        self.probs = self.counts * self.counts_sum_inv
        self.logprobs = self.probs.log()
        loss = -self.logprobs[range(self.n), y].mean()
        return loss

    def _training_code(self, x:torch.Tensor, y:torch.Tensor, h:float, reg_factor:float, optim_type:str)->None:
        #zero grad
        for param in self.params:
            param.grad = None

        #forward pass
        logits = self.forward(x, optim_type, 'train')

        #pytorch loss computation
        self.pytorch_computed_loss = F.cross_entropy(logits, y, reduction='mean', label_smoothing=reg_factor)

        #pytorch backward pass
        self.pytorch_computed_loss.backward()

        #manual loss computation
        self.manually_computed_loss = self._cross_entropy_loss_manual(logits, y)

        #manual backward pass
        self._manual_backward_pass(x,y)

        #grad update
        for param in self.params:
            param.data -= h*param.grad

    def gradient_descent(self, x_train:torch.Tensor, y_train:torch.Tensor, epochs:int, h:float, reg_factor:float)->None:
        #optional weights initializations done
        # self._kaiming_init_all_weights()
        self._squash_op_layer_params()

        for _ in range(epochs):
            self._training_code(x_train,y_train,h,reg_factor, "gradient_descent")
    
    def stochastic_gradient_descent(self, x_train:torch.Tensor, y_train:torch.Tensor, epochs:int, h:float, reg_factor:float)->None:
        # self._kaiming_init_all_weights()
        self._squash_op_layer_params()

        for _ in range(epochs):
            for example,label in zip(x_train, y_train):
                self._training_code(example,label,h,reg_factor, "stochastic_gradient_descent")


    def minibatch_gradient_descent(self, minibatch_size:int, x_train:torch.Tensor, y_train:torch.Tensor, epochs:int, h:float, reg_factor:float)->None:
        self._kaiming_init_all_weights()
        self._squash_op_layer_params()

        permutes = torch.randperm(x_train.shape[0], generator=self.generator)
        x_train, y_train = x_train[permutes], y_train[permutes]
        x_train_minibatches, y_train_minibatches = x_train.split(minibatch_size), y_train.split(minibatch_size)

        for _ in range(epochs):
            for x,y in zip(x_train_minibatches,y_train_minibatches):
                self._training_code(x,y,h,reg_factor, "minibatch_gradient_descent")

def stoi()->dict[str,int]:
    start_index, total_chars = 97, 26
    stoi_dict = {chr(i):i-start_index+1 for i in range(start_index, start_index+total_chars+1)}
    return {".":0, **stoi_dict}

def itos()->dict[int,str]:
    stoi_dict = stoi()
    return {v:k for k,v in stoi_dict.items()}