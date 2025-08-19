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
        # self.b1 = torch.randn(h, generator=self.generator, requires_grad=True)
        self.W1 = torch.randn((h,self.vocab_size), generator=self.generator, requires_grad=True)
        self.W2 = torch.randn((n*feature_dims,self.vocab_size), generator=self.generator, requires_grad=True)
        self.b2 = torch.randn(self.vocab_size, generator=self.generator, requires_grad=True)

        self.bn_epsilon = 1e-5
        self.H_bngain = torch.ones(h, requires_grad=True)
        self.H_bnbias = torch.zeros(h, requires_grad= True)
        self.H_bnmean_running = torch.zeros(h)
        self.H_bnstd_running = torch.ones(h)
        self.momentum:float = 0.999

        self.cross_entropy_loss = torch.tensor(0.0)

    @property
    def params(self)->list[torch.Tensor]:
        return [self.C, self.H, self.W1, self.W2, self.b2, self.H_bngain, self.H_bnbias]
    
    def _retain_intermediate_tensor_grads(self)->None:
        """
        Retain intermediate tensor gradients for debugging purposes.
        I'll be using it to compare the pytorch computes gradients of intermediate tensors,
        and my manual computaion of the graidents of the same intermediate tensors.
        """
        for param in [
            self.logprobs, self.probs, self.counts_sum_inv, self.counts_sum,
            self.counts, self.norm_logits, self.logit_maxes, self.logits,
            self.l1, self.l2, self.h, self.hpreact, self.H_bngain, self.H_bnbias,
            self.bnraw, self.bndiff, self.bnvar_inv, self.bnvar, self.bndiff2,
            self.bndiff, self.bnmean, self.hprebn, self.embcat, self.emb
        ]:
            param.retain_grad()

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
        batch_size = hprebn.shape[0]

        if run_type=='train':
            self.bnmean = 1/batch_size*self.hprebn.sum(0, keepdim=True)
            self.bndiff = self.hprebn - self.bnmean
            self.bndiff2 = self.bndiff**2
            self.bnvar = 1/(batch_size-1)*(self.bndiff2).sum(0, keepdim=True) # note: Bessel's correction (dividing by n-1, not n)
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

    def manual_backward_pass(self, x:torch.Tensor, y:torch.Tensor)->None:
        batch_size = x.shape[0]
        dlogprobs = torch.zeros([*self.logprobs.shape])
        dlogprobs[range(batch_size), y] += -1*(batch_size**-1)
        dprobs = dlogprobs*(1/self.probs)
        dcounts_sum_inv = (dprobs*self.counts).sum(1, keepdims=True)
        dcounts = dprobs*self.counts_sum_inv
        dcounts_sum = dcounts_sum_inv*(-1/self.counts_sum**2)
        dcounts = dcounts+ (dcounts_sum*1)
        dnorm_logits = dcounts*(self.norm_logits.exp())
        dlogit_maxes = dnorm_logits.sum(1, keepdim=True)*(-1)
        dlogits = torch.zeros([*self.logits.shape])
        dlogits[torch.arange(batch_size), self.logits.argmax(1)] += dlogit_maxes.view(-1)*1
        dlogits += dnorm_logits

        dl1, dl2 = dlogits.clone(), dlogits.clone()
        dh = dl1@self.W1.T
        dW1 = self.h.T @ dl1
        dembcat = dl2@self.W2.T
        dW2 = self.embcat.T@dl2
        db2 = (1*dl2).sum(0)
        dhpreact = (1-self.h**2)*dh

        dH_bngain = (dhpreact*self.bnraw).sum(0)
        dH_bnbias = (dhpreact*1).sum(0)
        dbnraw = dhpreact*self.H_bngain
        dbnvar_inv = (dbnraw*self.bndiff).sum(0, keepdim=True)
        dbndiff = dbnraw*self.bnvar_inv
        dbnvar = dbnvar_inv*(-0.5)*((self.bnvar + self.bn_epsilon)**-1.5)
        dbndiff2 = (1.0/(batch_size-1))*torch.ones_like(self.bndiff2)*dbnvar
        dbndiff += 2*self.bndiff*dbndiff2
        dhprebn = dbndiff.clone()
        dbnmean = -dbndiff.sum(0)
        dhprebn += (1/batch_size)*torch.ones_like(self.hprebn)*dbnmean
        dembcat = dembcat + dhprebn @ self.H.T
        dH = self.embcat.T @ dhprebn
        demb = dembcat.view(self.emb.shape)
        dC = torch.zeros_like(self.C)
        for i in range(len(x)):
            for j in range(self.n):
                dC[x[i,j]] += demb[i,j]
    
        cmp('logprobs', dlogprobs, self.logprobs)
        cmp('probs', dprobs, self.probs)
        cmp('counts_sum_inv', dcounts_sum_inv, self.counts_sum_inv)
        cmp('counts_sum', dcounts_sum, self.counts_sum)
        cmp('counts', dcounts, self.counts)
        cmp('norm_logits', dnorm_logits, self.norm_logits)
        cmp('logit_maxes', dlogit_maxes, self.logit_maxes)
        cmp('logits', dlogits, self.logits)
        cmp('l1', dl1, self.l1)
        cmp('l2', dl2, self.l2)
        cmp('h', dh, self.h)
        cmp('W1', dW1, self.W1)
        cmp('W2', dW2, self.W2)
        cmp('b2', db2, self.b2)
        cmp('hpreact', dhpreact, self.hpreact)
        cmp('dH_bngain', dH_bngain, self.H_bngain)
        cmp('dH_bnbias', dH_bnbias, self.H_bnbias)
        cmp('bnraw', dbnraw, self.bnraw)
        cmp('bnvar_inv', dbnvar_inv, self.bnvar_inv)
        cmp('bnvar', dbnvar, self.bnvar)
        cmp('bndiff2', dbndiff2, self.bndiff2)
        cmp('bndiff', dbndiff, self.bndiff)
        cmp('bnmean', dbnmean, self.bnmean)
        cmp('hprebn', dhprebn, self.hprebn)
        cmp('embcat', dembcat, self.embcat)
        cmp('H', dH, self.H)
        cmp('emb', demb, self.emb)
        cmp('C', dC, self.C)


    def manual_cross_entropy_loss(self, logits:torch.Tensor, y:torch.Tensor)->torch.Tensor:
        self.logit_maxes = logits.max(1, keepdim=True).values
        self.norm_logits = logits - self.logit_maxes # subtract max for numerical stability
        self.counts = self.norm_logits.exp()
        self.counts_sum = self.counts.sum(1, keepdims=True)
        self.counts_sum_inv = self.counts_sum**-1 # if I use (1.0 / counts_sum) instead then I can't get backprop to be bit exact...
        self.probs = self.counts * self.counts_sum_inv
        self.logprobs = self.probs.log()
        loss = -self.logprobs[range(logits.shape[0]), y].mean()
        return loss

    def _training_code(self, x:torch.Tensor, y:torch.Tensor, h:float, reg_factor:float, optim_type:str)->None:
        #zero grad
        for param in self.params:
            param.grad = None

        #forward pass
        logits = self.forward(x, optim_type, 'train')

        #manual loss computation
        # self.cross_entropy_loss = F.cross_entropy(logits, y, reduction='mean', label_smoothing=reg_factor)
        self.cross_entropy_loss = self.manual_cross_entropy_loss(logits, y)

        #retain grads of intermediate tensors
        self._retain_intermediate_tensor_grads()

        #pytorch backward pass
        self.cross_entropy_loss.backward()

        #manual backward pass
        self.manual_backward_pass(x,y)

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

def cmp(name:str, dt:torch.tensor, t:torch.Tensor)->None:
    """
    Utility function for comparing manual gradients to PyTorch gradients
    """
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    assert ex or app, f"Gradients for {name} do not match! Max difference: {maxdiff}"