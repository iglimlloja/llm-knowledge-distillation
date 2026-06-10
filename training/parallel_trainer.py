# Imports
import torch.multiprocessing as mp

import os, pickle, torch
from tqdm import tqdm
import numpy as np
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time

class quantizer_4b_torch:
    def quantize(self,a): # it is assumed that the distribution of the values is normal
        is_negative=torch.sign(a)<0
        
        x=torch.sqrt(torch.abs(a))*5.3-2.1
        x=torch.round(x)
        x=torch.clip(x,0,7).to(torch.int8)
        
        x=x+8*is_negative.to(torch.int8)
        return x
    
    def decode(self,x):
        with torch.no_grad():
            is_negative=x>=8
            a=(( torch.remainder(x,8).to(torch.float16)+2.1)/5.3)**2
            a=a*(1-is_negative.to(torch.int8)*2)
            return a
    
    def quantize_compress(self,a):
        ind=self.quantize(a)
        comp=ind[:,0::2]*16+ind[:,1::2]
        return comp.to(torch.uint8)

    def load_decode(self,data):
        with torch.no_grad():
            ind=torch.zeros((data.shape[0],data.shape[1]*2),dtype=torch.uint8,device=data.device)
            ind[:,0::2]=torch.floor_divide(data,16)
            ind[:,1::2]=torch.remainder(data,16)
            return self.decode(ind)

class representation_whitening_torch:
    
    def fit(self,a):
        self.mn=a.mean(dim=0)
        
        # low memory covariance calculation
        cov=torch.eye(a.shape[1],dtype=torch.float32,device=a.device)*0
        for i in tqdm(range(0,a.shape[0],10000)):
            aslice=(a[i:i+10000,:].to(torch.float32)-self.mn)
            cov+=( aslice.T@aslice)/a.shape[0]
        
        
        s2,self.q=torch.linalg.eigh(cov)
        self.q=self.q.to(torch.float16)
        self.s=torch.sqrt(s2).to(torch.float16)
        
        if self.s[0]<self.s[-1]:
            self.s=torch.flip(self.s,dims=[0])
            self.q=torch.flip(self.q,dims=[1])
       
    def encode(self,a):
        return (a-self.mn)@self.q /self.s
    
    def decode(self,e):
        with torch.no_grad():
            return (self.s*e)@self.q.T+self.mn
    
    def rescale(self,e):
        return self.s*e
    
    def save_state_dict(self,path):
        torch.save({'mn':self.mn,'q':self.q,'s':self.s}, path)

    def load_state_dict(self,path):
        with torch.no_grad():
            data=torch.load(path)
            self.mn=data['mn']
            self.q =data['q' ]
            self.s =data['s' ]

    def to(self, d):
        with torch.no_grad():
            self.mn = self.mn.to(d)
            self.q = self.q.to(d)
            self.s = self.s.to(d)
        
class LoggingManager():
    def __init__(self, tokens, log_step):
        self.tokens = tokens
        self.token_index_map = {k:v for v,k in enumerate(tokens)}
        self.log_step = log_step
        
        self.steps_taken = {tok.item():0 for tok in self.tokens}  # This counts the number of (possibly incomplete) batches that go through forward pass
        self.items_seen  = {tok.item():0 for tok in self.tokens}  # This counts the number of examples accumulated
        
        self.batch_losses = {tok.item():0 for tok in self.tokens}
        self.default_losses = {tok.item():0 for tok in self.tokens}
    
    def accumulate(self, configuration, token_id, no_items, batch_loss, default_loss, writer, train=True):
        return
        log_step = self.log_step
        self.batch_losses[token_id] += batch_loss
        self.default_losses[token_id] += default_loss
        self.items_seen[token_id] += no_items 
        if train:
            self.steps_taken[token_id] += 1
        if train and self.steps_taken[token_id] % log_step == 0:
            self.log(configuration, token_id, writer, train)
            
    def log(self, configuration, token_id, writer, train):
        loss = configuration['loss']
        if train:
            # writer.add_scalar(f'{loss.upper()}/Training Average Batch Loss', self.batch_losses[token_id]/(self.items_seen[token_id]), self.steps_taken[token_id])
            writer.add_scalar(f'{loss.upper()}/Training Batch Loss Ratio', self.batch_losses[token_id]/self.default_losses[token_id], self.steps_taken[token_id])
        else:
            self.steps_taken[token_id] += 1
            # writer.add_scalar(f'{loss.upper()}/Validation Average Batch Loss', self.batch_losses[token_id]/(self.items_seen[token_id]), self.steps_taken[token_id])
            writer.add_scalar(f'{loss.upper()}/Validation Batch Loss Ratio', self.batch_losses[token_id]/self.default_losses[token_id], self.steps_taken[token_id])
        
        self.batch_losses[token_id] = 0
        self.default_losses[token_id] = 0
        self.items_seen[token_id] = 0 

"""
    Utility class to index into the bookkeeper to get the correct logprobs
"""
class TokenIndexer():
    def __init__(self, ids, toi):
        with torch.no_grad():
            assert len(ids.shape) == 2 and ids.shape[1] == 10, f"ids not in the correct shape, got {ids.shape} expected (n , 10)"
            self.ids = ids    
            self.toi = toi
            self.prepare()
        
    def prepare(self):
        # Indexing into the first 10 columns : n x 10
        with torch.no_grad():
            vals, idx = torch.sort(self.ids.reshape(-1))
            vv, vc = torch.unique(vals, return_counts=True)
            begin = torch.zeros_like(vv)
            end = torch.zeros_like(vv)
            cs = torch.cumsum(vc, dim= 0)
            begin[1:] = cs[:-1]
            end[:] = cs
    
            is_toi = torch.zeros(256000, dtype=torch.bool, device=self.toi.device)
            is_toi[self.toi] = True
            select = is_toi[vv]
            # assert select.any(), "None of the tokens here are tokens of interest"
            vv = vv[select]
            begin = begin[select]
            end = end[select]
    
            self.valid_tokens = vv
            self.idx1 = idx // 10
            self.idx2 = idx % 10
            self.begin = begin
            self.end = end
            self.invert_map = {v.item():k for k,v in enumerate(vv)}
    
    def get_train_idx(self, token_id):
        with torch.no_grad():
            if type(token_id) is int:
                position = self.invert_map.get(token_id, None)
            elif torch.is_tensor(token_id):
                position = self.invert_map.get(token_id.item(), None)
            else:
                assert False, "Wrong parameter type passed to TokenIndexer.get_train_idx for token_id"
            if position is None:
                assert False, "Key not in invert_map"
            begin = self.begin[position]
            end = self.end[position]
            idx1 = self.idx1[begin:end]
            idx2 = self.idx2[begin:end]
    
            return idx1, idx2

"""
    Main data structure that keeps track of the student information during training

    Implements two main methods:
    * BookKeeper.get
    * BookKeeper.update
"""
class BookKeeper():
    def __init__(self, student_probs):
        with torch.no_grad():
            self.student_logprobs = (student_probs.float()).log() # Shape (num_examples, 100)
            #assert torch.isnan(self.student_logprobs).any() == False
            # Denominators of softmax
            self.S = torch.exp(self.student_logprobs).sum(dim=-1).float()          # Shape (num_examples)
            #assert torch.isnan(self.S).any() == False
            # This can be used if we have asynchronous updates to the sums (i.e. train all tokens with no communication)
            self.lazy_deltas = torch.zeros_like(self.S)
            self.kl_divs = torch.zeros_like(self.S)
            self.raw_kl_divs = torch.zeros_like(self.kl_divs)
            
    def update_kl_divs(self, example_id, teacher_prob=None, teacher_logprob=None):
        with torch.no_grad():
            if teacher_prob is not None and teacher_logprob is not None:
                q = torch.exp(self.student_logprobs[example_id])
                q = q / q.sum(dim=-1, keepdims=True)
                logq = q.log()
                new_kls = (teacher_prob * (teacher_logprob - logq)).sum(dim=-1)
                self.raw_kl_divs[example_id] = (teacher_prob * (teacher_logprob - self.student_logprobs[example_id] +   self.S[example_id].reshape(-1,1).log()  )).sum(dim=-1)
                self.raw_kl_divs[example_id] = new_kls
        
    # def lazy_update(self, ix, y_hat, new_vals): # fix this function
    #     old_logprobs = self.student_logprobs[i, j]
    #     old_logprobs += y_hat
    #     self.lazy_deltas[ix] += new_vals

    """
        Given example id(s) and rank(s), of the same shape, returns the data needed for
        the forward pass in the trainer
    """
    def get(self, example_id, rank):
        with torch.no_grad():
            out = {
                'logprobs' : self.student_logprobs[example_id,rank],
                'S'        : self.S[example_id],
                'kl'       : self.kl_divs[example_id]
            }
            out['logprobs'] -= out['S'].log() # Normalization step
            return out
    def normalize_S(self):
        with torch.no_grad():
            self.student_logprobs -= self.S.log().reshape(-1,1)
            self.S[:] = 1
        
    def update_old(self, example_id,rank, y_hat, tp):        
        old_logprobs = self.student_logprobs[example_id, rank]
        old_S=self.S[example_id]
        
        self.S[example_id] =  self.S[example_id] - (torch.exp(old_logprobs) * (1 - torch.exp(y_hat)))
        new_S=self.S[example_id]
        
        self.kl_divs[example_id]= self.kl_divs[example_id] - tp[example_id,rank]*y_hat - old_S.log() + new_S.log()
        self.student_logprobs[example_id, rank] += y_hat
    
    def update(self, example_id, rank, y_hat, tp, token_id, iterations, token_mse_error_tracker):
        with torch.no_grad():
            old_logprobs = self.student_logprobs[example_id, rank]
            old_S = self.S[example_id]
            
            #assert torch.isnan(self.S).any() == False
            #assert torch.isnan(y_hat).any() == False
            self.S[example_id] = self.S[example_id] - (torch.exp(old_logprobs) * (1 - torch.exp(y_hat)))
            new_S = self.S[example_id]
            
            #assert old_S.min() > 0, old_S.min()
            #assert new_S.min() > 0, new_S.min()
            
            self.kl_divs[example_id] = self.kl_divs[example_id] - tp[example_id,rank]*y_hat - old_S.log() + new_S.log()
            self.student_logprobs[example_id, rank] += y_hat
            
            # Calculate immediate MSE after update and store directly in dictionary
            new_probs = torch.exp(self.student_logprobs[example_id, rank]) / self.S[example_id]
            mse = F.mse_loss(new_probs, tp[example_id, rank]).item()
            # if len(token_mse_error_tracker[token_id.item()]) <= iterations:
            #     token_mse_error_tracker[token_id.item()].append([mse, None])
            # else:
            #     token_mse_error_tracker[token_id.item()][0].append(mse)
            token_mse_error_tracker[token_id.item()][0].append(mse)
            
    # def synchronize(self):
    #     self.S += self.lazy_deltas

def load_data(file_names):
    with torch.no_grad():
        token_ids, sprobs_list, logit_list, embed_list = [], [], [], []
        for file_name in file_names:
            index_file = torch.tensor(np.fromfile(f'/sanDisk{rank+1}/raw_logits_v3/{file_name}_student_index_new', dtype=np.uint16),device=configuration['device']).reshape(-1,2).int()
            tid = torch.tensor(np.fromfile(f'/sanDisk{rank+1}/raw_logits_v3/{file_name}_id_t',  dtype=np.int32), device=configuration['device']).reshape(1000, 1024, 100)
            gt  = torch.tensor(np.fromfile(f'/sanDisk{rank+1}/raw_logits_v3/{file_name}_id_gt', dtype=np.int32), device=configuration['device']).reshape(1000, 1024     )    
            mask = torch.isin(tid[..., :10], toi).any(dim=-1) + torch.isin(gt, toi)
            
            logit_t = torch.tensor(np.fromfile(f'/sanDisk{rank+1}/raw_logits_v3/{file_name}_logit_t',  dtype=np.float16), device=configuration['device']).reshape(1000, 1024, 100)
            sprobs = torch.tensor(np.fromfile(f'/sanDisk{rank+1}/raw_logits_v3/{file_name}_student_probs_new', dtype=np.float16), device=configuration['device']).reshape(-1, 100    )
            embeds = torch.tensor(np.fromfile(f'/sanDisk{rank+1}/raw_logits_v3/{file_name}_student_embeddings_new', dtype=np.int8), device=configuration['device']).reshape(-1, 1024    )
            
            tid = tid.reshape(-1, 100)[mask.reshape(-1)]
            gt = gt.reshape(-1)[mask.reshape(-1)]
            logit_t = logit_t.reshape(-1,100)[mask.reshape(-1)]
            # tprobs = torch.softmax(logit_t, dim=-1)
            token_ids.append((tid, gt))
            sprobs_list.append(sprobs)
            logit_list.append(logit_t)
            embed_list.append(embeds)
    
        tid = torch.concatenate(list(map(lambda x : x[0], token_ids)), dim=0)
        gt =  torch.concatenate(list(map(lambda x : x[1], token_ids)), dim=0)
        sprobs = torch.concatenate(sprobs_list, dim=0).float()
        embeds = torch.concatenate(embed_list, dim=0)
        logits = torch.concatenate(logit_list, dim=0)
        
        tprobs = torch.softmax(logits.float(), dim=-1)
        
        tp = (tprobs.to(torch.float32).to(configuration['device'])+1e-10)
        sp = (sprobs.to(torch.float32).to(configuration['device'])+1e-10)
        tp = tp / tp.sum(dim=-1, keepdims=True) #* tprobs.half().sum(dim=-1, keepdims=True).float()
        sp = sp / sp.sum(dim=-1, keepdims=True) #* sprobs.half().sum(dim=-1, keepdims=True).float()
        kl_divs = ((tp*((tp).log() - (sp).log() + sp.sum(dim=-1, keepdims=True).log())).sum(dim=-1))
        
        
        tid_copy = tid.clone()[..., :10]
        gt_oi = (torch.isin(gt, toi) * gt).reshape(-1, 1) # Ground truth positions of interest
        # positions where we have a ground truth of interest and no teacher prediction contains it.
        to_inject = ((~(tid_copy == gt_oi)).all(dim=-1)  * (gt_oi.reshape(-1) > 0))
        tid_copy[to_inject, -1] = gt_oi[to_inject, 0]
        
        BK = BookKeeper(sp)
        BK.kl_divs = kl_divs.float().to(sp.device)
        BK.raw_kl_divs[:] = BK.kl_divs
        
        TI = TokenIndexer(tid_copy, toi)
        return BK, TI, tp, embeds, sp, tid, gt
    
"""
    Function to compute sparse KL divergence of a distribution and GT
    
    Sorts the predicted ids and performs binary search of the ground truth ids for each example
    Then computes Cross-Entropy-like loss as there is only one entry of interest
    
    Requirements:
        GT_ID     -> Tensor(n_examples, 1)    the ground-truth token_ids
        dist_id   -> Tensor(n_examples, k)    predicted token_ids
        dist_prob -> Tensor(n_examples, k)    probabilities corresponding to predicted token_ids
        eps       -> (Optional) tensor        probability assigned to tokens not captured by the sparse distribution
                                              Consider this as penalty for uncaptured GT tokens
    Returns:
        Total KL_divergence over all examples passed 
"""
def GT_KL_divergence(GT_ID, distribution_id, distribution_probs, eps=torch.tensor(1e-5), return_hits_mask=False):
    distribution_id_sorted, argsorted = torch.sort(distribution_id, dim=-1)
    probs_sorted = torch.gather(distribution_probs, dim=-1, index=argsorted)

    idx_in_ids2 = torch.searchsorted(distribution_id_sorted, GT_ID).clamp(0, distribution_id.size(1)-1)
    queried_ids = torch.gather(distribution_id_sorted, dim=-1, index=idx_in_ids2)
    queried_probs = torch.gather(probs_sorted, dim=-1, index=idx_in_ids2)
    hits = (queried_ids == GT_ID)
    if torch.is_tensor(eps):
        total_kl = -(hits * queried_probs.log()).sum() - (~hits).sum() * torch.log(eps)
    else:
        raise Exception("Need to pass tensor for the epsilon argument")
    if return_hits_mask:
        return total_kl, hits
    return total_kl

class DebugAdamW(optim.AdamW):
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data.to(configuration['optimizer_bits'])
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')

                # Inspect gradient values
                # print(f"Gradient min: {grad.min()}, max: {grad.max()}, mean: {grad.mean()}")

                state = self.state[p]

                # Initialize state if first time
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data).to(configuration['optimizer_bits'])
                    state['exp_avg_sq'] = torch.zeros_like(p.data).to(configuration['optimizer_bits'])

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # Decay the first and second moment running average coefficient
                #grad=grad.to(torch.float32).clip(-1,1)*256*128
                grad=grad.to(torch.float32).clip(-configuration['grad_clipping_factor'], configuration['grad_clipping_factor'])*configuration['grad_multiplier']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected first and second moment estimates
                state['step'] += 1
                step = state['step']
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                denom = (exp_avg_sq.to(torch.float32).sqrt() / (bias_correction2 ** 0.5)).add_(group['eps']*configuration['grad_multiplier'])
                update= exp_avg.to(torch.float32) / denom
                step_size = group['lr'] / bias_correction1

                # Weight decay
                if group['weight_decay'] != 0:
                    # print(f"Applying weight decay: {group['weight_decay']}")
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Parameter update
                update = -step_size * update
                # print(f"Update min: {update.min()}, max: {update.max()}, mean: {update.mean()}")
                p.data.add_(update.to(torch.float16)) # TODO: read this from the configuration

                # Check for NaN or Inf in parameters
                # if torch.isnan(p.data).any() or torch.isinf(p.data).any():
                    # print(f"Parameter became NaN or Inf after update!")
                    # print(f"Parameter: {p.data}")
                    # print(f"Gradient: {grad}")
                    # print(f"Exp_avg: {exp_avg}")
                    # print(f"Exp_avg_sq: {exp_avg_sq}")

        return loss
    
#Multi
class multi_embedding(torch.nn.Module):
    def __init__(self, token_id, device, embedding_size = 2048, number_of_clusters = 2):
        super(multi_embedding, self).__init__()
        self.number_of_clusters=number_of_clusters
        self.embedding_size = embedding_size
        self.normalization_factor = torch.sqrt(torch.tensor(self.embedding_size).to(device))
        self.token_of_interest = token_id
        # self.linear1 =torch.nn.Linear(self.embedding_size, self.number_of_clusters, device=device,dtype=torch.float32,bias=True)
        self.linear1 =torch.nn.Linear(self.embedding_size, self.number_of_clusters, device=device,dtype=configuration['linear_bits'], bias=True)
        self.final_bias=torch.nn.Parameter(torch.zeros(1,device=device))
        self.y2_factor=torch.nn.Parameter(torch.zeros(1,device=device))
        self.nsfactor=torch.nn.Parameter(torch.zeros(1,device=device))
        
        # self.student_factor=torch.nn.Parameter(torch.ones(1,device=device))
        self.student_prob2_factor=torch.nn.Parameter(torch.zeros(1,device=device))
        # self.sy=torch.nn.Parameter(torch.zeros(1,device=device))
        self.student_factor = 0
        #self.student_prob2_factor = 0
        self.sy = 0

        self.device=device
        self.relu=torch.nn.ReLU()
        # self.relu=torch.nn.ELU()
    
    def forward(self, input_embs, student_logprob, teacher_logprob, teacher_prob, S, KL1, percentage=100, others=None):
        input_embs = input_embs.to(configuration['linear_bits'])
        student_logprob = student_logprob.to(torch.float32)#.to(configuration['linear_bits'])
        teacher_logprob = teacher_logprob.to(torch.float32)#.to(configuration['linear_bits'])
        
        # input_embs = input_embs.to(torch.float16)
        # student_logprob = student_logprob.to(torch.float16)
        # teacher_logprob = teacher_logprob.to(torch.float16)
        #assert torch.isnan(input_embs).any() == False
        scores = self.linear1(input_embs).to(torch.float32)
        #assert torch.isnan(scores).any() == False
        scores/=self.normalization_factor
        ps=self.relu( scores)
        #assert torch.isnan(ps).any() == False
        # ns=self.relu(-scores)
        split = int(ps.shape[1] * percentage / 100)
        if configuration['linear_bits'].itemsize == 2:
            ps = ps.to(torch.float32) #+ 1e-8
        
        y = ps[:, :split].sum(dim=-1) - ps[:, split:].sum(dim=-1)
        y= y+ self.final_bias/10 + y**2*self.y2_factor*10
        #assert torch.isnan(y).any() == False
        if others is not None:
            others['ps'] = ps.cpu().detach().numpy()
            others['final_bias'] = self.final_bias.data.cpu().detach().numpy()
            others['y2_factor'] = self.y2_factor.data.cpu().detach().numpy()
        # I want my range to be [-20, 0] for normalized logit case 
        if configuration['normalize']:
            y=torch.clip(y,-YHATRANGE, YHATRANGE)
            #assert torch.isnan(y).any() == False
            teacher_logprob = torch.clip(teacher_logprob, -20, 0)
        else: # If we don't normalize, teacher logits or residuals could vary greately 
            y = (y*20 + torch.sqrt(torch.abs(y))*torch.sign(y)*20) / 2
            #assert torch.isnan(y).any() == False
        #assert torch.isnan(y).any() == False

        # Global KL loss
        sprimeold = S * (1 - torch.exp(student_logprob) * (1 - torch.exp(y)))
        #assert torch.isnan(sprimeold).any() == False
        sprime=sprimeold
        # print(f"sprime_old min: {sprimeold.min().item():.05f}")
        # print(f"S min: {S.min().item():.05f}")
        global_kl_loss         = (KL1 - teacher_prob * y - S.log() + sprime.log() )
        global_kl_default_zero = (KL1)
        #assert torch.isnan(KL1).any() == False
        #assert torch.isnan(S).any() == False
        #assert torch.isnan(sprime).any() == False
        #assert torch.isnan(y).any() == False

        """
        # L1
        l1_loss = torch.abs ((teacher_logprob - y))
        l1_loss = l1_loss * teacher_prob
        l1_default_zero = torch.abs((teacher_logprob - student_logprob)) * teacher_prob
        
        # L2
        l2_loss = (teacher_logprob - y)**2
        l2_loss = l2_loss * teacher_prob
        l2_default_zero = ((teacher_logprob - (student_logprob - student_logprob.mean() + teacher_logprob.mean())) ** 2) * teacher_prob
        
        # MSE logit
        mse_logit = F.mse_loss(student_logprob + y, teacher_logprob)
        mse_logit_zero = F.mse_loss(student_logprob, teacher_logprob)
        
        # L2 on logits
        logit_l2_loss = (student_logprob + y - teacher_logprob)**2 * teacher_prob
        logit_l2_loss_default_zero = ((student_logprob+1e-9).log() - (teacher_logprob+1e-9).log())**2 * teacher_prob
        """
        
        losses = {
            # 'l1'           : l1_loss,
            # 'l1_0'         : l1_default_zero,
            # 'l2'           : l2_loss,
            # 'l2_0'         : l2_default_zero,
            # 'logit_l2'     : logit_l2_loss,
            # 'logit_l2_0'   : logit_l2_loss_default_zero,
            # 'mse_logit'    : mse_logit,
            # 'mse_logit_0'  : mse_logit_zero,
            'global'       : global_kl_loss,
            'global_0'     : global_kl_default_zero
        }
        return y, losses
        
class ParallelTrainer():
    def __init__(self, token_id, configuration, device='cpu', W=None):
        self.token_id = token_id
        self.device  = device
        torch.manual_seed(0)
        self.model = multi_embedding(token_id, self.device, number_of_clusters=configuration['num_clusters'])
        # self.optimizer = optim.AdamW(self.model.parameters(), lr=configuration['lr'], betas=(0.9, 0.99), weight_decay=configuration['weight_decay'], eps=configuration['optimizer_eps'])
        self.optimizer = DebugAdamW(self.model.parameters(), lr=configuration['lr'], betas=(0.9, 0.99), weight_decay=configuration['weight_decay'], eps=configuration['optimizer_eps'])
        self.optimizer.zero_grad()
        self.examples_collected = 0
        self.W = W
    def _optimizer_to(self, device):
        for param in self.optimizer.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)
    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        self._optimizer_to(device)
    
    def step(self, configuration, data, logger, writer, S, KL, train=True, others=None, already_decoded=False):
        # Book-keepingish
        batchsize = configuration['batchsize']
        loss = configuration['loss']
        model, optimizer = self.model, self.optimizer
        
        # Forward 
        embs, slogprob, tlogprob, tprob = data
        # torch.cuda.synchronize(configuration['device'])
        if not already_decoded:
            embs = self.W.rescale(Q.load_decode(embs))
        # torch.cuda.synchronize(configuration['device'])
        # WEIGHTS
        weights = tprob
        if train:
            model.train()
            # torch.cuda.synchronize(configuration['device'])
            output, losses = model(embs, slogprob * configuration['residual_scaler'], tlogprob * configuration['residual_scaler'], weights, S, KL, configuration['percentage'], others=others)
            # torch.cuda.synchronize(configuration['device'])
            # Batched loss
            lval = losses[loss].mean()
            # torch.cuda.synchronize(configuration['device'])
            #assert torch.isnan(lval).any() == False
            lval.backward()
            # print(lval.item())
            # print(self.token_id, self.model.linear1.weight.data.norm(), self.model.linear1.weight.grad.norm())
            # torch.cuda.synchronize(configuration['device'])
            # output.backward(gradient=(torch.exp(slogprob) - torch.exp(tlogprob))/embs.size(0))
            self.examples_collected += embs.shape[0]
        else:
            model.eval()
            # torch.cuda.synchronize(configuration['device'])
            output, losses = model(embs, slogprob * configuration['residual_scaler'], tlogprob * configuration['residual_scaler'], weights, S, KL, configuration['percentage'])
            # torch.cuda.synchronize(configuration['device'])
            
        if train and self.examples_collected >= batchsize:
            # torch.cuda.synchronize(configuration['device'])
            # print("Before step")
            # print(self.model.linear1.weight)
            # print(self.model.linear1.weight.grad)
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"checking {name} ", end='')
            #         if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
            #             print(f"{name} has NaN or Inf in gradients")
            # print()
            optimizer.step()
            # print("After step")
            # print(self.model.linear1.weight)
            
            optimizer.zero_grad()
            self.examples_collected = 0
            # torch.cuda.synchronize(configuration['device'])
        # logger.accumulate(configuration, self.token_id,  embs.shape[0], losses[loss].sum().item(), losses[loss+"_0"].sum().item(), writer, train)
        # torch.cuda.synchronize(configuration['device'])
        return output, losses
    
def consumer(queue, BK, token_mse_error_tracker, configuration):
    """Consumer function that processes updates from producers
    
    Args:
        queue: multiprocessing Queue containing update information
        BK: BookKeeper instance for maintaining model state
        token_mse_error_tracker: Dictionary tracking MSE errors for each token
    """
    while True:
        item = queue.get()
        if item is None:  # sentinel value to stop consumer
            break
            
        # Unpack and apply updates
        update_args = item['update']
        example_ids, ranks, y_hats, token_id, iterations = update_args
        BK.update(example_ids, ranks, y_hats, token_id=token_id, 
                iterations=iterations, token_mse_error_tracker=token_mse_error_tracker)
        
        # Update KL divergences
        update_kl_args = item['update_kl_divs']
        example_ids, teacher_prob, teacher_logprob = update_kl_args
        BK.update_kl_divs(example_ids, teacher_prob=teacher_prob, teacher_logprob=teacher_logprob)

def producer(queue, gpu_id, start_idx, end_idx, BK, TI, tprobs, embeds, sprobs, predicted_ids, gt_ids, token_ids, trainers, configuration):
    """Producer function that runs on specific GPU"""
    print(BK.student_logprobs)
    file_decode = False
    # Get local references to GPU data
    # local_data = gpu_data[gpu_id]
    # local_BK = local_data['BK']
    # local_TI = local_data['TI']
    # local_tprobs = local_data['tprobs']
    # local_embeds = local_data['embeds']

    # Process assigned tokens
    for iterations in range(1):
        for t in range(start_idx, end_idx):
            token_id = token_ids[t]
            Trainer = trainers[t]  # Already on correct GPU
            
            # Get data through BK interface - all data already on correct GPU
            example_ids, ranks = TI.get_train_idx(token_id)
            example_ids = example_ids.to(Trainer.device)
            ranks = ranks.to(Trainer.device)
            student_data = BK.get(example_ids, ranks)
            tprobs_slice = tprobs[example_ids, ranks].to(Trainer.device)
            tlogprobs_slice = (tprobs_slice+1e-20).log().to(Trainer.device)
            emb = embeds[example_ids].to(Trainer.device)
            
            # Training logic - everything already on correct GPU
            for ind in torch.split(torch.arange(tprobs_slice.size(0)), configuration['batchsize']):
                student_logprobs = student_data['logprobs'][ind].to(Trainer.device)
                S = student_data['S'][ind].to(Trainer.device)
                KL = student_data['kl'][ind].to(Trainer.device)
                
                y_hats, losses = Trainer.step(
                    configuration,
                    (emb[ind], student_logprobs, tlogprobs_slice[ind], tprobs_slice[ind]),
                    logger, writer, S, KL, train=True, already_decoded=file_decode
                )
                
                # Send only necessary data back to consumer
                queue_args = {
                    'update': (example_ids[ind].cpu(), ranks[ind].cpu(), 
                             y_hats.float().detach().cpu(), 
                             token_id.detach().cpu(), iterations),
                    'update_kl_divs': (example_ids[ind].cpu(), 
                                     tprobs[example_ids[ind]].cpu(), 
                                     (tprobs[example_ids[ind]]+1e-10).log().cpu())
                }
                queue.put(queue_args)

def parallel_train(n_devices, BK, TI, tprobs, embeds, sprobs, predicted_ids, gt_ids, token_ids, trainers, configuration):
    # Setup multiprocessing
    mp.set_start_method('spawn', force=True)
    queue = mp.Queue()
    chunk_size = len(token_ids) // n_devices
    chunks = [(i * chunk_size, (i + 1) * chunk_size if i < n_devices - 1 else len(token_ids)) 
             for i in range(n_devices)]
    # Start producers
    producers = []
    for gpu_id, (start, end) in enumerate(chunks):
        p = mp.Process(
            target=producer,
            args=(queue, gpu_id, start, end, BK, TI, tprobs, embeds, sprobs, predicted_ids, gt_ids, token_ids, trainers, configuration)
        )
        p.start()
        producers.append(p)
    
    # Consumer remains same as before
    consumer_process = mp.Process(
        target=consumer, 
        args=(queue, BK, token_mse_error_tracker, configuration)
    )
    consumer_process.start()
    
    # Wait for completion
    for p in producers:
        p.join()
    queue.put(None)
    consumer_process.join()
    
    return token_mse_error_tracker

if __name__ == "__main__":
    configuration = {
        'epochs' : 10,
        'logpath' : "/home/igli/final_models_logloss_quick/TP",
        'device' : 'cuda:0',  # this is changed in process_action function
        'batchsize' : 10000,
        'num_clusters' : 1024,
        'lr' : 0.0003,
        'optimizer' : 'adamw',
        'loss' : 'residual_l1',
        'weight_decay' : 1,
        'train_log_step' : 10,
        'test_log_step' : 100,
        'residual_scaler' : 1,
        'percentage' : 50,
        'normalize' : True, # Either normalized_logit or raw_logit,
        'residual' : False,
        'note': 'test',
        'steplimit' : 5000,
        'nfiles' : -1,
        'plot_params' : False,
        'optimizer_eps' : 1e-8,
        'linear_bits' : torch.float32,
    }

    with torch.no_grad():
        toi = torch.tensor(np.fromfile('/home/igli/tokens_of_interest', dtype=np.int32), device=configuration['device'])
    logger = LoggingManager(toi, 10)
    validation_logger = LoggingManager(toi, 1)
    with torch.no_grad():
        Q = quantizer_4b_torch()
        W = representation_whitening_torch()
        W.load_state_dict('/home/igli/whitening_params.pth')
        W.to(configuration['device'])
    with torch.no_grad():
        token_ids = toi[:300]
    configuration['epochs'] = 1

    configuration['num_clusters'] = 1024

    configuration['lr'] = 5e-4

    configuration['loss'] = 'global'

    YHATRANGE = 10

    token_mse_error_tracker = dict()

    step = 0
    epochs_per_lr = 6
    while step < 3*epochs_per_lr:
        # Best setting so far
        if step == 0 * epochs_per_lr:
            configuration['lr'] = 5e-4
            configuration['linear_bits'] = torch.float16
            configuration['optimizer_bits'] = torch.float32
            configuration['grad_clipping_factor'] = 1/128
            configuration['grad_multiplier'] = 256 * 1
            configuration['optimizer_eps'] = 1e-8
            
            configuration['device'] = 'cuda:0'
            W.to(configuration['device'])
            trainers = [ParallelTrainer(token_id.item(), configuration, device=configuration['device'], W=W) for token_id in token_ids[:100]]
            configuration['device'] = 'cuda:1'
            W.to(configuration['device'])
            trainers += [ParallelTrainer(token_id.item(), configuration, device=configuration['device'], W=W) for token_id in token_ids[100:200]]
            configuration['device'] = 'cuda:2'
            W.to(configuration['device'])
            trainers += [ParallelTrainer(token_id.item(), configuration, device=configuration['device'], W=W) for token_id in token_ids[200:300]]
            
            configuration['device'] = 'cuda:0'
            writer = SummaryWriter(f"paralleltrainer/{configuration['linear_bits']}_{configuration['optimizer_bits']}_clipping_{1/configuration['grad_clipping_factor']}_gradFactor_{configuration['grad_multiplier']}_lr_{configuration['lr']}_optimeps={configuration['optimizer_eps']}")
        
        # Smaller learning rate
        if step == 1 * epochs_per_lr:
            configuration['lr'] = 1e-4
            configuration['linear_bits'] = torch.float16
            configuration['optimizer_bits'] = torch.float32
            configuration['grad_clipping_factor'] = 1/128
            configuration['grad_multiplier'] = 256 * 1
            configuration['optimizer_eps'] = 1e-8
            trainers = [ParallelTrainer(token_id.item(), configuration, device=configuration['device'], W=W) for token_id in token_ids]
            writer = SummaryWriter(f"bigtest/{configuration['linear_bits']}_{configuration['optimizer_bits']}_clipping_{1/configuration['grad_clipping_factor']}_gradFactor_{configuration['grad_multiplier']}_lr_{configuration['lr']}_optimeps={configuration['optimizer_eps']}")
        
        # Bigger grad multiplier
        if step == 2 * epochs_per_lr:
            configuration['lr'] = 5e-4
            configuration['linear_bits'] = torch.float16
            configuration['optimizer_bits'] = torch.float32
            configuration['grad_clipping_factor'] = 1/128
            configuration['grad_multiplier'] = 256 * 128
            configuration['optimizer_eps'] = 1e-8
            trainers = [ParallelTrainer(token_id.item(), configuration, device=configuration['device'], W=W) for token_id in token_ids]
            writer = SummaryWriter(f"bigtest/{configuration['linear_bits']}_{configuration['optimizer_bits']}_clipping_{1/configuration['grad_clipping_factor']}_gradFactor_{configuration['grad_multiplier']}_lr_{configuration['lr']}_optimeps={configuration['optimizer_eps']}")
            
        for filename in range(0, 2000 ):
            rank = filename % 2
            BK, TI, tprobs, embeds, sprobs, predicted_ids, gt_ids = load_data([f'{filename:04d}'])
            gt_ids = gt_ids.unsqueeze(-1)
            ## Computing teacher-GT  KL loss
            print(f"File Number {filename:04d} Average KL(GT||teacher): {GT_KL_divergence(gt_ids, predicted_ids, tprobs)/gt_ids.size(0)}")
            ## Computing student-GT  KL loss
            vanilla_student_kl = GT_KL_divergence(gt_ids, predicted_ids, sprobs,)/gt_ids.size(0)
            print(f"File Number {filename:04d} Average KL(GT||vanilla student): {vanilla_student_kl}")
            start_time = time.time()
            parallel_train(3, BK, TI, tprobs, embeds, sprobs, predicted_ids, gt_ids, token_ids, trainers, configuration)
            end_time = time.time()
            torch.cuda.empty_cache()
            refined_model_kl = GT_KL_divergence(gt_ids, predicted_ids, torch.exp(BK.student_logprobs - BK.S.log().unsqueeze(-1)))/gt_ids.size(0)
            print(f"Time taken for refinement process: {end_time-start_time:0.4f}")
            print(f"File Number {filename:04d} Average KL(GT||refined student): {refined_model_kl}")
            print(f"Improvement: {vanilla_student_kl - refined_model_kl}\n")
            writer.add_scalar("Improvement", vanilla_student_kl - refined_model_kl, 2000*(step%epochs_per_lr) + filename)
        step += 1