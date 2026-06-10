# Imports
import os, pickle, torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import matplotlib.colors as pltc
import numpy as np
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import datetime
import threading
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
        is_negative=x>=8
        a=(( torch.remainder(x,8).to(torch.float16)+2.1)/5.3)**2
        a=a*(1-is_negative.to(torch.int8)*2)
        return a
    
    def quantize_compress(self,a):
        ind=self.quantize(a)
        comp=ind[:,0::2]*16+ind[:,1::2]
        return comp.to(torch.uint8)

    def load_decode(self,data):
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
        return (self.s*e)@self.q.T+self.mn
    
    def rescale(self,e):
        return self.s*e
    
    def save_state_dict(self,path):
        torch.save({'mn':self.mn,'q':self.q,'s':self.s}, path)

    def load_state_dict(self,path):
        data=torch.load(path)
        self.mn=data['mn']
        self.q =data['q' ]
        self.s =data['s' ]
        
    def to(self, d):
        self.mn = self.mn.to(d)
        self.q = self.q.to(d)
        self.s = self.s.to(d)

#Multi
class multi_embedding(torch.nn.Module):
    def __init__(self, token_id, device, embedding_size = 2048, number_of_clusters = 2):
        super(multi_embedding, self).__init__()
        self.number_of_clusters=number_of_clusters
        self.embedding_size = embedding_size
        self.normalization_factor = torch.sqrt(torch.tensor(self.embedding_size).to(device))
        self.token_of_interest = token_id
        self.linear1 =torch.nn.Linear(self.embedding_size, self.number_of_clusters, device=device,dtype=torch.float32,bias=True)
        self.final_bias=torch.nn.Parameter(torch.zeros(1,device=device))
        self.y2_factor=torch.nn.Parameter(torch.zeros(1,device=device))
        self.nsfactor=torch.nn.Parameter(torch.zeros(1,device=device))
        
        # self.student_factor=torch.nn.Parameter(torch.ones(1,device=device))
        # self.student_prob2_factor=torch.nn.Parameter(torch.zeros(1,device=device))
        # self.sy=torch.nn.Parameter(torch.zeros(1,device=device))
        self.student_factor = 0
        self.student_prob2_factor = 0
        self.sy = 0

        self.device=device
        self.relu=torch.nn.ReLU()
    
    def forward(self, input_embs, student_prob, teacher_prob, weight, percentage=100):
        input_embs = input_embs.to(torch.float32)
        student_prob = student_prob.to(torch.float32)
        teacher_prob = teacher_prob.to(torch.float32)
        scores = self.linear1(input_embs)
        scores/=self.normalization_factor
        ps=self.relu( scores)
        ns=self.relu(-scores)
        
        split = int(ps.shape[1] * percentage / 100)

        y = ps[:, :split].sum(dim=-1) - ps[:, split:].sum(dim=-1)
        y = y+student_prob*self.student_factor+self.final_bias+(student_prob**2*self.student_prob2_factor+y**2*self.y2_factor+student_prob*y*self.sy)*10
        y=torch.clip(y,0,configuration['residual_scaler'])
        # L1
        l1_loss = torch.abs ((teacher_prob - y))
        l1_loss = l1_loss * weight
        # L2
        l2_loss = (teacher_prob - y)**2
        l2_loss = l2_loss * weight
        # KL
        kl_loss = torch.abs(teacher_prob * ((y+1e-9).log() - (teacher_prob+1e-9).log()))
       
        #default all-zero-l1
        l1_default_zero = torch.abs((teacher_prob - student_prob)) * weight
        #default zero l2
        l2_default_zero = ((teacher_prob - student_prob) ** 2) * weight
        losses = {
            'l1' : l1_loss,
            'l2' : l2_loss,
            'kl' : kl_loss,
            'l1_0':l1_default_zero,
            'l2_0':l2_default_zero,
        }
        
        return y, losses

class ParallelTrainer():
    def __init__(self, token_id, configuration, device='cpu', W=None):
        self.token_id = token_id
        self.device  = device
        torch.manual_seed(0)
        self.model = multi_embedding(token_id, self.device, number_of_clusters=configuration['num_clusters'])
        self.optimizer = optim.AdamW(self.model.parameters(), lr=configuration['lr'], betas=(0.9, 0.99), weight_decay=configuration['weight_decay'])
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
    
    def step(self, configuration, data, logger, writer, train=True):
        # Book-keepingish
        batchsize = configuration['batchsize']
        loss = configuration['loss']
        model, optimizer = self.model, self.optimizer
        
        # Forward 
        embs, sprob, tprob, gtprob = data
        if int(configuration['device'][-1]) > 1:
            tprob = gtprob   ## Here tprob is more like the target probability vs teacher probability
        embs = self.W.rescale(Q.load_decode(embs))
        weights = torch.ones_like(tprob)
        if train:
            model.train()
            output, losses = model(embs, sprob * configuration['residual_scaler'], tprob * configuration['residual_scaler'], weights, configuration['percentage'])
            # Batched loss
            lval = losses[loss].mean()
            lval.backward()
            self.examples_collected += embs.shape[0]
        else:
            model.eval()
            output, losses = model(embs, sprob * configuration['residual_scaler'], tprob * configuration['residual_scaler'], weights, configuration['percentage'])
            
        if train and self.examples_collected >= batchsize:
            optimizer.step()
            optimizer.zero_grad()
            self.examples_collected = 0
        logger.accumulate(configuration, self.token_id,  embs.shape[0], losses[loss].sum().item(), losses[loss+"_0"].sum().item(), writer, train)
        
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
        log_step = self.log_step
        self.batch_losses[token_id] += batch_loss
        self.default_losses[token_id] += default_loss
        self.items_seen[token_id] += no_items 
        self.steps_taken[token_id] += 1
        if train and self.steps_taken[token_id] % log_step == 0:
            self.log(configuration, token_id, writer, train)
            
    def log(self, configuration, token_id, writer, train):
        loss = configuration['loss']
        if train:
            # writer.add_scalar(f'{loss.upper()}/Training Average Batch Loss', self.batch_losses[token_id]/(self.items_seen[token_id]), self.steps_taken[token_id])
            writer.add_scalar(f'{loss.upper()}/Training Batch Loss Ratio', self.batch_losses[token_id]/self.default_losses[token_id], self.steps_taken[token_id])
        else:
            # writer.add_scalar(f'{loss.upper()}/Validation Average Batch Loss', self.batch_losses[token_id]/(self.items_seen[token_id]), self.steps_taken[token_id])
            writer.add_scalar(f'{loss.upper()}/Validation Batch Loss Ratio', self.batch_losses[token_id]/self.default_losses[token_id], self.steps_taken[token_id])
        
        self.batch_losses[token_id] = 0
        self.default_losses[token_id] = 0
        self.items_seen[token_id] = 0 

def run_models_training(toi, configuration, W, logger, validation_logger):
    import subprocess as s
    print(configuration)
    for token_id in toi:
        configuration['batchsize'] = 10000
        torch.cuda.empty_cache()
        print(token_id)

        configuration['note'] = f"{token_id.item()}_batchsize_{configuration['batchsize']}"
        writer = SummaryWriter(f"{configuration['logpath']}/{configuration['note']}")
        Trainer = ParallelTrainer(token_id.item(), configuration, device=configuration['device'], W=W)
        
        probfiles = [*s.run(['find', '/sanDisk1/token_index_data/', '-name', f'{token_id.item()}_*_probs'], capture_output=True).stdout.decode('utf-8').split(), *s.run(['find', '/sanDisk2/token_index_data/', '-name', f'{token_id.item()}_*_probs'], capture_output=True).stdout.decode('utf-8').split()]
        embdfiles = [x[:-5]+"embed" for x in probfiles]
        if len(probfiles) == 0:
            assert len(embdfiles) == 0
            continue
    
        probs = torch.tensor(np.concatenate([np.fromfile(pfile, dtype=np.float16).reshape(-1,3) for pfile in probfiles])[:10000000], device=configuration['device'])
        embeds = torch.tensor(np.concatenate([np.fromfile(efile, dtype=np.uint8).reshape(-1,1024) for efile in embdfiles])[:10000000], device=configuration['device'])
        if probs.size(0) != embeds.size(0):
            print("ERROR FOR TOKEN:", token_id)
            print("\t\tprobs size:", probs.size(0), "embs size:", embeds.size(0))
            continue
        assert probs.size(0) == embeds.size(0)
        reperm = torch.randperm(probs.size(0))

        train_size = max(int(len(probs) * 0.8), len(probs) - 1000)
        configuration['batchsize'] = min(train_size, configuration['batchsize'])
        train_size = train_size // configuration['batchsize'] * configuration['batchsize']
                
        epoch = 0
        torch.manual_seed(0)

        while epoch < configuration['epochs'] and logger.steps_taken[token_id.item()] < configuration['steplimit']:
            perm = torch.randperm(train_size)
            # Training step
            for ind in torch.split(perm, configuration['batchsize']):
                student_probs, teacher_probs, gt_prob = probs[reperm[ind]][:,0], probs[reperm[ind]][:,1], probs[reperm[ind]][:,2]
                embeddings = embeds[reperm[ind]]
                Trainer.step(configuration, (embeddings, student_probs, teacher_probs, gt_prob), logger, writer, train=True)
            
            # Validation step
            for ind in torch.split(torch.arange(train_size, embeds.size(0), dtype=torch.int), configuration['batchsize']):
                student_probs, teacher_probs, gt_prob = probs[reperm[ind]][:,0], probs[reperm[ind]][:,1], probs[reperm[ind]][:,2]
                embeddings = embeds[reperm[ind]]
                Trainer.step(configuration, (embeddings, student_probs, teacher_probs, gt_prob), validation_logger, writer, train=False)
                validation_logger.log(configuration, Trainer.token_id, writer, train=False)   
            epoch += 1
        
        torch.save(Trainer.model.state_dict(), f"/amin/tempigli/{token_id.item()}")
        torch.cuda.empty_cache()
        for efile, pfile in zip(embdfiles, probfiles):
            os.remove(efile)
            os.remove(pfile)

def process_action(i, toi):
    configuration['device'] = f'cuda:{i}'
    chunksize = toi.size(0) // 2

    inner_toi = toi[chunksize*(i%2):chunksize*((i%2)+1)].cpu().to(configuration['device'])
    
    print(inner_toi)
        
    W = representation_whitening_torch()
    W.load_state_dict('/home/igli/whitening_params.pth')
    W.to(configuration['device'])
    logger = LoggingManager(inner_toi, 10)
    validation_logger = LoggingManager(inner_toi, 1)

    run_models_training(inner_toi, configuration, W, logger, validation_logger)

configuration = {
    'epochs' : 300,
    'logpath' : "/home/igli/final_logs",
    # 'device' : 'cpu',
    'device' : 'cuda:0',  # this is changed in process_action function
    'batchsize' : 10000,
    'num_clusters' : 128,
    'lr' : 0.0003,
    'optimizer' : 'adamw',
    'loss' : 'l1',
    'weight_decay' : 1,
    'train_log_step' : 10,
    'test_log_step' : 100,
    'residual_scaler' : 10,
    'percentage' : 100,
    'target' : 'teacher', # Either teacher or residual,
    'note': 'test',
    'steplimit' : 100
}

# High Frequency Tokens
toi = torch.tensor(np.fromfile('/home/igli/tokens_of_interest', dtype=np.int32), device=configuration['device'])[:100]
torch.manual_seed(0)
toi = toi[torch.randperm(len(toi))] # Shuffling the contents of TOI (hoping that memory wont be abused if we parallelize)
'''
# # Low Frequency Tokens
toi = torch.tensor(np.fromfile('/home/igli/tokens_of_interest', dtype=np.int32), device=configuration['device'])[200:]
torch.manual_seed(0)
toi = toi[torch.randperm(len(toi))[:200]] # Getting a random subset of the tokens
is_toi=torch.zeros(256000,device=configuration['device'],dtype=torch.bool)
is_toi[toi]=1

# 200 - 1000 bucket
toi200_1000 = np.fromfile('/home/igli/tokens_of_interest', dtype=np.int32)[200:1000]
is_toi_200_1000=torch.zeros(256000,device=configuration['device'],dtype=torch.bool)
is_toi_200_1000[toi200_1000]=1

toi = (is_toi_200_1000 * (~is_toi)).nonzero().reshape(-1)
'''

# Tokens that have exceeded 10M examples in the generation
# toi = np.fromfile('/home/igli/tokens_of_interest', dtype=np.int32)[2500:]
# computed = set(map(int, os.listdir('/amin/tempigli')))
# remaining = sorted(set(toi).difference(computed))
# toi = torch.tensor(remaining, device=configuration['device'])

# toi = torch.tensor([5952, 28642, 37059, 28553, 142029, 5968, 32625, 5941, 6394, 6397], device=configuration['device'])

tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b')
Q = quantizer_4b_torch()

process_action(0, toi)