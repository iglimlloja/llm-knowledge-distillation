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
import torch.nn.functional as F 

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
        self.student_prob2_factor=torch.nn.Parameter(torch.zeros(1,device=device))
        # self.sy=torch.nn.Parameter(torch.zeros(1,device=device))
        self.student_factor = 0
        #self.student_prob2_factor = 0
        self.sy = 0

        self.device=device
        self.relu=torch.nn.ReLU()
        # self.relu=torch.nn.ELU()

    
    def forward_old(self, input_embs, student_prob, teacher_prob, weight, percentage=100, others=None):
        input_embs = input_embs.to(torch.float32)
        student_prob = student_prob.to(torch.float32)
        teacher_prob = teacher_prob.to(torch.float32)
        scores = self.linear1(input_embs )
        scores/=self.normalization_factor
        ps=self.relu( scores)
        ns=self.relu(-scores)
        split = int(ps.shape[1] * percentage / 100)

        y = ps[:, :split].sum(dim=-1) - ps[:, split:].sum(dim=-1)
        y= y+ self.final_bias/10 + y**2*self.y2_factor*10
        if others is not None:
            others['ps'] = ps.cpu().detach().numpy()
            others['final_bias'] = self.final_bias.data.cpu().detach().numpy()
            others['y2_factor'] = self.y2_factor.data.cpu().detach().numpy()

        #y = y+student_prob*self.student_factor+self.final_bias+(student_prob**2*self.student_prob2_factor+y**2*self.y2_factor+student_prob*y*self.sy)*10
        y=torch.clip(y,0,configuration['residual_scaler'])
        y=torch.exp(-(1-y)*6)  # Still a probability

        #y=torch.clip(y,-1,1)
        #y=torch.exp(3*y-3)
        
        # y=torch.clip(y,-3,3)
        # y=torch.exp(y-3)
        
        # L1
        l1_loss = torch.abs ((teacher_prob - y))
        l1_loss = l1_loss * weight
        # L2
        l2_loss = (teacher_prob - y)**2
        l2_loss = l2_loss * weight
        # KL
        kl_loss = torch.abs(teacher_prob * ((y+1e-9).log() - (teacher_prob+1e-9).log()))
        #
        logit_l2_loss = ((y+1e-9).log() - (teacher_prob+1e-9).log())**2
        # MSE logit
        mse_logit = F.mse_loss((y+1e-9).log(), (teacher_prob+1e-9).log())
        #default all-zero-l1
        l1_default_zero = torch.abs((teacher_prob - student_prob)) * weight
        #default zero l2
        l2_default_zero = ((teacher_prob - student_prob) ** 2) * weight
        logit_l2_loss_default_zero = ((student_prob+1e-9).log() - (teacher_prob+1e-9).log())**2
        #default mse
        mse_logit_zero = F.mse_loss((student_prob+1e-9).log(), (teacher_prob+1e-9).log())
        losses = {
            'l1' : l1_loss,
            'l2' : l2_loss,
            'kl' : kl_loss,
            'logit_l2' : logit_l2_loss,
            'l1_0':l1_default_zero,
            'l2_0':l2_default_zero,
            'logit_l2_0' : logit_l2_loss_default_zero,
            'mse_logit' : mse_logit,
            'mse_logit_0' : mse_logit_zero
        }
        
        return y, losses

    def forward(self, input_embs, student_prob, teacher_prob, weight, percentage=100, others=None):
        input_embs = input_embs.to(torch.float32)
        student_prob = student_prob.to(torch.float32)
        teacher_prob = teacher_prob.to(torch.float32)
        
        scores = self.linear1(input_embs )
        scores/=self.normalization_factor
        ps=self.relu( scores)
        ns=self.relu(-scores)
        split = int(ps.shape[1] * percentage / 100)

        y = ps[:, :split].sum(dim=-1) - ps[:, split:].sum(dim=-1)
        y= y+ self.final_bias/10 + y**2*self.y2_factor*10
        if others is not None:
            others['ps'] = ps.cpu().detach().numpy()
            others['final_bias'] = self.final_bias.data.cpu().detach().numpy()
            others['y2_factor'] = self.y2_factor.data.cpu().detach().numpy()
        # I want my range to be [-20, 0] for normalized logit case 
        if configuration['normalize']:
            y=torch.clip(y,0,configuration['residual_scaler'])
            teacher_prob = torch.clip(teacher_prob, -20, 0)
            y= ((-(1-torch.sqrt(y))*20) + (-(1-y)*20))/2  
        else: # If we don't normalize, teacher logits or residuals could vary greately 
            y = (y*20 + torch.sqrt(torch.abs(y))*torch.sign(y)*20) / 2

        # L1
        l1_loss = torch.abs ((teacher_prob - y))
        l1_loss = l1_loss * weight
        # L2
        l2_loss = (teacher_prob - y)**2
        l2_loss = l2_loss * weight
        # KL
        kl_loss = torch.abs(teacher_prob * ((y+1e-9).log() - (teacher_prob+1e-9).log()))
        #
        logit_l2_loss = ((y+1e-9).log() - (teacher_prob+1e-9).log())**2
        # MSE logit
        mse_logit = F.mse_loss((y+1e-9).log(), (teacher_prob+1e-9).log())
        #default all-zero-l1
        l1_default_zero = torch.abs((teacher_prob - student_prob)) * weight
        #default zero l2    
        l2_default_zero = ((teacher_prob - (student_prob - student_prob.mean() + teacher_prob.mean())) ** 2) * weight
        # l2_default_zero = ((teacher_prob - (student_zero_mean/old_std*new_std + teacher_prob.mean())) ** 2) * weight
        logit_l2_loss_default_zero = ((student_prob+1e-9).log() - (teacher_prob+1e-9).log())**2
        #default mse
        mse_logit_zero = F.mse_loss((student_prob+1e-9).log(), (teacher_prob+1e-9).log())
        losses = {
            'l1' : l1_loss,
            'l2' : l2_loss,
            'kl' : kl_loss,
            'logit_l2' : logit_l2_loss,
            'l1_0':l1_default_zero,
            'l2_0':l2_default_zero,
            'logit_l2_0' : logit_l2_loss_default_zero,
            'mse_logit' : mse_logit,
            'mse_logit_0' : mse_logit_zero
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
    
    def step(self, configuration, data, logger, writer, train=True, others=None):
        # Book-keepingish
        batchsize = configuration['batchsize']
        loss = configuration['loss']
        model, optimizer = self.model, self.optimizer
        
        # Forward 
        embs, sprob, tprob, gtprob = data
        embs = self.W.rescale(Q.load_decode(embs))
        weights = torch.ones_like(tprob)
        
        if train:
            model.train()
            output, losses = model(embs, sprob * configuration['residual_scaler'], tprob * configuration['residual_scaler'], weights, configuration['percentage'], others=others)
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

def run_models_training(toi, configuration, W, logger, validation_logger):
    import subprocess as s
    from datetime import datetime
    print(configuration)
    for token_id in toi:
        if token_id.item() in already_computed:
            continue
        torch.cuda.empty_cache()
        print(token_id)
        currtime = datetime.now().strftime('%H:%M:%S')
        configuration['note'] = f"{token_id.item()}_batchsize_{configuration['batchsize']}_{currtime}"

        writer = SummaryWriter(f"{configuration['logpath']}/{configuration['note']}")
        Trainer = ParallelTrainer(token_id.item(), configuration, device=configuration['device'], W=W)
        
        probfiles = [*s.run(['find', '/sanDisk1/token_index_data/', '-name', f'{token_id.item()}_*_probs'], capture_output=True).stdout.decode('utf-8').split(), *s.run(['find', '/sanDisk2/token_index_data/', '-name', f'{token_id.item()}_*_probs'], capture_output=True).stdout.decode('utf-8').split()]
        if configuration['nfiles'] > 0:
            probfiles = probfiles[:configuration['nfiles']]
            sizes = list(map(lambda x : os.path.getsize(x) // 2 // 4, probfiles))
            cumsize = 0
            chunk = 0
            while cumsize < 20000000 and chunk < len(sizes):
                cumsize += sizes[chunk]
                chunk += 1
            probfiles = probfiles[:chunk]
            print("Chunksize =", cumsize)
        else:
            sizes = list(map(lambda x : os.path.getsize(x) // 2 // 4, probfiles))
            cumsize = 0
            chunk = 0
            while cumsize < 20000000 and chunk < len(sizes):
                cumsize += sizes[chunk]
                chunk += 1
            probfiles = probfiles[:chunk]
            print("Chunksize =", cumsize)
        embdfiles = [x[:-5]+"embed" for x in probfiles]
        
        if len(probfiles) == 0:
            assert len(embdfiles) == 0
            continue
        print(f"{len(probfiles)} Valid files found")
        print("Loading data to GPU...")
        probs  = torch.zeros((cumsize, 4   ), dtype=torch.float16, device=configuration['device'])
        embeds = torch.zeros((cumsize, 1024), dtype=torch.uint8  , device=configuration['device'])
        L = len(probfiles)
        
        probs1 = torch.tensor(np.concatenate([np.fromfile(pfile, dtype=np.float16).reshape(-1,4) for pfile in probfiles[:L//2]])[:cumsize])
        embeds1 = torch.tensor(np.concatenate([np.fromfile(efile, dtype=np.uint8).reshape(-1,1024) for efile in embdfiles[:L//2]])[:cumsize])
        c1 = probs1.shape[0]
        probs[:c1] = probs1
        embeds[:c1] = embeds1
        del probs1, embeds1
        
        probs2 = torch.tensor(np.concatenate([np.fromfile(pfile, dtype=np.float16).reshape(-1,4) for pfile in probfiles[L//2:]])[:cumsize])
        embeds2 = torch.tensor(np.concatenate([np.fromfile(efile, dtype=np.uint8).reshape(-1,1024) for efile in embdfiles[L//2:]])[:cumsize])
        probs [c1:] = probs2[:cumsize-c1]
        embeds[c1:] = embeds2[:cumsize-c1]
        del probs2, embeds2
        # Converting student probabilities to logits in the range (-20, 0): These are always *normalized* as we don't store raw student logits
        probs[:, 0] = torch.clip((probs[:, 0] + 1e-9).log(), -20, 0)
        
        if configuration['normalize']: # Making the label the normalized teacher logit --> (log(softmax(original_teacher_logits)))
            probs = probs[:, [0,2,3]]
            # Basically allowing probabilities from 1e-9 to 1
            probs[:, 1] = torch.clip(probs[:, 1], -20, 0)   
        else:                          # Making the label the raw logit                --> original_teacher_logit
            probs = probs[:, [0,1,3]] 
        if configuration['residual']:  # Subtracting student logit from the target
            probs[:, [0,1]] -= probs[:, 0].unsqueeze(1)
        
        if probs.size(0) != embeds.size(0):
            print("ERROR FOR TOKEN:", token_id)
            print("\t\tprobs size:", probs.size(0), "embs size:", embeds.size(0))
            continue
        assert probs.size(0) == embeds.size(0)
        print(f"{probs.size(0)} examples loaded")
        reperm = torch.randperm(probs.size(0))

        train_size = max(int(len(probs) * 0.8), len(probs) - 100000)
        configuration['batchsize'] = min(train_size, configuration['batchsize'])
        train_size = train_size // configuration['batchsize'] * configuration['batchsize']
                
        epoch = 0
        torch.manual_seed(0)
        others = dict() if configuration['plot_params'] else None
        while epoch < configuration['epochs'] and logger.steps_taken[token_id.item()] < configuration['steplimit']:
            perm = torch.randperm(train_size)
            # Training step
            for ind in torch.split(perm, configuration['batchsize']):
                student_probs, teacher_probs, gt_prob = probs[reperm[ind]][:,0], probs[reperm[ind]][:,1], probs[reperm[ind]][:,2]
                embeddings = embeds[reperm[ind]]
                Trainer.step(configuration, (embeddings, student_probs, teacher_probs, gt_prob), logger, writer, train=True, others=others)
                if others is not None and logger.steps_taken[token_id.item()] % logger.log_step == 0:
                    others['ps'] = others['ps'][:configuration['num_clusters']]
                    others['ps'] /= others['ps'].max()
                    writer.add_image("Parameters/ps", others['ps'][:configuration['num_clusters']][None,...], logger.steps_taken[token_id.item()])
                    writer.add_scalar("Parameters/final_bias", others['final_bias'], logger.steps_taken[token_id.item()])
                    writer.add_scalar("Parameters/y2_factor", others['y2_factor'], logger.steps_taken[token_id.item()])

            # Validation step
            for ind in torch.split(torch.arange(train_size, embeds.size(0), dtype=torch.int), configuration['batchsize']):
                student_probs, teacher_probs, gt_prob = probs[reperm[ind]][:,0], probs[reperm[ind]][:,1], probs[reperm[ind]][:,2]
                embeddings = embeds[reperm[ind]]
                Trainer.step(configuration, (embeddings, student_probs, teacher_probs, gt_prob), validation_logger, writer, train=False)
            validation_logger.log(configuration, Trainer.token_id, writer, train=False)
            epoch += 1
            
        torch.save(Trainer.model.state_dict(), f"/amin/tempigli/{os.path.basename(configuration['logpath'])}/{token_id.item()}_1024")

        print("Deleting")
        del probs, embeds
        torch.cuda.empty_cache()
        # for efile, pfile in zip(embdfiles, probfiles):
        #     os.remove(efile)
        #     os.remove(pfile)

def process_action(i, toi):
    inner_toi = toi
    print(inner_toi)
       
    W = representation_whitening_torch()
    W.load_state_dict('/home/igli/whitening_params.pth')
    W.to(configuration['device'])
    logger = LoggingManager(inner_toi, 10)
    validation_logger = LoggingManager(inner_toi, 1)

    run_models_training(inner_toi, configuration, W, logger, validation_logger)

configuration = {
    'epochs' : 300,
    'logpath' : "/home/igli/logit_training",
    # 'device' : 'cpu',
    'device' : 'cuda:0',  # this is changed in process_action function
    'batchsize' : 10000,
    'num_clusters' : 1024,
    'lr' : 0.0003,
    'optimizer' : 'adamw',
    'loss' : 'l2',
    'weight_decay' : 1,
    'train_log_step' : 10,
    'test_log_step' : 100,
    'residual_scaler' : 1,
    'percentage' : 50,
    'target' : 'teacher', # Either teacher or residual,
    'note': 'test',
    'steplimit' : 4000,
    'nfiles' : -1,
    'plot_params' : False,
}

# High Frequency Tokens
toi = torch.tensor(np.fromfile('/home/igli/tokens_of_interest', dtype=np.int32), device=configuration['device'])[:100]
torch.manual_seed(0)
toi = toi[torch.randperm(len(toi))] # Shuffling the contents of TOI (hoping that memory wont be abused if we parallelize)
already_computed = []
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

# toi = torch.tensor([0,0,0,0, 846, 708, 974, 1378])
# toi = torch.tensor([846, 708, 974, 1378])
tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b')
Q = quantizer_4b_torch()