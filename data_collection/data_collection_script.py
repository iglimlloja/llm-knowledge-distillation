import torch
from torch.utils.data import DataLoader
import os
from datasets import load_dataset

import numpy as np
from tqdm import tqdm
import time

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers import LlamaForCausalLM, LlamaTokenizerFast
import transformers

from torch import nn
from torch import optim

from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
import pickle
import datasets
import threading

def asyncwrite(content, file):
    content.tofile(file)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        global paths
        global epoch
        self.data = data
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        line=self.data[i]['text'][0:8000]
        batch  = self.tokenizer(line, max_length=1025, padding='max_length', truncation=True)
        batch  = self.tokenizer(line, max_length=1025, padding='max_length', truncation=True)
        batch['input_ids']=torch.tensor(batch['input_ids'])
        return batch


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
        for i in range(0,a.shape[0],10000):
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


def v3student(student_model, tokens_of_interest, nfiles, start_file, rank, minibatch_size=8):
    tokens_of_interest = tokens_of_interest.to(student_model.device)

    for file_no in range(4*start_file+rank, nfiles, 4):
        ID   = torch.tensor(np.fromfile(os.path.join(LOCATION, f"{file_no:04d}_id_gt"), dtype=np.int32), device=student_model.device).reshape(1000, 1024)
        T_ID = torch.tensor(np.fromfile(os.path.join(LOCATION, f"{file_no:04d}_id_t"), dtype=np.int32), device=student_model.device).reshape(1000, 1024, 100)
        
        with open(os.path.join(LOCATION, f'{file_no:04d}_student_index_new'), 'ba') as index_file,\
             open(os.path.join(LOCATION, f'{file_no:04d}_student_embeddings_new'), 'ba') as embs_file,\
             open(os.path.join(LOCATION, f'{file_no:04d}_student_probs_new'), 'ba') as student_probs_file:
        
            batch_size = ID.size(0) # This is old_batch_size

            start_tokens = torch.tensor(2, dtype=torch.int32).to(student_model.device).repeat(batch_size, 1)
            input_ids = torch.concatenate((start_tokens, ID), dim=1).to(student_model.device)

            for minibatch in tqdm(range(0, batch_size, minibatch_size)):
                next_limit = min(minibatch+minibatch_size, batch_size)
                embeddings = student_model.model(input_ids[minibatch:next_limit], return_dict=True,use_cache=False,output_hidden_states=False)['last_hidden_state']
                output = student_model.lm_head(embeddings)[..., :-1, :]
                embeddings = embeddings[..., :-1, :]
                # Populating the files
                ind = torch.logical_or(
                    torch.isin(ID[minibatch:next_limit], tokens_of_interest), 
                    torch.isin(T_ID[minibatch:next_limit, :, :10], tokens_of_interest).any(dim=-1)
                    ).nonzero()
                
                # TODO: Quantize these   
                embs = W.encode(embeddings[ind[:,0], ind[:,1]])
                quantized = q4.quantize_compress(embs)
                # quantized.cpu().detach().numpy().tofile(embs_file)
                quantized.cpu().detach().numpy().tofile(embs_file)

                torch.softmax(output, dim=-1)[ind[:, 0].reshape(-1, 1), ind[:, 1].reshape(-1, 1), T_ID[minibatch:next_limit][ind[:,0], ind[:,1]]].cpu().to(torch.float16).detach().numpy().tofile(student_probs_file)
                ind[:,0] += minibatch
                
                ind.cpu().to(torch.int16).detach().numpy().tofile(index_file)
                # ind.cpu().detach().numpy().astype('int16').tofile(index_file)
                ind = ind.cpu()
                torch.cuda.empty_cache()
                # torch.cuda.synchronize()

def continue_from(path, rank):
    files = list(map(lambda x: int(x[:4]), filter(lambda x : ('student_embeddings_new' in x) and int(x[:4])%4 == rank, os.listdir(path))))
    files = sorted(files)
    last_file = files[-1]
    print(last_file)
    os.remove(os.path.join(path, f"{last_file:04d}_student_embeddings_new"))
    os.remove(os.path.join(path, f"{last_file:04d}_student_index_new"))
    os.remove(os.path.join(path, f"{last_file:04d}_student_probs_new"))
    return last_file // 4

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('rank', type=int)
    args = parser.parse_args()
    rank = args.rank
    LOCATION = "/igli/raw_logits_v2"
    start_file = continue_from(LOCATION, rank)
    # start_file=119
    print(start_file)
    torch. set_grad_enabled(False) 
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-7b')
    device = f'cuda:{rank}'
    gemma2 = AutoModelForCausalLM.from_pretrained('google/gemma-2b', torch_dtype=torch.float16,device_map=device)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    # texts = datasets.load_from_disk('~/.cache/huggingface/datasets/filtered-math')
    # dataset = Dataset(texts, tokenizer)
    W = representation_whitening_torch()
    W.load_state_dict("wtorch.pth")
    W.mn = W.mn.to(gemma2.device)
    W.q = W.q.to(gemma2.device)
    W.s = W.s.to(gemma2.device)
    q4 = quantizer_4b_torch()

    tokens_of_interest = torch.tensor(np.fromfile('tokens_of_interest', dtype=np.int32))
    v3student(gemma2, tokens_of_interest, 2662, start_file, rank, 8)

