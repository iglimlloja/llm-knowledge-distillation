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
import time
import subprocess
LOCATION = '/igli/raw_logits_v2/'
device = 'cuda:0'

LOCATION = "/igli/raw_logits_v2"
t = list(map(lambda x : x.decode('utf-8').strip(), subprocess.run(["find", LOCATION, "-name", "*_embeddings_new"], stdout=subprocess.PIPE).stdout.split()))
tt = [os.path.basename(x)[:4] for x in t]

tokens_of_interest = np.fromfile('tokens_of_interest', dtype=np.int32)

nfiles=1
limit_per_file = 2**16
ind = np.zeros((len(tokens_of_interest), limit_per_file, 3), dtype=np.int16) 
counts = np.zeros(len(tokens_of_interest), dtype=np.uint32)
filemapper = np.array([f"{0:04d}_{token_id}_index" for token_id in tokens_of_interest])
ind_map = {token_id : i for i, token_id in enumerate(tokens_of_interest)}

def update(indexes, i, j):
    file_ids = np.zeros(len(indexes), dtype=np.uint16)        
    file_ids[np.arange(len(indexes))] = [int(filemapper[idx][:4]) for idx in indexes]
    file_ids = file_ids.reshape(-1, 1)
    to_inject = np.concatenate((file_ids, np.repeat([[i,j]], len(indexes), axis=0)), axis=-1)
    ind[indexes, counts[indexes]] = to_inject
    counts[indexes] += 1
        
def check_flush_tokens(recently_updated):
    to_flush = (counts[recently_updated] == limit_per_file).nonzero()[0]
    for flushable in to_flush:
        file = filemapper[recently_updated[flushable]]
        # with open(file, "w") as f:
        #     ind[recently_updated[flushable]].tofile(f)
        newFile_name = f"{int(file[:4])+1:04d}{file[4:]}"
        print("closing", recently_updated[flushable])
        # file.close()
        filemapper[recently_updated[flushable]] = newFile_name
        counts[recently_updated[flushable]] = 0
t1 = time.time()

for file_id_start in tqdm(range(0, nfiles, 50)):
    tokens=np.zeros( (1000,1024,11,50),dtype=np.int32)
    filenames =  tt[file_id_start:file_id_start+50]
    for j, jj in tqdm(enumerate(filenames)):
        ts=np.fromfile(f"{LOCATION}/{jj}_id_t" , dtype=np.int32).reshape(-1, 1024, 100)[..., :10]
        gs=np.fromfile(f"{LOCATION}/{jj}_id_gt", dtype=np.int32).reshape(1000, 1024)
        tokens[:,:,0:10,j]=ts
        tokens[:,:,-  1,j]=gs

    file_id   =np.tile(np.array([int(x)for x in filenames],dtype=np.uint16).reshape( (1   ,   1,1,50) ), (1000,1024,11,  1) )
    example_id=np.tile(np.arange(1000,                     dtype=np.uint16).reshape( (1000,   1,1, 1) ), (   1,1024,11, 50) )
    position  =np.tile(np.arange(1024,                     dtype=np.uint16).reshape( (1   ,1024,1, 1) ), (1000,   1,11, 50) )
    
    tokens_cuda = torch.tensor(tokens).to('cuda:3').reshape(-1)
    toi_cuda = torch.tensor(tokens_of_interest).to('cuda:3')
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    IX=torch.argsort(tokens_cuda)#.cpu().numpy()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    reshuffled = tokens_cuda[IX]
    del tokens_cuda
    torch.cuda.empty_cache()
    mask = torch.isin(reshuffled, toi_cuda)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    file_id   =file_id   .reshape(-1)[IX][mask.cpu()]
    example_id=example_id.reshape(-1)[IX][mask.cpu()]
    position  =position  .reshape(-1)[IX][mask.cpu()]
    tokens    =tokens    .reshape(-1)[IX][mask.cpu()]
    