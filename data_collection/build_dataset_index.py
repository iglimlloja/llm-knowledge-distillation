import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import gc
import io 
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from torch import nn
from torch import optim
import pickle 
import multiprocessing 
from datasets import load_from_disk

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        global paths
        global epoch
        self.data = data
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        line=self.data[i]['text'][0:5000]
        batch  = self.tokenizer(line, max_length=1025, padding='max_length', truncation=True)
        batch['input_ids']=torch.tensor(batch['input_ids'])
        return batch

import multiprocessing
import os

def generate_index(token_id):
    global residuals, approximate_probs, T_ID, ID, path
    token_of_interest = token_id
    all3 = ((ID[:, 0][..., None, None] != 0) *(T_ID == token_of_interest)).nonzero()
    ex, ps, top10id = all3[:,0], all3[:,1], all3[:,2]
    Y = residuals[ex, ps, top10id]
    P = approximate_probs[ex, ps, top10id]
    file_nums = ex // 1000 + 1
    
    locations = torch.concatenate((file_nums.unsqueeze(1), ps.unsqueeze(1)), dim=1).numpy().astype('uint16')
    locations.tofile(os.path.join(path, f'{token_of_interest:06d}_locations'))
    
    data = torch.concatenate((Y.unsqueeze(1), P.unsqueeze(1)), dim=1).numpy()
    data.tofile(os.path.join(path, f'{token_of_interest:06d}_data'))
    

    
    
def target_fn(rank, input_ids, models):
    new_input_ids = input_ids[i,...].unsqueeze(0).to(models[i].device)
    print(models[i](new_input_ids,return_dict=True,use_cache=False,output_hidden_states=False)['logits'])

def generate_confusion_data(model, dataset, nfiles = 1000, nbatches = 1000, batch_size = 4, top_k = 10, max_length = 1024, file_start = 0):
    LOCATION = "/media/igli/c8d7d852-a9fd-4353-be47-000a0fe82201/raw_logits"
    with torch.no_grad():
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        loop=tqdm(loader, leave=True)

        file_no = file_start 

        # Ground Truth IDs
        ID   = torch.zeros((nbatches*batch_size, max_length)).type(torch.int32).to(device=model.device)
        # Teacher Logits
        TL = torch.zeros((nbatches*batch_size, max_length, top_k)).type(torch.float16).to(device=model.device)
        # Teacher IDs
        T_ID   = torch.zeros((nbatches*batch_size, max_length, top_k)).type(torch.int32).to(device=model.device)

        for (i,batch) in enumerate(loop):
            if i > 0 and i % nbatches == 0:
                with open(os.path.join(LOCATION,f"{file_no:04d}_id_gt.pkl"),   "wb") as f: pickle.dump(ID, f)
                with open(os.path.join(LOCATION,f"{file_no:04d}_id_t.pkl"),    "wb") as f: pickle.dump(T_ID, f)
                with open(os.path.join(LOCATION,f"{file_no:04d}_logit_t.pkl"), "wb") as f: pickle.dump(TL, f)
                
                file_no += 1
                if file_no == nfiles: break

                ID   = torch.zeros((nbatches*batch_size, max_length)).type(torch.int32).to(device=model.device)
                TL   = torch.zeros((nbatches*batch_size, max_length, top_k)).type(torch.float16).to(device=model.device)
                T_ID = torch.zeros((nbatches*batch_size, max_length, top_k)).type(torch.int32).to(device=model.device)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            i = i % nbatches
            labels = batch['input_ids'][..., 1:].clone().detach()
            input_ids      = batch['input_ids'].to(model.device)
            output=model.forward(input_ids,return_dict=True,use_cache=False,output_hidden_states=False)

            logits = output['logits'][..., :-1, :]
            sort_results = torch.topk(logits, top_k, dim=-1).indices
            ID  [i*batch_size : (i+1)*batch_size] = labels
            T_ID[i*batch_size : (i+1)*batch_size] = sort_results
            TL  [i*batch_size : (i+1)*batch_size] = logits[torch.arange(batch_size).reshape(batch_size,1,1), torch.arange(max_length).reshape(1,max_length,1), sort_results]

def generate_student_data(student_model, nfiles, minibatch_size=6):
    LOCATION = "/media/igli/c8d7d852-a9fd-4353-be47-000a0fe82201/raw_logits"

    embedding_size = student_model.lm_head.in_features
    for file_no in tqdm(range(nfiles)):
        with open(os.path.join(LOCATION, f"{file_no:04d}_id_gt.pkl"),   "rb") as f: ID = pickle.load(f).to(student_model.device)
        with open(os.path.join(LOCATION, f"{file_no:04d}_id_t.pkl"),    "rb") as f: T_ID = pickle.load(f).to(student_model.device)
        with open(os.path.join(LOCATION, f"{file_no:04d}_logit_t.pkl"), "rb") as f: TL = pickle.load(f).to(student_model.device)

        # Student Logits
        SL = torch.zeros_like(TL, device=student_model.device)
        # Student Embeddings
        SE = torch.zeros((SL.size(0), SL.size(1), embedding_size), device=student_model.device, dtype=torch.float16)
        
        batch_size = ID.size(0) # This is old_batch_size
        max_length = ID.size(1)

        start_tokens = torch.tensor(2, dtype=torch.int32).to(student_model.device).repeat(batch_size, 1)
        input_ids = torch.concatenate((start_tokens, ID), dim=1).to(student_model.device)

        # TODO: Loop through minibatches here if memory fails
        # minibatch_size = 16
        for minibatch in tqdm(range(0, batch_size, minibatch_size)):
            next_limit = min(minibatch+minibatch_size, batch_size)
            embeddings = student_model.model(input_ids[minibatch:next_limit], return_dict=True,use_cache=False,output_hidden_states=False)['last_hidden_state']
            output = student_model.lm_head(embeddings)[..., :-1, :]
            embeddings = embeddings[..., :-1, :]
            
            SL[minibatch : next_limit] = output[torch.arange(next_limit-minibatch).reshape(next_limit-minibatch,1,1), torch.arange(max_length).reshape(1,max_length,1), T_ID[minibatch:next_limit]]
            SE[minibatch : next_limit] = embeddings
    
        with open(os.path.join(LOCATION, f"{file_no:04d}_logit_s.pkl"), "wb") as f: pickle.dump(SL, f)
        buff = io.BytesIO()
        torch.save(SE, buff)
        with open(os.path.join(LOCATION,f"{file_no:04d}_emb_s")   , "wb") as f: f.write(buff.getbuffer())

        del SL
        del SE
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

def binarize16(filename):
    print("In file", filename)
    try:
        t = torch.load(filename)
        t = t.to('cpu').detach().numpy().astype(np.float16).tofile(filename)
    except:
        print("Error in", filename)

if __name__ == "__main__":
    # Collecting data
    LOCATION = "/igli/raw_logits"
    l=[]
    for file_no in tqdm(range(1,29)):
        with open(os.path.join(LOCATION,f"{file_no:04d}_id_gt.pkl"),   "rb") as f: l.append(pickle.load(f).cpu())
    ID=torch.concatenate(l,dim=0)
    l=[]
    for file_no in tqdm(range(1,29)):
        with open(os.path.join(LOCATION,f"{file_no:04d}_id_t.pkl"),   "rb") as f: l.append(pickle.load(f).cpu())
    T_ID = torch.concatenate(l, dim=0)
    l=[]
    for file_no in tqdm(range(1,29)):
        with open(os.path.join(LOCATION,f"{file_no:04d}_logit_s.pkl"),   "rb") as f: l.append(pickle.load(f).cpu())
    SL = torch.concatenate(l, dim=0)
    l=[]
    for file_no in tqdm(range(1,29)):
        with open(os.path.join(LOCATION,f"{file_no:04d}_logit_t.pkl"),   "rb") as f: l.append(pickle.load(f).cpu())
    TL = torch.concatenate(l, dim=0)
    
    multiprocessing.set_start_method('spawn')
    
    with multiprocessing.Pool(processes=64) as pool:
        pool.map(generate_index, range(256000))


