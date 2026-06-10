import torch
import numpy as np

fnumbers = [f"{i:04d}" for i in range(2600)]

"""
There are two possible designs for this step:
1. Keep track of logprobs
        Whenever we query a value, we exp, divide by sum
        Whenever we query a value for trainig we take the log after querying
        Whenever we update sums, we subtract exp(q_i) and we add exp(q_i+y_hat)
        Whenever we update data, we add y_hat to q_i
2. Keep track of probs
        Whenever we query a value we just divide by the sum
        Whenever we query a value for trainig we take the log after querying
        Whenever we update sums, we subtract q_i and we add q_i*exp(y_hat)      
        Whenever we update data, we multiply q_i by exp(y_hat)    
"""
class BookKeeper():
    def __init__(self, student_probs):
        self.student_logprobs = student_probs.log() # Shape (num_examples, 100)
        
        # Denominators of softmax
        self.S = student_probs.sum(dim=-1)          # Shape (num_examples)
        
        # This can be used if we have asynchronous updates to the sums (i.e. train all tokens with no communication)
        self.lazy_deltas = torch.zeros_like(self.S)       
        self.kl_divs = torch.zeros_like(self.S)
    
    def update_kl_divs(self, ix, new_kl):
        self.kl_divs[ix] = new_kl
    def lazy_update(self, ix, new_vals):
        self.lazy_deltas[ix] += new_vals
    
    def get(self, i, j): # returns probability q_i,j
        return torch.exp(self.student_logprobs[i,j]) / self.S[i]
    def update(self, i,j, y_hat):
        old_logprobs = self.student_logprobs[i, j]
        self.S[i] = self.S[i] - torch.exp(old_logprobs) + torch.exp(old_logprobs+y_hat)
        old_logprobs[:] += y_hat

class BookKeeperProbs():
    def __init__(self, student_probs):
        self.student_probs = student_probs() # Shape (num_examples, 100)
        
        # Denominators of softmax
        self.S = student_probs.sum(dim=-1)          # Shape (num_examples)
        
        # This can be used if we have asynchronous updates to the sums (i.e. train all tokens with no communication)
        self.lazy_deltas = torch.zeros_like(self.S)       
        self.kl_divs = torch.zeros_like(self.S)
    
    def update_kl_divs(self, ix, new_kl):
        self.kl_divs[ix] = new_kl
    def lazy_update(self, ix, new_vals):
        self.lazy_deltas[ix] += new_vals
    
    def get(self, i, j): # returns probability q_i,j
        return self.student_probs[i,j] / self.S[i]
    def update(self, i,j, y_hat):
        old_logprobs = self.student_probs[i, j]
        self.S[i] = self.S[i] - (old_logprobs) + (old_logprobs*torch.exp(y_hat))
        old_logprobs[:] *= torch.exp(y_hat)

# Loading of the data from disk
def LoadData(file_batch):
    global fnumbers
    def _load_teacher(i):
        tensor_list = []
        mask_list   = []
        tidlist, gtlist = [], []
        for file_id in range(i*10, i*10+10):
            rank = file_batch%2
            index_file = torch.tensor(np.fromfile(f'/sanDisk{rank+1}/raw_logits_v3/{fnumbers[file_id]}_student_index_new', dtype=np.uint16),device=configuration['device']).reshape(-1,2).int()
            tid = torch.tensor(np.fromfile(f'/sanDisk{rank+1}/raw_logits_v3/{fnumbers[file_id]}_id_t',  dtype=np.int32), device=configuration['device']).reshape(1000, 1024, 100)[index_file[:, 0], index_file[:, 1], :10].reshape(-1,10)
            gt  = torch.tensor(np.fromfile(f'/sanDisk{rank+1}/raw_logits_v3/{fnumbers[file_id]}_id_gt', dtype=np.int32), device=configuration['device']).reshape(1000, 1024     )[index_file[:, 0], index_file[:, 1]]    
            mask = torch.isin(tid, toi).any(dim=-1) + torch.isin(gt, toi)
            mask_list.append(mask)
            tensor_list.append(index_file[mask])
            tidlist.append(tid[mask].to('cpu'))
            gtlist.append(gt[mask].to('cpu'))
        
    def _load_student(i):
        pass

    top100_teacher_ids, teacher_logits = _load_teacher(file_batch) # Shape (num_examples, 100)
    teacher_probs = torch.softmax(teacher_logits)                  # Shape (num_examples, 100)
    student_probs = _load_student(file_batch)                      # Shape (num_examples, 100)
    BK = BookKeeper(student_probs)

# Doing the forward step for the new batch of files
def Synchronize():
    ## QUESTION: Does it make more sense to update S after every token forward
    ##           or is it better to aggregate the differences in the end
    for token_id in toi:
        ix, iy = get_indices_of_interest(top100_teacher_ids, token_id)
        student_emb, student_probs = load_student(batch, token_id) # Shape (num_examples, 2048), (num_examples, 100)
        assert student_probs == BK.get(ix, iy)
        y_hats, losses = refiners[token_id](student_emb, student_probs)
        new_vals = student_probs * torch.exp(y_hats)
        BK.update(ix, iy, new_vals)

def Train():
    for token_id in toi:
        ix, iy = get_indices_of_interest(top100_teacher_ids, token_id)
        student_emb, student_probs = load_student(batch, token_id) # Shape (num_examples, 2048), (num_examples, 100)
        assert student_probs == BK.get(ix, iy)
        y_hats = refiners[token_id](student_emb, student_probs)
        new_vals = student_probs * torch.exp(y_hats)
        BK.update(ix, iy, new_vals)

def train_batch(i):
    LoadData()
    Synchronize()
    Train()