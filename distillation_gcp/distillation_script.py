import os
import numpy as np
import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM
from datasets import load_from_disk

import threading
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

GCP = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

TOI = np.fromfile("gemma3_TOI", dtype=np.int32)
if GCP:
    model = Gemma3ForCausalLM.from_pretrained("google/gemma-3-1b-pt", trust_remote_code=True, cache_dir="/image-generation/imlloja", device_map='cuda:0')  
    teacher = Gemma3ForCausalLM.from_pretrained("google/gemma-3-4b-pt", trust_remote_code=True, cache_dir="/image-generation/imlloja", device_map='cuda:1')  
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt", trust_remote_code=True, cache_dir="/image_generation/imlloja")
    ds = load_from_disk("/image_generation/imlloja")
else:
    model = Gemma3ForCausalLM.from_pretrained("google/gemma-3-1b-pt", trust_remote_code=True, cache_dir="/nvmes/A4", device_map='cuda:0')  
    teacher = Gemma3ForCausalLM.from_pretrained("google/gemma-3-4b-pt", trust_remote_code=True, cache_dir="/nvmes/A4", device_map='cuda:1')  
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt", trust_remote_code=True, cache_dir="/nvmes/A4")
    ds = load_from_disk("/home/igli/.cache/huggingface/datasets/filtered-math")

loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
scaler = GradScaler()

VOCABSIZE = tokenizer.vocab_size

class InflatedDecoder(torch.nn.Module):
    def __init__(self,old_decoder, counts, mode='fresh', noise_init_fn = None):
        super().__init__()
        with torch.no_grad():
            if noise_init_fn:
                noise = noise_init_fn(torch.empty(counts.sum().item(), old_decoder.weight.data.size(1))).to('cuda:2')*0.2
            if mode == "centered":
                data = torch.repeat_interleave(old_decoder.weight.data, counts.to('cuda:0'), dim=0).to(device='cuda:2', dtype=torch.float32)
            if mode == "fresh":
                data = torch.empty((counts.sum().item(), old_decoder.weight.data.size(1)), dtype=torch.float32)
                torch.nn.init.kaiming_uniform_(data, a=torch.math.sqrt(5))
            if mode == "noise":
                data = torch.empty((counts.sum().item(), old_decoder.weight.data.size(1)), dtype=torch.float32)
                torch.nn.init.zeros_(data)
            if noise_init_fn:
                data += noise
        self.weights = torch.nn.Parameter(data)
        self.bias = None 
    def forward(self, x):
        x = x.to(dtype=torch.float32, device=self.weights.device).contiguous()
        return torch.nn.functional.linear(x, self.weights, self.bias)

def loss_fn(input_dist, target_dist):
    def normalize(x):
        return x / x.sum(dim=-1, keepdim=True)
    ## TODO: Implement the loss fn
    # In our old implementation we were computing this in the forward.
    input_dist = torch.clamp(input_dist, min=1e-12)
    target_dist = torch.clamp(target_dist, min=1e-12)
    with autocast(dtype=torch.float32):
        target_dist = target_dist.to(input_dist.device)
        return torch.nn.functional.kl_div(normalize(input_dist[...,:VOCABSIZE]), normalize(target_dist[...,:VOCABSIZE]))

def get_teacher_distribution(model, input_ids, attention_mask):
    with torch.no_grad():
        out = model.forward(input_ids, attention_mask, return_dict=True, use_cache=False, output_hidden_states=False)
        return out['logits'].softmax(dim=-1)

def get_student_distribution(student_model, inflated_decoder, cummulative_neuron_counts, input_ids, attention_mask):
    with torch.no_grad():
        backbone_out = student_model.model(input_ids, attention_mask, return_dict=True,use_cache=False)
        torch.cuda.synchronize(device=backbone_out.last_hidden_state.device)

    
    with autocast(dtype=torch.float32):
        with torch.cuda.device('cuda:2'):
            inflated_input = backbone_out.last_hidden_state.to(dtype=torch.float32, device=inflated_decoder.weights.device).contiguous()
            try:
                inflated_logits = inflated_decoder(inflated_input)
            except RuntimeError as e:
                raise
            torch.cuda.set_device('cuda:2')
            torch.cuda.synchronize()

        with torch.no_grad():
            stabilizers = inflated_logits.max(dim=-1, keepdims=True).values.detach()
        inflated_logits = inflated_logits - stabilizers
        accummulated_logits = torch.exp(inflated_logits).cumsum(dim=-1)
        assert cummulative_neuron_counts[0] == 0
        assert cummulative_neuron_counts[-1] == accummulated_logits.shape[-1] 
        accummulated_logits = torch.nn.functional.pad(accummulated_logits, (1,0), mode='constant', value=0)
    
        distribution = accummulated_logits[..., cummulative_neuron_counts[1:]] - accummulated_logits[..., cummulative_neuron_counts[:-1]]
        distribution = distribution / accummulated_logits[..., -1:]

        del accummulated_logits, stabilizers
        torch.cuda.empty_cache()
        return backbone_out.last_hidden_state, distribution

def learn_one_step(student_model, optimizer, distributions, inflated_decoder, cummulative_neuron_counts,input_ids,attention_mask, student_forward_results, teacher_dist):
    optimizer.zero_grad()
    with autocast(dtype=torch.float32):
        hidden_states, student_distribution = get_student_distribution(student_model, inflated_decoder, cummulative_neuron_counts, input_ids, attention_mask)
        loss = loss_fn(student_distribution, teacher_dist)
    loss.backward()
    torch.nn.utils.clip_grad_value_(inflated_decoder.parameters(), clip_value=1e-5)
    optimizer.step()

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    student_forward_results.append((hidden_states, loss.item()))

def teacher_prepare(model, input_ids, attention_mask):
    return get_teacher_distribution(model, input_ids.to(model.device), attention_mask.to(model.device))

def teacher_loop(dataloader, configuration):
    teacher_model = configuration['teacher_model']
    for i,batch in enumerate(dataloader):
        with condition:
            while shared_list:
                condition.wait()
            print("Teacher Loop : Shared list length =", len(shared_list))
            lines = batch['text']
            batch = tokenizer(lines, max_length=512, padding=True, truncation=True, return_tensors="pt")
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            teacher_dist = teacher_prepare(teacher_model, input_ids, attention_mask)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            shared_list.append((teacher_dist, input_ids, attention_mask, i))
            condition.notify()
    with condition:
        shared_list.append(None)
        condition.notify()

def student_loop(configuration):
    writer = SummaryWriter(f"{configuration['log_path']}/{configuration['experiment_name']}")
    student_model = configuration['student_model']
    cum_neurons = torch.nn.functional.pad(torch.cumsum(configuration['neuron_counts'], dim=-1), (1,0), mode='constant', value=0)
    inflated_decoder = InflatedDecoder(student_model.lm_head, configuration['neuron_counts'], mode="centered", noise_init_fn=configuration['noise_init_fn']).to('cuda:2')
    optimizer = configuration['optimizer'](inflated_decoder.parameters())

    while True:
        with condition:
            while not shared_list:
                condition.wait()
            print("Student Loop : Shared list length =", len(shared_list))
            data = shared_list.pop(0)

            if data is None:
                break
            teacher_dist, input_ids, attention_masks, i = data
            condition.notify()
        student_results = []
        learn_one_step(student_model, optimizer, (None), inflated_decoder, cum_neurons, input_ids.to(student_model.device), attention_masks.to(student_model.device), student_results, teacher_dist)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if i % configuration['log_step'] == 0:
            student_results = student_results.pop(0)
            original_distributions = student_model.lm_head(student_results[0]).softmax(dim=-1).detach()
            original_loss = loss_fn(original_distributions, teacher_dist)
            inflated_decoder_loss = student_results[1]

            writer.add_scalar('original loss', original_loss, i)
            writer.add_scalar('improvement', original_loss - inflated_decoder_loss, i)
        del teacher_dist, input_ids, attention_masks, student_results
        torch.cuda.empty_cache()
        
condition = threading.Condition()
shared_list = []
def main():
    neuron_counts = torch.ones(VOCABSIZE, dtype=torch.int32)
    neuron_counts[TOI[    : 200]] = 500
    neuron_counts[TOI[ 200:1000]] = 100
    neuron_counts[TOI[1000:5000]] = 50
    configuration = {
        'teacher_model' : teacher,
        'student_model' : model,
        'optimizer' : lambda x : torch.optim.Adam(x, lr=0.001),
        'log_step' : 10,
        'log_path' : 'gcptest',
        'experiment_name' : 'progressive_clustering/noisy_centered_looseClamp',
        'neuron_counts' : neuron_counts,
        'noise_init_fn' : lambda x : torch.nn.init.normal_(x, mean=0, std=0.03)
    }
    t1 = threading.Thread(target=teacher_loop, args=(loader, configuration))
    t2 = threading.Thread(target=student_loop, args=(configuration,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
