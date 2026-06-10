# Test
import os
import numpy as np
import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM
from datasets import load_from_disk

import threading
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

import torch.cuda.nvtx as nvtx
import time


# Shared resources
condition = threading.Condition()
shared_list = []

TOI = np.fromfile("gemma3_TOI", dtype=np.int32)
devices = ('cuda:3', 'cuda:4', 'cuda:5')
model   = Gemma3ForCausalLM.from_pretrained("google/gemma-3-1b-pt", trust_remote_code=True, cache_dir="/image-generation/imlloja", device_map = devices[0])  
teacher = Gemma3ForCausalLM.from_pretrained("google/gemma-3-4b-pt", trust_remote_code=True, cache_dir="/image-generation/imlloja", device_map = devices[1])  
tokenizer   = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt", trust_remote_code=True, cache_dir="/image-generation/imlloja")
ds = load_from_disk("/image-generation/imlloja/filtered-math")

VOCABSIZE = tokenizer.vocab_size

neuron_counts = torch.ones(model.lm_head.weight.shape[0], dtype=torch.int32)

configuration = {
    'teacher_model' : teacher,
    'student_model' : model,
    'optimizer' : lambda x : torch.optim.Adam(x, lr=0.001),
    'log_step' : 10,
    'log_path' : 'tensorboards',
    'experiment_name' : 'condvar_nonvtx',
    'neuron_counts' : neuron_counts,
    'noise_init_fn' : lambda x : torch.nn.init.normal_(x, mean=0, std=0.03),
    'line_length' : 1000,
    'batch_size' : 4,
    'inflated_decoder_sizes' : {
        'cutoffs' : [0,1],
        'counts'  : [   1]
    }
}

for i in range(len(configuration['inflated_decoder_sizes']['cutoffs'])-1):
    counts, cutoffs = configuration['inflated_decoder_sizes']['counts'], configuration['inflated_decoder_sizes']['cutoffs']
    configuration['neuron_counts'][cutoffs[i] : cutoffs[i+1]] = counts[i]

loader = torch.utils.data.DataLoader(ds, batch_size=configuration['batch_size'], shuffle=False)
scaler = GradScaler()


'''
# def teacher_loop(dataloader, configuration):
    # nvtx.range_push("teacher_loop")
#     teacher_model = configuration['teacher_model']
#     for i, batch in enumerate(dataloader):
        # nvtx.range_push(f"teacher_step_{i}")
#         lines = batch['text']
#         batch = tokenizer(lines, padding=True, truncation=True, return_tensors="pt")
#         input_ids = batch['input_ids']
#         attention_mask = batch['attention_mask']

#         with condition:
#             while shared_list:
#                 condition.wait()
#             with torch.no_grad():
#                 logits = teacher_model(input_ids.to(teacher_model.device), attention_mask.to(teacher_model.device), return_dict=True, use_cache=False)['logits']
#                 teacher_dist = logits.softmax(dim=-1)
                torch.cuda.synchronize()
#             shared_list.append((teacher_dist, input_ids, attention_mask, i))
#             condition.notify()
        # nvtx.range_pop()
#         if i > 30:
#             break
#     with condition:
#         shared_list.append(None)
#         condition.notify()
    # nvtx.range_pop()


# def student_loop(configuration):

#     def normalize(x): return x / x.sum(dim=-1, keepdim=True)

#     writer = SummaryWriter(f"{configuration['log_path']}/{configuration['experiment_name']}")
#     student_model = configuration['student_model']
#     decoder_device = devices[2]
#     counts = configuration['neuron_counts']
#     cum_neurons = torch.nn.functional.pad(torch.cumsum(counts, dim=-1), (1,0), mode='constant', value=0)

#     with torch.no_grad():
#         data = torch.repeat_interleave(student_model.lm_head.weight.data, counts.to(devices[0]), dim=0).to(device=decoder_device, dtype=torch.float32)
#     inflated_decoder = torch.nn.Linear(data.shape[1], data.shape[0], bias=False).to(decoder_device)
#     inflated_decoder.weight.data.copy_(data)

#     optimizer = configuration['optimizer'](inflated_decoder.parameters())

#     while True:
#         with condition:
#             while not shared_list:
#                 condition.wait()

#             data = shared_list.pop(0)
#             if data is None:
#                 break
#             teacher_dist, input_ids, attention_masks, i = data
#             condition.notify()

        # nvtx.range_push(f"student_step_{i}")

#         optimizer.zero_grad()
        torch.cuda.synchronize()

#         with torch.no_grad():
            # nvtx.range_push("forward_backbone")
#             backbone_out = student_model.model(input_ids.to(student_model.device), attention_masks.to(student_model.device), return_dict=True, use_cache=False)
            # nvtx.range_pop()

        # nvtx.range_push("prepare_decoder_input")
#         inflated_input = backbone_out.last_hidden_state.to(dtype=torch.float32, device=decoder_device).contiguous()
        torch.cuda.synchronize()
        # nvtx.range_pop()

        # nvtx.range_push("decoder_forward")
#         inflated_logits = inflated_decoder(inflated_input)
        torch.cuda.synchronize()
        # nvtx.range_pop()

        # nvtx.range_push("normalize_logits")
#         with torch.no_grad():
#             stabilizers = inflated_logits.max(dim=-1, keepdim=True).values.detach()
#         inflated_logits = inflated_logits - stabilizers
        torch.cuda.synchronize()
        # nvtx.range_pop()

        # nvtx.range_push("compute_distribution")
#         acc_logits = torch.exp(inflated_logits).cumsum(dim=-1)
#         acc_logits = torch.nn.functional.pad(acc_logits, (1,0), mode='constant', value=0)
#         distribution = acc_logits[..., cum_neurons[1:]] - acc_logits[..., cum_neurons[:-1]]
#         distribution = distribution / acc_logits[..., -1:]
        torch.cuda.synchronize()
        # nvtx.range_pop()

        # nvtx.range_push("kl_and_backward")
#         tempi = torch.clamp(distribution, min=1e-12)
#         tempt = torch.clamp(teacher_dist.to(tempi.device), min=1e-12)
#         kl = torch.nn.functional.kl_div(normalize(tempi)[...,:VOCABSIZE].log(), normalize(tempt)[...,:VOCABSIZE])

#         kl.backward()
        torch.cuda.synchronize()
#         torch.nn.utils.clip_grad_value_(inflated_decoder.parameters(), clip_value=1e-5)
#         optimizer.step()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # nvtx.range_pop()

#         if i % configuration['log_step'] == 0:
            # nvtx.range_push("logging")
#             original_logits = student_model.lm_head(backbone_out.last_hidden_state).softmax(dim=-1).detach()
#             original_loss = torch.nn.functional.kl_div(normalize(original_logits)[...,:VOCABSIZE].log().to(tempt.device), normalize(tempt)[...,:VOCABSIZE])
#             writer.add_scalar('original loss', original_loss, i)
#             writer.add_scalar('improvement', original_loss - kl.item(), i)
            # nvtx.range_pop()
#             del original_logits, original_loss

#         del teacher_dist, input_ids, attention_masks, backbone_out
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # nvtx.range_pop()
'''
# @profile
# # import torch.cuda.nvtx as nvtx

def teacher_loop(dataloader, configuration):
    # nvtx.range_push("teacher_loop")
    teacher_model = configuration['teacher_model']
    for i, batch in enumerate(dataloader):
        # nvtx.range_push(f"teacher_step_{i}")
        lines = batch['text']
        batch = tokenizer(lines, max_length=configuration['line_length'], padding=True, truncation=True, return_tensors="pt")
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        with torch.no_grad():
            logits = teacher_model(input_ids.to(teacher_model.device), attention_mask.to(teacher_model.device), return_dict=True, use_cache=False).logits
            teacher_dist = logits.softmax(dim=-1)
            # torch.cuda.synchronize()

        with condition:
            while shared_list:
                condition.wait()

            shared_list.append((teacher_dist, input_ids, attention_mask, i))
            # torch.cuda.synchronize()
            # torch.cuda.empty_cache()
            condition.notify()
        # nvtx.range_pop()
        if i > 230:
            break

    with condition:
        shared_list.append(None)
        condition.notify()
    # nvtx.range_pop()

# @profile
def student_loop(configuration):
    def normalize(x): return x / x.sum(dim=-1, keepdim=True)
    
    writer = SummaryWriter(f"{configuration['log_path']}/{configuration['experiment_name']}")
    student_model = configuration['student_model']
    decoder_device = devices[2]
    counts = configuration['neuron_counts']
    cum_neurons = torch.nn.functional.pad(torch.cumsum(counts, dim=-1), (1,0), mode='constant', value=0)

    with torch.no_grad():
        data = torch.repeat_interleave(student_model.lm_head.weight.data, counts.to(devices[0]), dim=0).to(device=decoder_device, dtype=torch.float32)
    inflated_decoder = torch.nn.Linear(data.shape[1], data.shape[0], bias=False).to(decoder_device)
    inflated_decoder.weight.data.copy_(data)

    optimizer = configuration['optimizer'](inflated_decoder.parameters())

    # nvtx.range_push("student_loop")
    while True:
        with condition:
            while not shared_list:
                condition.wait()

            data = shared_list.pop(0)
            if data is None:
                break
            teacher_dist, input_ids, attention_masks, i = data
            condition.notify()
        # nvtx.range_push(f"student_step_{i}")
        optimizer.zero_grad()
        # torch.cuda.synchronize()

        with torch.no_grad():
            # nvtx.range_push("backbone_forward")
            backbone_out = student_model.model(input_ids.to(student_model.device), attention_masks.to(student_model.device), return_dict=True, use_cache=False)
            # torch.cuda.synchronize()
            # nvtx.range_pop()

        # nvtx.range_push("Transfer data to decoder")
        inflated_input = backbone_out.last_hidden_state.to(dtype=torch.float32, device=decoder_device).contiguous()
        # torch.cuda.synchronize()
        # nvtx.range_pop()
        

        # nvtx.range_push("decoder_forward")
        inflated_logits = inflated_decoder(inflated_input)
        # torch.cuda.synchronize()
        # nvtx.range_pop()

        # nvtx.range_push("Logit normalization")
        with torch.no_grad():
            stabilizers = inflated_logits.max(dim=-1, keepdims=True).values.detach()
        inflated_logits = inflated_logits - stabilizers
        # torch.cuda.synchronize()
        # nvtx.range_pop()

        # nvtx.range_push("Logit reduce opertation")
        acc_logits = torch.exp(inflated_logits).cumsum(dim=-1)
        acc_logits = torch.nn.functional.pad(acc_logits, (1,0), mode='constant', value=0)
        distribution = acc_logits[..., cum_neurons[1:]] - acc_logits[..., cum_neurons[:-1]]
        distribution = distribution / acc_logits[..., -1:]
        # torch.cuda.synchronize()
        # nvtx.range_pop()

        # nvtx.range_push("Loss computation")
        tempi = torch.clamp(distribution, min=1e-12)
        tempt = torch.clamp(teacher_dist.to(tempi.device), min=1e-12)
        kl = torch.nn.functional.kl_div(normalize(tempi[...,:VOCABSIZE]).log(), normalize(tempt[...,:VOCABSIZE]))
        # nvtx.range_pop()

        # nvtx.range_push("Backward pass")
        kl.backward()
        # torch.cuda.synchronize()
        # nvtx.range_pop()

        # nvtx.range_push("Update step")
        torch.nn.utils.clip_grad_value_(inflated_decoder.parameters(), clip_value=1e-5)
        optimizer.step()
        # torch.cuda.synchronize()
        # torch.cuda.empty_cache()
        # nvtx.range_pop()

        if i % configuration['log_step'] == 0:
            # nvtx.range_push("Original decoder step")
            original_logits = student_model.lm_head(backbone_out.last_hidden_state).softmax(dim=-1).detach()
            # torch.cuda.synchronize()
            # nvtx.range_pop()

            # nvtx.range_push("Original loss computation")
            original_loss = torch.nn.functional.kl_div(normalize(original_logits[...,:VOCABSIZE]).log().to(tempt.device), normalize(tempt[...,:VOCABSIZE]))
            # torch.cuda.synchronize()
            # nvtx.range_pop()

            writer.add_scalar('original loss', original_loss, i)
            writer.add_scalar('improvement', original_loss - kl.item(), i)

        del teacher_dist, input_ids, attention_masks, backbone_out
        # torch.cuda.synchronize()
        # torch.cuda.empty_cache()

        # nvtx.range_pop()

    # nvtx.range_pop()
def main(config):
    torch.cuda.cudart().cudaProfilerStart()

    t1 = threading.Thread(target=teacher_loop, args=(loader, config))
    t2 = threading.Thread(target=student_loop, args=(config,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    torch.cuda.cudart().cudaProfilerStop()


if __name__ == '__main__':
    main(configuration)