# LLM Knowledge Distillation

A teacher-student knowledge distillation system designed to correct a small student LLM's output distribution token-by-token, using lightweight per-token residual networks trained on compressed student embeddings.

## Motivation

The student model (Gemma-2B / Gemma-3-1B) is already a reasonable next-token predictor, but has systematic per-token biases relative to a larger teacher (Gemma-7B / Gemma-3-4B). Rather than fine-tuning the student's weights directly, this project trains a small corrector for each vocabulary token that learns to predict the **residual** between the teacher's probability and the student's probability, conditioned on the student's last-layer embedding. At inference time, these cheap correctors adjust the student's logits without touching the base model.

## Pipeline Overview

1. **Data Collection** — Run teacher and student models on a math dataset. For each token position, save the teacher's top-100 logits and the student's last-layer embedding. Four parallel instances (one per GPU) handle different file shards.
2. **Indexing** — Build an inverted index mapping each vocabulary token to all dataset positions where it appeared as the ground truth. This enables per-token data retrieval during training.
3. **Compression** — Student embeddings (2048-dim float32) are compressed using whitening (PCA-based rotation to decorrelate dimensions) followed by mixed-precision quantization (1/8 of dims at 4-bit, 4/8 at 2-bit, 3/8 at 1-bit), reducing each embedding from 2048 floats to ~480 stored values. This makes storing millions of embeddings on disk tractable.
4. **Residual Training** — For each vocabulary token, a small neural network (`multi_embedding`, with 100–1024 clusters and linear projection layers) is trained to predict the residual between teacher and student probabilities given the compressed student embedding. Loss is KL divergence or L1/L2 on log-probabilities.
5. **Parallel / Multi-GPU Training** — A `BookKeeper` data structure maintains running refined output distributions per token, updated asynchronously across multiple GPUs. GCP variants adapt this for cloud infrastructure using a threaded producer-consumer pattern that separates teacher inference, student inference, and decoder training into independent threads.

## Directory Structure

```
llm-knowledge-distillation/
├── data_collection/
│   ├── data_collection.ipynb         Main notebook for running teacher+student inference and saving raw logits/embeddings
│   ├── data_collection_script.py     Script version of the above; extracts teacher top-100 logits and quantized student embeddings to binary files
│   ├── build_dataset_index.py        Builds confusion matrices from model outputs and constructs inverted index structures mapping tokens to file locations
│   ├── build_token_index.py          GPU-accelerated token indexing: loads logits from batches of files, sorts by token ID, maps to per-token index files
│   └── build_inverted_index.ipynb    Notebook that processes raw logit files and constructs the final inverted index (token → [file_id, example_id, position])
│
├── compression/
│   ├── quantizer_numpy.ipynb         NumPy implementation of 1/2/4-bit quantizers, mixed-precision quantizer, and PCA-based whitening transform; tests on real 661K×2048 embedding data
│   └── quantizer_torch.ipynb         PyTorch/GPU version of the same quantization and whitening pipeline; includes error heatmaps and compression ratio analysis
│
├── training/
│   ├── utils.py                      Core shared utilities: quantizer classes, whitening transform, multi_embedding module, ParallelTrainer, LoggingManager, BookKeeper, TokenIndexer
│   ├── trainer.py                    Per-token residual trainer (v1): loads compressed embeddings and teacher logits for a specific token, trains multi_embedding with KL/L1/L2 loss
│   ├── trainer_v2.py                 Revised trainer with updated loss formulation and training loop improvements
│   ├── parallel_trainer.py           Multi-GPU parallel training framework; BookKeeper tracks per-token student probabilities and KL divergences across asynchronous producer processes
│   ├── pseudocode.py                 Design notes and pseudocode for the BookKeeper update algorithm (both logprob-based and prob-based variants)
│   ├── gemma3_token_selection.ipynb  Identifies high-frequency tokens of interest from the math dataset; initializes the inflated decoder vocabulary expansion module
│   ├── final_trainer.ipynb           End-to-end threaded distillation pipeline integrating teacher inference, student inference, and inflated decoder training with KL divergence loss
│   ├── dataset_loader.ipynb          Dataset class for loading compressed embeddings (4-bit quantized) and teacher logits; handles whitening reconstruction and probability computation
│   ├── residual_trainer_prototype.ipynb  Early prototype: multi-cluster embedding model trained on residuals (teacher_probs − student_probs) with L1/L2/KL losses
│   ├── residual_probability_trainer.ipynb  Trains multi_embedding on per-token residual probabilities using SGD; extracts and validates per-token datasets
│   └── whitening_residual_probability_trainer.ipynb  Same as above but uses whitened embeddings as input to the residual network
│
├── distillation_gcp/
│   ├── distillation_script.py        Main GCP distillation script: InflatedDecoder with configurable initialization (centered/fresh/noise), KL loss with cumulative logit reduction
│   ├── sparse_residual_distillation_script.py  Alternative GCP distillation script using a SparseResidualDecoder (optional LayerNorm, grouped-softmax via cumulative-sum, residual injection on top of the student's own distribution); sweeps multiple decoder configs in parallel
│   ├── profiling.py                  Profiling harness for the teacher-student pipeline on GCP; measures throughput of the producer-consumer training loop
│   ├── gemma3_gcp.ipynb              GCP-adapted notebook: token frequency analysis on the math dataset, inflated decoder setup, and training loop
│   └── debugging.ipynb               Debugging notebook: compares multiple decoder configurations and sparse residual decoders with layer norm; also explores modeling the teacher−student residual with regularized least squares over engineered features (square roots, pairwise products, binary rank indicators, VIF analysis)
│
└── analysis/
    ├── statistical_analysis.ipynb    Token frequency analysis across the full dataset; rank-frequency plots (log-log), top-10K token identification, linear regression on embeddings
    ├── visualize_per_token_loss.ipynb  Visualizes Llama-2-7B prediction distributions as color-coded probability heatmaps; builds global token co-occurrence confusion matrices
    ├── two_stage_cluster_decoding.ipynb  Explores efficient candidate-token decoding over the large vocabulary: spherical/Euclidean k-means clustering of the teacher lm_head, a cascaded TwoStageClusterDecoder (coarse → fine routing), and several aggregation strategies; measures whether the ground-truth top-k tokens survive into the kept candidate pool (recall@k / coverage). Also sketches graph-traversal and Hadamard-hashing retrieval variants
    └── gradient_structure_analysis.ipynb  Low-rank gradient probe: hooks the final-layer MLP gate_proj to capture (input, output, grad) and verifies the per-example rank-1 structure of the weight gradient (G·Xᵀ), assessing feasibility of low-rank / compressed gradient updates
```

## Notes on versioning

- **Two distillation scripts are kept intentionally.** `distillation_gcp/distillation_script.py` and `distillation_gcp/sparse_residual_distillation_script.py` are *not* duplicate versions of the same file — they implement two different decoder designs (the `InflatedDecoder`, which repeats each vocabulary logit into a fixed number of neurons, vs. the `SparseResidualDecoder`, which learns a residual correction on top of the student's own distribution). Both are retained because they represent distinct experimental approaches.
- **`debugging.ipynb` is the fuller version.** The notebook here is the complete debugging session (it is a strict superset of an earlier 28-cell extract), so only the more recent/complete copy is kept to avoid a near-duplicate.
