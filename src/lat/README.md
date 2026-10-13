# Latent Adversarial Training (LAT) for Qwen2.5-7B-Instruct

This directory contains the LAT implementation adapted for **Qwen2.5-7B-Instruct** in our study of safety alignment under adaptive, multi-turn jailbreak attacks. The implementation is based on the original [Latent Adversarial Training repository](https://github.com/aengusl/latent-adversarial-training) associated with *Targeted Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs* (Sheshadri et al.).

We adapted the upstream implementation to support Qwen2.5 model internals, Qwen chat formatting, and fully offline training on an HPC cluster. The model resulting from training is stored as a **LoRA adapter**; it must be loaded together with the corresponding Qwen2.5-7B-Instruct base model.

## Repository contents

The following files are central to our adaptation:

| Path | Purpose |
| --- | --- |
| `latent_at/lat_methods.py` | Uses the model's configured hidden size rather than a hard-coded dimension. |
| `latent_at/lat_helpers.py` | Removes debugging breakpoints that would interrupt unattended training. |
| `latent_at/lat_datasets.py` | Adapts chat-template generation and beginning-of-sequence token handling for Qwen. |
| `notebooks/qwen25_jailbreaks.ipynb` | Notebook used to configure and run Qwen2.5 LAT with locally stored models and datasets. |

Other files in `latent_at/` are retained because the notebook depends on the upstream LAT implementation.

## Training configuration

The final LAT run used the following settings:

| Setting | Value |
| --- | --- |
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Training method | Targeted latent adversarial training with LoRA |
| Training steps | 100 |
| LoRA rank | 64 |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `down_proj` |
| Perturbed layers | `embedding`, `7`, `14`, `21`, `26` |
| PGD iterations | 16 |
| Model iterations | 4 |
| Perturbation budget (`epsilon`) | 6.0 |
| Inner learning rate | `1e-3` |
| Outer learning rate | `8e-5` |
| Batch size | 16 |
| Maximum batch per accumulation | 2 |
| Truncation length | 2048 tokens |
| `add_completions_pgd` | `True` |

The final 100-step training run took approximately **2 hours and 33 minutes** on the HPC system using an NVIDIA A100 80 GB GPU. This figure describes the training run only, not the total compute used for preparation, debugging, or evaluation.

For the executable configuration, **the notebook is the source of truth**. Verify that its saved cells match the configuration above before releasing or rerunning the experiment.

## Setup

The implementation was run with Python 3.11 and a CUDA-enabled PyTorch environment. Install the repository's dependencies in a compatible environment:

```bash
python -m pip install -r requirements.txt
```

The upstream project also supplies `install_tasks_from_github.sh`; if additional tasks or packages are required, install them **before** moving to an offline compute node. The original dependency file is not a fully pinned environment lockfile, so exact package versions and compatibility should be checked against the final training environment.

For reference, the experimental environment used PyTorch `2.7.0+cu126`; other package versions should be verified against the environment used for the final run rather than inferred from the upstream requirements file.

## Models and data

The compute nodes used for training did not have internet access. Accordingly, the following resources must be prepared **locally** before starting a GPU job:

1. A local copy of `Qwen2.5-7B-Instruct`, including tokenizer files.
2. Harmful and benign training data, prepared from the LAT data source and stored as local JSONL files.
3. All Python packages and additional resources needed by the notebook.

Update the model path, dataset paths, and output/checkpoint directory in `notebooks/qwen25_jailbreaks.ipynb` to match your environment. The original cluster-specific absolute paths are not portable. See the notebook for the exact JSONL fields consumed by the training code; this README does not assume a different dataset schema.

No pretrained model weights, training datasets, or final adapter weights are distributed in this directory.

## Running LAT training

From the root of this LAT directory, activate the prepared environment and launch the notebook:

```bash
# Optional: explicitly disable online Hugging Face lookups.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline

jupyter notebook notebooks/qwen25_jailbreaks.ipynb
```

Before executing the training cells:

1. Select the correct Python environment/kernel.
2. Set the local Qwen2.5 model and JSONL dataset paths in the notebook.
3. Set a writable checkpoint/output directory.
4. Confirm the LAT/LoRA settings against the configuration table above.
5. Run the notebook on a compute node with sufficient GPU memory.

The final trained artifact is a LoRA adapter. Load it with the matching base model when running downstream evaluations; do not treat it as an independently loadable full model.

## Changes relative to upstream

- **Model dimensions:** Replaced a fixed hidden-state dimension with `model.config.hidden_size` in `latent_at/lat_methods.py`.
- **Unattended execution:** Removed debugging `breakpoint()` statements in `latent_at/lat_helpers.py`.
- **Qwen formatting:** Updated the dataset/prompt handling in `latent_at/lat_datasets.py` to use the Qwen chat-generation prefix (`add_generation_prompt=True`) and handle tokenizers with no BOS token.
- **Offline workflow:** Adapted `notebooks/qwen25_jailbreaks.ipynb` to use local models, local JSONL datasets, and Qwen-specific training settings.

## Reproducibility notes

- This directory documents **LAT training**, not the full multi-turn jailbreak evaluation pipeline. Consult the main project repository for the evaluation procedure, attacker/judge models, and benchmark results.
- The final experiment used one training run; the reported training time is not a comprehensive GPU-hour budget.
- For exact reproduction, record the upstream LAT Git commit, the version of the modified code, dependency versions, and dataset preparation steps.
- The notebook and code were adapted for an offline cluster environment; paths must be adjusted before running elsewhere.

## Attribution and license

This code is derived from [aengusl/latent-adversarial-training](https://github.com/aengusl/latent-adversarial-training), developed for the work by Sheshadri et al., *Targeted Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs*.

The upstream repository is released under the **MIT License**. If you redistribute substantial portions of its code, retain the original copyright and license notice in the accompanying `LICENSE` file. Models and datasets may have **separate licenses and usage terms**.

Upstream revision used: **[add the actual commit hash before submission]**.
