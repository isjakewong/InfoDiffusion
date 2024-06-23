# [InfoDiffusion: Representation Learning Using Information Maximizing Diffusion Models](https://arxiv.org/abs/2306.08757)
By [Yingheng Wang](https://isjakewong.github.io/), [Yair Schiff](https://yair-schiff.github.io), [Aaron Gokaslan](https://skylion007.github.io), [Weishen Pan](https://vivo.weill.cornell.edu/display/cwid-wep4001),
[Fei Wang](https://wcm-wanglab.github.io/), [Chris De Sa](https://www.cs.cornell.edu/~cdesa/), [Volodymyr Kuleshov](https://www.cs.cornell.edu/~kuleshov/)

[![deploy](https://img.shields.io/badge/Blog%20%20-8A2BE2)]()
[![arXiv](https://img.shields.io/badge/arXiv-2406.07524-red.svg)](https://arxiv.org/abs/2306.08757)

<div align=center><img src="flowchart.drawio_-1.png" width="95%"></div>
<div align=center><img src="graphicalabstract.drawio_v2-1.png" width="75%"></div>

We introduce *MDLM*, a **M**asked discrete **D**iffusion **L**anguage **M**odel that features
a novel (SUBS)titution based
parameterization which simplifies the absorbing state diffusion
loss to a mixture of
classical masked language modeling losses. In doing so, we achieve
SOTA perplexity numbers on LM1B and OpenWebText among diffusion models while achiving competitive zero-shot perplexity with SOTA AR models on numerous datasets. We provide a demo in this [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18nC6q7dWq154fI1BXPLwmtnS7Zvbrv6p?usp=sharing/) notebook.


In this repo, we release:
* **The MDLM framework.**
  1. SUBStitution based parameterization
  2. Simplified loss calculation for masked diffusion processes
* **Baseline implementations** [[Examples]](#baselines):
  1. Autoregressive model that matches the SOTA AR performance on LM1B.
  2. Score Entropy Based Discrete Diffusion [SEDD](https://arxiv.org/abs/2310.16834).
  3. An efficient implementation of the absorbing state [D3PM](https://arxiv.org/abs/2107.03006) that beats the previous state of the art diffuision model SEDD on LM1B.
* **Samplers**
  1. Ancestral sampling as proposed in D3PM.
  2. Analytic sampler as proposed in SEDD.
  3. Our proposed efficient sampler that
     - makes MDLM **~3-4x** faster than the existing diffusion models. [[Example]](#sample-gen)
     - supports semi-autoregressive (SAR) generation.  [[Example]](#semi-ar-gen)

<a name="code-organization"></a>
## Code Organization
1. ```main.py```: Routines for training and evaluation
2. ```noise_schedule.py```: Noise schedules
3. ```diffusion.py```: Forward/reverse diffusion
4. ```dataloader.py```: Dataloaders
5. ```utils.py```: LR scheduler, logging, `fsspec` handling
6. ```models/```: Denoising network architectures. Supports [DiT](https://arxiv.org/abs/2212.09748), AR transformer, and [Mamba](https://arxiv.org/abs/2312.00752)
7. ```configs/```: Config files for datasets/denoising networks/noise schedules/LR schedules
8. ```scripts/```: Shell scripts for training/evaluation


<a name="getting_started"></a>

## Getting started in this repository

To get started, create a conda environment containing the required dependencies.

```bash
conda env create -f requirements.yaml
conda activate mdlm
```

Create the following directories to store saved models and slurm logs:
```bash
mkdir outputs
mkdir watch_folder
```
and run the training as a batch job:
```bash
sbatch scripts/train_owt_mdlm.sh
```

### Checkpoints

We have uploaded MDLM model trained on OpenWebText for 1M training steps to the Huggingface hub 🤗:
[kuleshov-group/mdlm-owt](https://huggingface.co/kuleshov-group/mdlm-owt)
Furthermore, we have released the checkpoints for the AR and SEDD baselines trained on OpenWebText in this [Google Drive folder](https://drive.google.com/drive/folders/16LuuptK7Xfk-vzhQYZBZ0SA-B-BFluau?usp=sharing).

## Reproducing Experiments

Below, we describe the steps required for reproducing the experiments in the paper.
Throughout, the main entry point for running experiments is the [`main.py`](./main.py) script.
We also provide sample `slurm` scripts for launching pre-training and downstream fine-tuning experiments in the [`scrips/`](./scripts) directory.


### Generate Samples
<a name="sample-gen"></a>
The argument to `sampling.predictor` specifies the sampler which takes one of the following values:
* `ddpm_cache`: our proposed sampler that's **~3-4x** faster than the samplers propsed in D3PM and SEDD.
* `ddpm`: Ancestral sampling proposed in D3PM.
* `analytic`: Analytic sampler proposed in SEDD.

In the following table we report wall clock time to generate 64 samples on a single A5000 GPU with `batch_size=1`. $T$ denotes the time discretization of the reverse process.
|                         | $T=5k (\downarrow)$ | $T=10k (\downarrow)$ |
|-------------------------|---------------------|----------------------|
| **SEDD**                | 127.1               | 229.3                |
| **MDLM** + `ddpm`       | 113.8               | 206.6                |
| **MDLM** +`ddpm_cache`  | **40.1**            | **60.4**             |


To generate samples from a pre-trained model use one of the following commands:
#### Huggingface model
```bash
python main.py \
  mode=sample_eval \
  eval.checkpoint_path=kuleshov-group/mdlm-owt \
  data=openwebtext-split  \
  model.length=1024  \
  sampling.predictor=ddpm_cache  \
  sampling.steps=1000 \
  loader.eval_batch_size=1 \
  sampling.num_sample_batches=10 \
  backbone=hf_dit
```
#### Local checkpoint
```bash
python main.py \
  mode=sample_eval \
  eval.checkpoint_path=/path/to/checkpoint/mdlm.ckpt \
  data=openwebtext-split  \
  model.length=1024  \
  sampling.predictor=ddpm_cache  \
  sampling.steps=10000 \
  loader.eval_batch_size=1 \
  sampling.num_sample_batches=1 \
  backbone=dit
```

### Semi-AR sample generation
<a name="semi-ar-gen"></a>
MDLM can also generate samples of arbitrary length in a semi-autoregressive (SAR) manner.
We generate 200 sequences of length 2048 tokens on a single `3090` GPU and evaluate generative perplexity under a pre-trained GPT-2 model. In the below table we find that in addition to achieving better generative perplexity, MDLM enables **25-30x** faster SAR decoding relative to [SSD-LM](https://arxiv.org/abs/2210.17432).

|                     | Gen. PPL ($\downarrow$) | Sec/Seq ($\downarrow$) |
|---------------------|-------------------------|------------------------|
| **SSD-LM**          | 35.43                   | 2473.9                 |
| **MDLM** +`ddpm_cache`  | **27.18**               | **89.3**               |

*Gen. PPL: Generation Perplexity, Sec/Seq: Seconds per Sequence*

```bash
python main.py \
  mode=sample_eval \
  eval.checkpoint_path=kuleshov-group/mdlm-owt \
  data=openwebtext-split \
  parameterization=subs \
  model.length=1024  \
  sampling.predictor=ddpm_cache  \
  sampling.steps=1000 \
  loader.eval_batch_size=1 \
  sampling.num_sample_batches=2 \
  sampling.semi_ar=True \
  sampling.stride_length=512 \
  sampling.num_strides=2 \
  backbone=hf_dit
```

### Train
To train MDLM from scratch on OpenWebText use the following command:
```
python main.py \
  model=small \
  data=openwebtext-split \
  wandb.name=mdlm-owt \
  parameterization=subs \
  model.length=1024 \
  eval.compute_generative_perplexity=True \
  sampling.steps=1000
```
The arguments `loader.batch_size` and `loader.eval_batch_size` allow you to control the global batch size and the batch size per GPU. If `loader.batch_size * num_gpus` is less than the global batch size, PyTorch Lightning will resort to gradient accumulation. You can also launch a training job on Slurm using the command: `sbatch scripts/train_owt_mdlm.sh`. The slurm scripts to train the Auto-regressive and SEDD baselines are as follows respectively: [`scripts/train_lm1b_ar.sh`](scripts/train_lm1b_ar.sh), [`scripts/train_owt_sedd.sh`](scripts/train_owt_sedd.sh).

### Eval 
To compute test perplexity, use `mode=ppl_eval`. Example scripts provided in `scripts/`. An example command for perplexity evaluation on OpenWebText is:
```
python main.py \
  mode=ppl_eval \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  data=openwebtext-split \
  model=small \
  parameterization=subs \
  backbone=dit \
  model.length=1024 \
  eval.checkpoint_path=/path/to/checkpoint/mdlm.ckpt \
  +wandb.offline=true
```

### Baseline evaluation
<a name="baselines"></a>
We release the checkpoints for the baselines: SEDD and AR trained on OpenWebText in this [Google Drive folder](https://drive.google.com/drive/folders/16LuuptK7Xfk-vzhQYZBZ0SA-B-BFluau?usp=sharing). Download the checkpoints: `ar.ckpt`, `sedd.ckpt` and use the following commands to compute test perplexity:
#### AR
```bash
python main.py \
  mode=ppl_eval \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  data=openwebtext-split \
  model=small-ar \
  parameterization=ar \
  backbone=ar \
  model.length=1024 \
  eval.checkpoint_path=/path/to/checkpoint/ar.ckpt \
  +wandb.offline=true
```
#### SEDD
```bash
python main.py \
  mode=ppl_eval \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  data=openwebtext-split \
  model=small \
  parameterization=sedd \
  backbone=dit \
  model.length=1024 \
  eval.checkpoint_path=/path/to/checkpoint/sedd.ckpt \
  time_conditioning=True \
  +wandb.offline=true
```

### Disclaimer
This research code is provided as-is, without any support or guarantee of quality. However, if you identify any issues or areas for improvement, please feel free to raise an issue or submit a pull request. We will do our best to address them.

## Citation
```
  @inproceedings{wang2023infodiffusion,
    title={Infodiffusion: Representation learning using information maximizing diffusion models},
    author={Wang, Yingheng and Schiff, Yair and Gokaslan, Aaron and Pan, Weishen and Wang, Fei and De Sa, Christopher and Kuleshov, Volodymyr},
    booktitle={International Conference on Machine Learning},
    pages={36336--36354},
    year={2023},
    organization={PMLR}
  }
```
