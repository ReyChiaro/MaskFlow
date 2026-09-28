<img src="assets/readme/maskflow-header.svg" width="100%" alt="MaskFlow" />

# 🌊 MaskFlow: Precise, Consistent and Seamless Regional Image Editing

<p align="center">
  <a href="https://reychiaro.github.io/MaskFlow"><img src="https://img.shields.io/badge/Project_Page-7C3AED?logo=googlechrome&amp;logoColor=white" alt="Project Page" /></a>
  <a href="https://arxiv.org/abs/2608.06929"><img src="https://img.shields.io/badge/arXiv-Paper-751D38?logo=arxiv&amp;logoColor=white" alt="arXiv Paper" /></a>
  <a href="https://github.com/ReyChiaro/MaskFlow"><img src="https://img.shields.io/badge/GitHub-Code-E38DA7?logo=github&amp;logoColor=white" alt="GitHub Code" /></a>
  <a href="https://huggingface.co/ReyChiaro/MaskFlow"><img src="https://img.shields.io/badge/Hugging_Face-Model-EFD046?logo=huggingface&amp;logoColor=white" alt="Hugging Face Model" /></a>
  <a href="https://huggingface.co/datasets/ReyChiaro/MaskEdit-10k"><img src="https://img.shields.io/badge/Hugging_Face-Dataset-EFD046?logo=huggingface&amp;logoColor=white" alt="Hugging Face Dataset" /></a>
  <a href="https://github.com/ModelTC/LightX2V"><img src="https://img.shields.io/badge/LightX2V-Demo-67A7E8?logo=github&amp;logoColor=white" alt="LightX2V Demo" /></a>
</p>

> ## Overview
> 🌊 <u>**Models**</u>: This repository is the official implementation for paper "MaskFlow: Precise, Consistent and Seamless Regional Image Editing", including:
> - Pipelines
> - Schedulers
> - DataModule
> - Trainers
> - Evaluators
> - Editor
>
> 🎨 <u>**Dataset**</u>: MaskEdit-10k is available on [🤗 Hugging Face](https://huggingface.co/datasets/ReyChiaro/MaskEdit-10k).

> 💜 <u>**Local Editor**</u>: Editor is relsead for pratical **user-specified** and **freeform** masks designing, the editor can be deployed on the local and server.
>
> 🩵 <u>**Demo (Comming soon)**</u>: MaskFlow is integrated into [LightX2V](https://github.com/ModelTC/LightX2V) for an accessible inference workflow.

> ⭐️ **Please leave your star if these can help you to create attractive artworks** ⭐️


## Contents

- [Overview](#overview)
- [Introduction](#introduction)
- [Editor](#editor)
- [Quick Start](#quick-start)
  - [1. Set up the environment](#1-set-up-the-environment)
  - [2. Prepare the inputs](#2-prepare-the-inputs)
  - [3. Choose a checkpoint](#3-choose-a-checkpoint)
  - [4. Run standard 50-step inference](#4-run-standard-50-step-inference)
- [Visualization](#visualization)
- [License](#license)
- [Citation](#citation)

<a id="introduction"></a>

## 🌊 Introduction

MaskFlow is a mask-aware framework for precise regional image editing. Given a source image, a spatial mask, and a text instruction, it edits the selected region while preserving the surrounding content. Its localized generation process and Soft-Poisson refinement improve regional control, background consistency, and boundary quality.

<a id="editor"></a>

## 🪄 [New] Editor

We release the image editor for convenient usage, supporting:

- **Freeform Mask:** User can draw masks on the source image with any shapes to identify the editable region. The masks can also be saved for future use!
- **Online Inference:** The editor can be deployed on the server to share the convenience to more people.

![editor-demo](assets/readme/editor-demo.png)

### Deployment

After environment is ready, just run `uv run python -m editor`, and this editor will deployed on `http://127.0.0.1:7890` on the local by default.

<a id="quick-start"></a>

## 🍪 Quick Start

### 1. Set up the environment

MaskFlow requires Python 3.12 or later. An NVIDIA GPU is recommended for inference.

```bash
git clone https://github.com/ReyChiaro/MaskFlow.git
cd MaskFlow

# Install uv if it is not already available.
python -m pip install uv

# Reproduce the locked Python 3.12 environment.
uv python install 3.12
uv sync
```

<details>
<summary>Alternative installation with venv and pip</summary>

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install a PyTorch build compatible with your CUDA environment if the automatically resolved build does not match your system.

</details>

### 2. Prepare the inputs

Prepare a source RGB image and a spatially aligned mask. White pixels in the mask indicate the region to edit; black pixels indicate the region to preserve. If a prompt comes from MaskEdit-10k and contains `[MASK_AREA]`, replace the placeholder with a natural referring phrase before inference.

### 3. Choose a checkpoint

Use `latest.safetensors` for general scenes and `latest-I.safetensors` for infographics. These are standard SFT LoRA adapters; load one as `maskflow`. The older S/SEC checkpoints remain available for their matching distilled adapters. Do not substitute a latest adapter for the SFT adapter required by an older distilled checkpoint.

The LoRA adapters are hosted in [`ReyChiaro/MaskFlow`](https://huggingface.co/ReyChiaro/MaskFlow). Diffusers downloads and caches the selected files automatically.

| File | Variant | Steps | Text CFG | Intended use |
|---|---|---:|---:|---|
| `latest.safetensors` | Latest · general scenes | 50 | 4.0 | General scene editing |
| `latest-I.safetensors` | Latest · infographics | 50 | 4.0 | Infographic editing |
| `maskflow-S.safetensors` | S | 50 | 4.0 | Standard checkpoint trained on the `scene` split |
| `maskflow-S-tcfg4-step8.safetensors` | S distilled | 8 | 4.0 | Accelerated scene editing |
| `maskflow-S-tcfg4-step16.safetensors` | S distilled | 16 | 4.0 | Accelerated scene editing |
| `maskflow-SEC.safetensors` | SEC | 50 | 4.0 | Standard checkpoint trained on all MaskEdit-10k splits |
| `maskflow-SEC-tcfg4-step8.safetensors` | SEC distilled | 8 | 4.0 | Accelerated general editing |
| `maskflow-SEC-tcfg4-step16.safetensors` | SEC distilled | 16 | 4.0 | Accelerated general editing |

`S` denotes training on the scene split, while `SEC` denotes training on all scene and infographic splits. A distilled LoRA is a residual adapter: it must be used with its matching standard SFT LoRA (`S` with `S`, or `SEC` with `SEC`). The SFT adapter is loaded as `maskflow`, and the distilled adapter is loaded as `dmd`.

### 4. Run standard 50-step inference

```bash
uv run python inference.py \
  input.source=/absolute/path/to/source.png \
  input.mask=/absolute/path/to/mask.png \
  'input.prompt=Replace the masked object with a red ceramic vase.' \
  checkpoint.sft_path=ReyChiaro/MaskFlow \
  checkpoint.sft_weight_name=latest.safetensors \
  runtime.num_inference_steps=50 \
  runtime.text_cfg_scale=4.0 \
  output.path=outputs/result.png
```

For infographics, use `checkpoint.sft_weight_name=latest-I.safetensors` with the same command.

The base model defaults to [`Qwen/Qwen-Image-Edit-2511`](https://huggingface.co/Qwen/Qwen-Image-Edit-2511), and the output directory is created automatically.

## Visualization

### General scene editing: object removal

The first comparison from Appendix E.1.4 shows removal within the masked region. The masked source is shown first and MaskFlow appears in the final panel.

![General scene object removal comparison](assets/readme/scene-removal.jpg)

### Infographic editing: object replacement

The first comparison from Appendix E.2.3 shows replacement of a selected element in an infographic. The comparison includes the masked source, baseline outputs, and MaskFlow in the final panel.

![Infographic object replacement comparison](assets/readme/infographics-replacement.jpg)

## License

MaskFlow code and adapter weights are released under the [MIT License](LICENSE). Use of the Qwen base model and third-party datasets remains subject to their respective licenses and terms.

## Citation

```bibtex
@misc{xu2026maskflowpreciseconsistentseamless,
  title={MaskFlow: Precise, Consistent and Seamless Regional Image Editing},
  author={Rui Xu and Yang Yong and Shunzi Yang and Ruihao Gong and Chengtao Lv},
  year={2026},
  eprint={2608.06929},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2608.06929},
}
```
