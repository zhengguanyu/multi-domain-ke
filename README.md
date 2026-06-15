# ASMem: Anchor Sparse Memory for Multi-Domain Knowledge Editing of LLMs

Official code release for the paper *ASMem: Anchor Sparse Memory for Multi-Domain Knowledge Editing of LLMs*.

## Repository layout

```
Multi-domain/
├── easyeditor/  
│   └── models/asmem/
├── hparams/          
├── data/             
│   ├── HalluEditBench/   
│   └── CKnowEdit/        
└── experiments/      
```

## Quick start

1. Install dependencies.
2. Download the base LLMs (links below) and update the `model_name` field in the relevant YAML under `hparams/`.
3. Download the precomputed Wikipedia covariance statistics (see **Stats** below) and set `stats_dir` in the YAML.
4. Run, e.g.:
   ```bash
   python experiments/run_editing.py --dataset hallu --method ASMem \
       --hparams_dir hparams/ASMem/cloud-llama3-8b.yaml
   ```

## LLMs

### LLaMA series
- **LLaMA-3-8B-Instruct** — https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
- **LLaMA-3.2-1B-Instruct** — https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
- **LLaMA-3.2-3B-Instruct** — https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

Access requires accepting Meta's license on the Hugging Face model page.

### Qwen series
- **Qwen2.5-7B-Instruct** — https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- **Qwen2.5-0.5B-Instruct** — https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- **Qwen2.5-1.5B-Instruct** — https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- **Qwen2.5-3B-Instruct** — https://huggingface.co/Qwen/Qwen2.5-3B-Instruct

### SBERT
- **all-MiniLM-L6-v2** — https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- **all-mpnet-base-v2**  — https://huggingface.co/sentence-transformers/all-mpnet-base-v2
- **bge-m3** — https://huggingface.co/BAAI/bge-m3

## Stats

Locate-then-edit baselines (ROME/ MEMIT / AlphaEdit / DeltaEdit / BLUE / RECT) and the null-space projection used by AlphaEdit all depend on precomputed second-moment statistics over a Wikipedia subset. These files are large and are distributed separately.

- **EN Download link: https://pan.baidu.com/s/1A5J-hiCm9qENYe1KZvZRXA** password-bagf

- **ZH Download link: https://pan.baidu.com/s/1A5J-hiCm9qENYe1KZvZRXA** password-bagf

After downloading, extract into a directory and point `stats_dir` (in each hparams YAML) at it. Expected contents:

- `Meta-Llama-3-8B-Instruct/` — layers 4–8 `.npz` covariance tensors
- `Qwen2.5-7B-Instruct/` — layers 4–8 `.npz` covariance tensors

## Datasets

Experiments use HalluEditBench and CKnowEdit. Dataset files are not bundled; obtain them from the original releases and place under `data/` (path configured per experiment script).

- **HalluEditBench**
  - Dataset: `https://github.com/baixianghuang/HalluEditBench`
  - Paper: `https://arxiv.org/abs/2410.16251`
- **CKnowEdit**
  - Dataset: `https://huggingface.co/datasets/zjunlp/CKnowEdit`
  - Paper: `https://aclanthology.org/2025.acl-long.430/`

## Citation

```bibtex
@article{zheng2026asmem,
  title={ASMem: Anchor Sparse Memory for Multi-Domain Knowledge Editing of Large Language Models},
  author={Zheng, Guanyu and Wang, Zhenyu and Zhao, Yang and He, Tingting and Wang, Xv and Wang, Haochang and Zhao, Tiejun and Zong, Chengqing},
  journal={Neural Networks},
  pages={109230},
  year={2026},
  publisher={Elsevier}
}
```

## Acknowledgements

This codebase builds on [EasyEdit](https://github.com/zjunlp/EasyEdit). Baseline implementations (AlphaEdit, DeltaEdit, NDEdit, ROME, MEMIT, BLUE, RECT, WISE, GRACE) are adapted from their respective official repositories.
