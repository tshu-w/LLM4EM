<div align="center">
  <h2 id="llm4em">Match, Compare, or Select? An Investigation of Large Language Models for Entity Matching</h2>
  <p>
    <a href="https://arxiv.org/abs/2405.16884"><img src="http://img.shields.io/badge/arXiv-2405.16884-B31B1B.svg?style=flat-square" alt="Arxiv" /></a>
    <a href=""><img src="http://img.shields.io/badge/COLING-2025-4b44ce.svg?style=flat-square" alt="Conference" /></a>
  </p>
  <img align=middle src="https://github.com/tshu-w/ComEM/assets/13161779/6b776084-2312-44cd-8572-eda8205f628b" alt="Three strategies for LLM-based entity matching." width="45%">
  <img align=middle src="https://github.com/tshu-w/ComEM/assets/13161779/41790e40-db87-4061-8442-0383402865b2" alt="Compound EM framework" width="45%">
</div>

## News

- [2024-12-01] 🎉 Our paper has been accepted at [COLING 2025](https://coling2025.org).

## Description

Entity matching (EM) is a critical step in entity resolution (ER). Recently, entity matching based on large language models (LLMs) has shown great promise. However, current LLM-based entity matching approaches typically follow a binary matching paradigm that ignores the global consistency between record relationships. In this paper, we investigate various methodologies for LLM-based entity matching that incorporate record interactions from different perspectives. Specifically, we comprehensively compare three representative strategies: matching, comparing, and selecting, and analyze their respective advantages and challenges in diverse scenarios. Based on our findings, we further design a compound entity matching framework (ComEM) that leverages the composition of multiple strategies and LLMs. ComEM benefits from the advantages of different sides and achieves improvements in both effectiveness and efficiency. Experimental results on 8 ER datasets and 9 LLMs verify the superiority of incorporating record interactions through the selecting strategy, as well as the further cost-effectiveness brought by ComEM.

## How to run
First, install dependencies and prepare the data
```console
# clone project
git clone https://github.com/tshu-w/ComEM.git
cd ComEM

# [SUGGESTED] use conda environment
conda env create -f environment.yaml
conda activate llm4em

# [ALTERNATIVE] install requirements directly
pip install -r requirements.txt

# prepare the data
git clone https://github.com/AI-team-UoA/pyJedAI data/pyJedAI
python src/blocking.py
```

Next, to obtain the main results of the paper:
```console
python src/{strategy}.py
```

## Citation
```
@article{DBLP:journals/corr/abs-2405-16884,
  author       = {Tianshu Wang and Hongyu Lin and Xiaoyang Chen and Xianpei Han
                  and Hao Wang and Zhenyu Zeng and Le Sun},
  title        = {Match, Compare, or Select? An Investigation of Large Language
                  Models for Entity Matching},
  journal      = {CoRR},
  year         = 2024,
  volume       = {abs/2405.16884},
  doi          = {10.48550/ARXIV.2405.16884},
  eprint       = {2405.16884},
  eprinttype   = {arXiv},
  url          = {https://doi.org/10.48550/arXiv.2405.16884},
  timestamp    = {Tue, 18 Jun 2024 16:10:22 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2405-16884.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
