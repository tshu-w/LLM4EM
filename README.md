<div align="center">
  <h2 id="llm4em">Match, Compare, or Select? An Investigation of Large Language Models for Entity Matching</h2>
  <p>
    <a href="https://aclanthology.org/2025.coling-main.8/"><img src="http://img.shields.io/badge/COLING-2025-4b44ce.svg?style=flat-square" alt="Conference" /></a>
    <a href="https://arxiv.org/abs/2405.16884"><img src="http://img.shields.io/badge/arXiv-2405.16884-B31B1B.svg?style=flat-square" alt="Arxiv" /></a>
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
@inproceedings{wang-etal-2025-match,
    title = "Match, Compare, or Select? An Investigation of Large Language Models for Entity Matching",
    author = "Wang, Tianshu  and
      Chen, Xiaoyang  and
      Lin, Hongyu  and
      Chen, Xuanang  and
      Han, Xianpei  and
      Sun, Le  and
      Wang, Hao  and
      Zeng, Zhenyu",
    editor = "Rambow, Owen  and
      Wanner, Leo  and
      Apidianaki, Marianna  and
      Al-Khalifa, Hend  and
      Eugenio, Barbara Di  and
      Schockaert, Steven",
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.coling-main.8/",
    pages = "96--109",
}
```
