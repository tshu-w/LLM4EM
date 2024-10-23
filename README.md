<div align="center">
  <h2 id="llm4em">Match, Compare, or Select? An Investigation of Large Language Models for Entity Matching</h2>
  <img align=middle src="https://github.com/tshu-w/LLM4EM/assets/13161779/6b776084-2312-44cd-8572-eda8205f628b" alt="Three strategies for LLM-based entity matching." width="45%">
  <img align=middle src="https://github.com/tshu-w/LLM4EM/assets/13161779/41790e40-db87-4061-8442-0383402865b2" alt="Compound EM framework" width="45%">
</div>

## Description

Entity matching (EM) is a critical step in entity resolution (ER). Recently, entity matching based on large language models (LLMs) has shown great promise. However, current LLM-based entity matching approaches typically follow a binary matching paradigm that ignores the global consistency between record relationships. In this paper, we investigate various methodologies for LLM-based entity matching that incorporate record interactions from different perspectives. Specifically, we comprehensively compare three representative strategies: matching, comparing, and selecting, and analyze their respective advantages and challenges in diverse scenarios. Based on our findings, we further design a compound entity matching framework (ComEM) that leverages the composition of multiple strategies and LLMs. ComEM benefits from the advantages of different sides and achieves improvements in both effectiveness and efficiency. Experimental results on 8 ER datasets and 9 LLMs verify the superiority of incorporating record interactions through the selecting strategy, as well as the further cost-effectiveness brought by ComEM.

## How to run
First, install dependencies and prepare the data
```console
# clone project
git clone https://github.com/YourGithubName/your-repository-name
cd LLM4EM

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
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```
