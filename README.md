<div align="center">

<h2 id="llm4em">Match, Compare, or Select? An Investigation of Large Language Models for Entity Matching</h2>

</div>

## Description
Entity matching (EM) is a critical step in entity resolution. Recently, entity matching based on large language models (LLMs) has shown great promise. However, current LLM-based entity matching approaches typically follow a binary matching paradigm that ignores the global consistency between different record relationships. In this paper, we investigate various methodologies for LLM-based entity matching that incorporate record interactions from different perspectives. Specifically, we comprehensively compare three representative strategies: matching, comparing, and selecting, and analyze their respective advantages and challenges in diverse scenarios. Based on our findings, we further design a compound entity matching framework (ComEM) that leverages the composition of multiple strategies and LLMs. In this way, ComEM can benefit from the advantages of different sides and achieve improvements in both effectiveness and efficiency. Experimental results verify that ComEM not only achieves significant performance gains on various datasets, but also reduces the cost of LLM-based entity matching for practical applications.

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
