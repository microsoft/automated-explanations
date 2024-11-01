<h1 align="center">   <img src="https://microsoft.github.io/augmented-interpretable-models/auggam_gif.gif" width="25%"> Automated explanations <img src="https://microsoft.github.io/augmented-interpretable-models/auggam_gif.gif" width="25%"></h1>

<p align="center"> A generative framework to bridge data-driven models and scientific theories in language neuroscience (<a href="https://arxiv.org/abs/2305.09863">arXiv 2024</a>)
</p>
<p align="center"> Explaining black box text modules in natural language with language models (<a href="https://arxiv.org/abs/2305.09863">arXiv 2023</a>)
</p>

This repo contains code to reproduce the experiments in the [GEM-V paper](https://arxiv.org/abs/2305.09863) and [the SASC paper](https://arxiv.org/abs/2305.09863). SASC takes in a text module and produces a natural explanation for it that describes what it types of inputs elicit the largest response from the module (see Fig below). GEM-V tests tests this in detail in an fMRI setting.

<img src="https://microsoft.github.io/automated-explanations/fig.svg?sanitize=True&kill_cache=1" width="90%">

SASC is similar to the nice [concurrent paper](https://github.com/openai/automated-interpretability) by OpenAI, but simplifies explanations to describe the function rather than produce token-level activations. This makes it simpler/faster, and makes it more effective at describing semantic functions from limited data (e.g. fMRI voxels) but worse at finding patterns that depend on sequences / ordering.


For a simple scikit-learn interface to use SASC, use the [imodelsX library](https://github.com/csinva/imodelsX). Install with `pip install imodelsx` then the below shows a quickstart example.

```python
from imodelsx import explain_module_sasc
# a toy module that responds to the length of a string
mod = lambda str_list: np.array([len(s) for s in str_list])

# a toy dataset where the longest strings are animals
text_str_list = ["red", "blue", "x", "1", "2", "hippopotamus", "elephant", "rhinoceros"]
explanation_dict = explain_module_sasc(
    text_str_list,
    mod,
    ngrams=1,
)
```

# Reference
- See related [fMRI experiments](https://github.com/csinva/fmri)
- Built from [this template](https://github.com/csinva/cookiecutter-ml-research)

```r
@misc{antonello2024generativeframeworkbridgedatadriven,
      title={A generative framework to bridge data-driven models and scientific theories in language neuroscience}, 
      author={Richard Antonello and Chandan Singh and Shailee Jain and Aliyah Hsu and Jianfeng Gao and Bin Yu and Alexander Huth},
      year={2024},
      eprint={2410.00812},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.00812}, 
}

@misc{singh2023explaining,
      title={Explaining black box text modules in natural language with language models}, 
      author={Chandan Singh and Aliyah R. Hsu and Richard Antonello and Shailee Jain and Alexander G. Huth and Bin Yu and Jianfeng Gao},
      year={2023},
      eprint={2305.09863},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```
