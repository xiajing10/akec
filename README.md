## 
## Introduction
Description of code and data for *Unsupervised Key-phrase Extraction and Clustering for Classification Scheme in Scientific Publications*.
## Requirements
###  Packages
- [tqdm](https://tqdm.github.io/)
- [AllenNLP](https://github.com/allenai/allennlp#installation)
- [pke](https://github.com/boudinfl/pke)
- [pytorch](https://pytorch.org/)
- [spacy](https://spacy.io/usage)
- [numpy](https://numpy.org/install/)
- [scispacy](https://allenai.github.io/scispacy/)
- [spherecluster](https://pypi.org/project/spherecluster/)

### Resources
- scispacy models: ["en_core_sci_sm"](https://github.com/allenai/scispacy#available-models)
- word embedding (save them to `./src/embed_data/`:
	- [ConceptNet Numberbatch 19.08 (English only)](https://github.com/commonsense/conceptnet-numberbatch#downloads)
	- [ELMo](https://allennlp.org/elmo) weights and options


## Description
- `./dataset/`: 
	- `ieee_xai.csv`: publication dataset collected from [IEEE xplore](https://ieeexplore.ieee.org/Xplore/home.jsp)
	- `domain_terms.txt`: domain glossary terms

- ` ./src/` :
	- `test.ipynb` has experiments demo
	- `kprank.py` has the main utility functions in keyphrase extraction
	- `clustering.py` has the main utility functions in keyphrase clustering
	- `param.py` defines the local links to recourses and data files required in code

## Reference
- [SIFRank: A New Baseline for Unsupervised Keyphrase Extraction Based on Pre-trained Language Model](https://github.com/sunyilgdx/SIFRank)
- [TaxoGen: Unsupervised Topic Taxonomy Construction by Adaptive Term Embedding and Clustering](https://github.com/franticnerd/taxogen)
