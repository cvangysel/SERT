Semantic Entity Retrieval Toolkit
=================================

The Semantic Entity Retrieval Toolkit (SERT) is a collection of neural entity retrieval algorithms.

Currently, it hosts an implementation of the following models:

   * the [log-linear model for expertise retrieval](EXPERT_FINDING.md), published at [WWW 2016](https://arxiv.org/abs/1608.06651)
   * the [latent vector space model for product search](PRODUCT_SEARCH.md), published at [CIKM 2016](https://arxiv.org/abs/1608.07253)

Prerequisites
-------------

SERT requires Python 3.5 and assorted [modules](requirements.txt). The [trec_eval](https://github.com/usnistgov/trec_eval) utility is required for evaluation and the end-to-end scripts. If you wish to train your models on GPGPUs, you will need a GPU compatible with [Theano](http://deeplearning.net/software/theano/).

Getting started
---------------

To begin, create a virtual Python environment and install dependencies:

    [cvangysel@ilps cvangysel] git clone git@github.com:cvangysel/SERT.git
    [cvangysel@ilps cvangysel] cd SERT

    [cvangysel@ilps SERT] virtualenv SERT-dev
    Using base prefix '/Users/cvangysel/anaconda3'
    New python executable in /home/cvangysel/SERT/SERT-dev/bin/python
    Installing setuptools, pip, wheel...done.

    [cvangysel@ilps SERT] source SERT-dev/bin/activate

    (SERT-dev) [cvangysel@ilps SERT] pip install -r requirements.txt

Afterwards, follow the examples for [expertise retrieval](EXPERT_FINDING.md) or [product search](PRODUCT_SEARCH.md).

Citation
--------

If you use SERT to produce results for your scientific publication, please refer to our [WWW 2016](https://arxiv.org/abs/1608.06651), [CIKM 2016](https://arxiv.org/abs/1608.07253), [ICTIR 2017](https://arxiv.org/abs/1707.07930) and [software overview](https://arxiv.org/abs/1706.03757) papers:

```
@inproceedings{VanGysel2016experts,
  title={Unsupervised, Efficient and Semantic Expertise Retrieval},
  author={Van Gysel, Christophe and de Rijke, Maarten and Worring, Marcel},
  booktitle={WWW},
  volume={2016},
  pages={1069--1079},
  year={2016},
  organization={The International World Wide Web Conferences Steering Committee}
}

@inproceedings{VanGysel2016products,
  title={Learning Latent Vector Spaces for Product Search},
  author={Van Gysel, Christophe and de Rijke, Maarten and Kanoulas, Evangelos},
  booktitle={CIKM},
  volume={2016},
  pages={165--174},
  year={2016},
  organization={ACM}
}

@inproceedings{VanGysel2017entityregularities,
  title={Structural Regularities in Text-based Entity Vector Spaces},
  author={Van Gysel, Christophe and de Rijke, Maarten and Kanoulas, Evangelos},
  booktitle={ICTIR},
  volume={2017},
  year={2017},
  organization={ACM}
}

@inproceedings{VanGysel2017sert,
  title={Semantic Entity Retrieval Toolkit},
  author={Van Gysel, Christophe and de Rijke, Maarten and Kanoulas, Evangelos},
  booktitle={SIGIR 2017 Workshop on Neural Information Retrieval (Neu-IR'17)},
  year={2017},
}
```

License
-------

SERT is licensed under the [MIT license](LICENSE). If you modify SERT in any way, please link back to this repository.