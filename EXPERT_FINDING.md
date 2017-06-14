Unsupervised, Efficient and Semantic Expertise Retrieval
===

Usage
-----

To replicate the experiments of the paper on **unsupervised and semantic expertise finding**, have a look at [this script](W3C-expert-finding.sh) which builds a log-linear model on the [W3C collection](http://research.microsoft.com/en-us/um/people/nickcr/w3c-summary.html). The script then evaluates the model on the [2005](http://trec.nist.gov/data/t14_enterprise.html) and [2006](http://trec.nist.gov/data/t15_enterprise.html) editions of TREC Enterprise track.

    [cvangysel@ilps SERT] ./W3C-expert-finding.sh <path-to-W3C-corpus> <path-to-nonexisting-temporary-directory>

    Verifying W3C corpus.

    Creating output directory.

    Fetching topics and relevance judgments.

    Constructing log-linear model on W3C collection.

    Evaluating on TREC Enterprise tracks.
    2005 Enterprise Track: ndcg=0.5474; map=0.2603; recip_rank=0.6209; P_5=0.4098;
    2006 Enterprise Track: ndcg=0.7883; map=0.4937; recip_rank=0.8834; P_5=0.7000;

Citation
--------

If you use SERT to produce results for your scientific publication, please refer to our [WWW 2016](https://arxiv.org/abs/1608.06651) paper on expert finding, our [ICTIR 2016](http://chri.stophr.be) paper on structural regularities in text-based vector spaces and our [software overview](https://arxiv.org/abs/1706.03757) paper:

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

@inproceedings{VanGysel2017entityregularities,
  title={Structural Regularities in Text-based Entity Vector Spaces},
  author={Van Gysel, Christophe and de Rijke, Maarten and Kanoulas, Evangelos},
  booktitle={ICTIR},
  volume={2017},
  year={2017},
  organization={ACM}
}

@article{VanGysel2017sert,
  title = "{Semantic Entity Retrieval Toolkit}",
  author={Van Gysel, Christophe and de Rijke, Maarten and Kanoulas, Evangelos},
  journal={arXiv preprint arXiv:1706.03757},
  year={2017},
  url={https://arxiv.org/abs/1706.03757},
}
```