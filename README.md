Semantic Entity Retrieval Toolkit
=================================

The Semantic Entity Retrieval Toolkit (SERT) is a collection of neural entity retrieval algorithms. Currently, it hosts an implementation of the log-linear model for expertise retrieval, published at [WWW 2016](http://chri.stophr.be/WWW2016-VanGysel.pdf).

__NOTE__: the implementation of the neural latent vector space model (to appear at CIKM2016) will be made available soon.

Prerequisites
-------------

SERT requires Python 2.7 and assorted [modules](requirements.txt). If you wish to train your models on GPGPUs, you will need a GPU compatible with [Theano](http://deeplearning.net/software/theano/).

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

If you use SERT to produce results for your scientific publication, please refer to [this paper](http://chri.stophr.be/WWW2016-VanGysel.pdf):

```
@inproceedings{VanGysel2016-WWW,
  title={Unsupervised, Efficient and Semantic Expertise Retrieval},
  author={Van Gysel, Christophe and de Rijke, Maarten and Worring, Marcel},
  booktitle={WWW},
  volume={2016},
  year={2016},
  organization={The International World Wide Web Conferences Steering Committee}
}
```

License
-------

SERT is licensed under the [MIT license](LICENSE). If you modify SERT in any way, please link back to this repository.