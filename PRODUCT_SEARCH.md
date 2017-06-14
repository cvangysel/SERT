Learning Latent Vector Spaces for Product Search
===

Data collection
----

We use the [Amazon product data](http://jmcauley.ucsd.edu/data/amazon) published by McAuley et al. at SIGIR 2015. You can obtain the data by following the [provided instructions](http://jmcauley.ucsd.edu/data/amazon/amazon_readme.txt).

We make use of the `Home_and_Kitchen`, `Clothing_Shoes_and_Jewelry`, `Pet_Supplies` and `Sports_and_Outdoors` reviews and metadata data files (full, not the 5-core).

Product lists, topics (i.e., textual representations of categories) and ground truth relevance information is provided in this repository. In addition, we provide utilities ([here](bin/amazon/amazon_products_to_trec.py) and [here](bin/amazon/amazon_reviews_to_trec.py)) to convert the descriptions and reviews of the domains to TREC text format.

Here's an overview of all the files included in this repository used for dataset construction and model evaluation (see below for more information on how to use these files).

|      | Home & Kitchen | Clothing, Shoes & Jewelry | Pet Supplies | Sports & Outdoors |
| ---- | ------------------ | ---------------------------- | -------------- | --------------------- |
| MD5 hashes | [md5](resources/product-search/home_and_kitchen/md5) | [md5](resources/product-search/clothing_shoes_and_jewelry/md5) | [md5](resources/product-search/pet_supplies/md5) | [md5](resources/product-search/sports_and_outdoors/md5) |
| Product lists | [product_list](resources/product-search/home_and_kitchen/product_list) | [product_list](resources/product-search/clothing_shoes_and_jewelry/product_list) | [product_list](resources/product-search/pet_supplies/product_list) | [product_list](resources/product-search/sports_and_outdoors/product_list) |
| Document-product associations | [assocs](resources/product-search/home_and_kitchen/assocs) | [assocs](resources/product-search/clothing_shoes_and_jewelry/assocs) | [assocs](resources/product-search/pet_supplies/assocs) | [assocs](resources/product-search/sports_and_outdoors/assocs) |
| Topics | [topics](resources/product-search/home_and_kitchen/topics) | [topics](resources/product-search/clothing_shoes_and_jewelry/topics) | [topics](resources/product-search/pet_supplies/topics) | [topics](resources/product-search/sports_and_outdoors/topics) |
| Relevance | [qrel_test](resources/product-search/home_and_kitchen/qrel_test) [qrel_validation](resources/product-search/home_and_kitchen/qrel_validation) | [qrel_test](resources/product-search/clothing_shoes_and_jewelry/qrel_test) [qrel_validation](resources/product-search/clothing_shoes_and_jewelry/qrel_validation) | [qrel_test](resources/product-search/pet_supplies/qrel_test) [qrel_validation](resources/product-search/pet_supplies/qrel_validation) | [qrel_test](resources/product-search/sports_and_outdoors/qrel_test) [qrel_validation](resources/product-search/sports_and_outdoors/qrel_validation) |

Usage
----

To replicate the experiments of the paper on **learning latent vector spaces for product search**, have a look the [product_search.sh](product-search.sh) script.

First, obtain the data described above. Download the 8 files (metadata + reviews) to your local disk without decompressing them.

Here's an overview:

* `meta_Clothing_Shoes_and_Jewelry.json.gz` (268M)
* `meta_Home_and_Kitchen.json.gz` (146M)
* `meta_Pet_Supplies.json.gz` (40M)
* `meta_Sports_and_Outdoors.json.gz` (175M)
* `reviews_Clothing_Shoes_and_Jewelry.json.gz` (847M)
* `reviews_Home_and_Kitchen.json.gz` (783M)
* `reviews_Pet_Supplies.json.gz` (234M)
* `reviews_Sports_and_Outdoors.json.gz` (594M)

Afterwards, we can construct models from 4-gram using 300-dimensional word representations and 128-dimensional entity representations as follows (the third argument is optional and defaults to `cpu`):

    [cvangysel@ilps SERT] ./product-search.sh \
        <path-to-directory-with-gzipped-amazon-data> \
        <path-to-nonexisting-temporary-directory> \
        [cpu|gpu]
    Processing clothing_shoes_and_jewelry.

    Creating output directory.

    Verifying corpus.

    Extracting product descriptions and reviews.

    Constructing LSE model on clothing_shoes_and_jewelry collection.

    NDCG@100 (validation): 0.1790
    NDCG@100 (test): 0.1479
    Processing sports_and_outdoors.

    Creating output directory.

    Verifying corpus.

    Extracting product descriptions and reviews.

    Constructing LSE model on sports_and_outdoors collection.

    NDCG@100 (validation): 0.1707
    NDCG@100 (test): 0.1761
    Processing home_and_kitchen.

    Creating output directory.

    Verifying corpus.

    Extracting product descriptions and reviews.

    Constructing LSE model on home_and_kitchen collection.

    NDCG@100 (validation): 0.2151
    NDCG@100 (test): 0.2423
    Processing pet_supplies.

    Creating output directory.

    Verifying corpus.

    Extracting product descriptions and reviews.

    Constructing LSE model on pet_supplies collection.

    NDCG@100 (validation): 0.2308
    NDCG@100 (test): 0.2578

    All done!


Citation
----

If you use SERT to produce results for your scientific publication, please refer to our [CIKM 2016](https://arxiv.org/abs/1608.07253) paper on product search and our [software overview](https://arxiv.org/abs/1706.03757) paper:

```
@inproceedings{VanGysel2016products,
  title={Learning Latent Vector Spaces for Product Search},
  author={Van Gysel, Christophe and de Rijke, Maarten and Kanoulas, Evangelos},
  booktitle={CIKM},
  volume={2016},
  pages={165--174},
  year={2016},
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