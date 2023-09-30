# Taxonomy-Structured Domain Adaptation (TSDA) (under construction)
This repo contains the code for our ICML 2023 paper:<br>
**Taxonomy-Structured Domain Adaptation**<br>
Tianyi Liu*, Zihao Xu*, Hao He, Guang-Yuan Hao, Guang-He Lee, Hao Wang<br>
*Fortieth International Conference on Machine Learning (ICML), 2023*<br>
[[Paper](https://arxiv.org/abs/2306.07874)] [[OpenReview](https://openreview.net/forum?id=ybl9lzdZw7)] [[PPT](https://shsjxzh.github.io/files/TSDA_5_minutes.pdf)] [[Talk (Youtube)](https://www.youtube.com/watch?app=desktop&v=hRWfAsi0Uks)] [[Talk (Bilibili)](https://www.bilibili.com/video/BV13g4y1A7Uq/?spm_id_from=333.999.list.card_archive.click&vd_source=38c48d8008e903abbc6aa45a5cc63d8f)]
*"*" indicates equal contribution.*

## Outline for This README
* [Brief Introduction for TSDA](#brief-introduction-for-tsda)
* [Method Overview](#method-overview)
* [Theorem (Informal)](#theorem-informal-see-formal-definition-in-the-paper)
* [Installation](#installation)
* [Code for Different Datasets](#code-for-different-datasets)
* [Quantitative Result](#quantitative-result)
* [Related Works](#also-check-our-relevant-work)
* [Reference](#reference)

## Brief Introduction for TSDA
For classical domain adaptation methods such as DANN, they enforce uniform alignment to boost the generalization ability of models. However, recent studies have shown that, such uniform alignment can harm domain adaptation performance. To deal with this problem, we incorporate domain taxonomy into domain adaptation process. With domain taxonomy, we can break the uniform alignment in domain adaptation. We build on the classic adversarial framework and introduce a novel taxonomist, which competes with the adversarial discriminator to preserve the taxonomy information. The equilibrium recovers the classic adversarial domain adaptation’s solution if given a non-informative domain taxonomy (e.g., a flat taxonomy where all leaf nodes connect to the root node) while yielding non-trivial results with other taxonomies.

<p align="center">
<img src="fig/example_taxonomy.png" alt="" data-canonical-src="fig/example_taxonomy.png" width="93%"/>
</p>
<p>
    <em>Figure 1. An example of using domain taxonomy to break the uniform alignment. For our model, the middle representation for basset and Beagle should be more similar than the one for basset and tabby.</em>
</p>

## Method Overview


## Theorem (Informal, See Formal Definition in the Paper)
* The introduction of the taxonomist prevents the discriminator from enforcing uniform alignment.
* TSDA can recover DANN with a non-informative taxonomy.
* DANN with weighted pairwise discriminators can only produce uniform alignment.