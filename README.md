# Taxonomy-Structured Domain Adaptation (TSDA) (under construction)
This repo contains the code for our ICML 2023 paper:<br>
**Taxonomy-Structured Domain Adaptation**<br>
Tianyi Liu*, Zihao Xu*, Hao He, Guang-Yuan Hao, Guang-He Lee, Hao Wang<br>
*Fortieth International Conference on Machine Learning (ICML), 2023*<br>
[[Paper](https://arxiv.org/abs/2306.07874)] [[OpenReview](https://openreview.net/forum?id=ybl9lzdZw7)] [[PPT](https://shsjxzh.github.io/files/TSDA_5_minutes.pdf)] [[Talk (Youtube)](https://www.youtube.com/watch?app=desktop&v=hRWfAsi0Uks)] [[Talk (Bilibili)](https://www.bilibili.com/video/BV13g4y1A7Uq/?spm_id_from=333.999.list.card_archive.click&vd_source=38c48d8008e903abbc6aa45a5cc63d8f)]

## Outline for This README
* [Brief Introduction for TSDA](#brief-introduction-for-tsda)
* [Method Overview](#method-overview)
* [Installation](#installation)
* [Code for Different Datasets](#code-for-different-datasets)
* [Quantitative Result](#quantitative-result)
* [Related Works](#also-check-our-relevant-work)
* [Reference](#reference)

## Brief Introduction for TSDA
Domain adaptation aims to mitigate distribution shifts among different domains. However, traditional formulations are mostly limited to categorical domains, greatly simplifying nuanced domain relationships in the real world. In this work, we tackle a generalization with taxonomystructured domains, which formalizes domains with nested, hierarchical similarity structures such as animal species and product catalogs. We build on the classic adversarial framework and introduce a novel taxonomist, which competes with the adversarial discriminator to preserve the taxonomy information. The equilibrium recovers the classic adversarial domain adaptation’s solution if given a non-informative domain taxonomy (e.g., a flat taxonomy where all leaf nodes connect to the root node) while yielding non-trivial results with other taxonomies.

<p align="center">
<img src="fig/example_taxonomy.png" alt="test" data-canonical-src="fig/example_taxonomy.png" width="93%"/>
</p>