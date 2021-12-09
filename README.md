

# About

These are my notes (and maybe later labs) while taking selected lessons of the [spring 2021 full stack deep learning online course](https://fullstackdeeplearning.com/spring2021/).


## Overview

The lifecycle of ML projects is discussed in lecture 5 (see image below).

![img](readme.assets/image4.png)

- per-project activities
  - Planning & project setup is discussed in lecture 5. It focuses on how to prioritize projects, namely based on impact and feasibility (including how to perform a quick feasibility study). Moreover, it emphasizes the importance of baselines.
  - Data collection & labeling is briefly discussed in lecture 8. However, it focuses more on tools than on methods (although there are some recommendations concerning whom to hire for labeling).
  - Training  & debugging is discussed in lecture 7. It recommends a systematic step-by-step guide to train models (which boils down to: start simple, make sure to overfit first, address underfitting and distribution shift later)
  - Deploying is discussed in lecture 11. It discusses some methods and best-practices as well as some tools. Moreover, it outlines how you should monitor your model after deployment (and advocates to empower your monitoring to become a [data flywheel](https://www.modyo.com/blog/data-flywheel-scaling-a-world-class-data-strategy))
  - Testing is discussed in lecture 10. It advocates to not only unit test your model, but the whole ML system (including e.g. data preprocessing, training system, online serving system, etc.) via several integration tests. (Lecture 10 also has a short digression about explainable and interpretable AI.)
- cross-project infrastructure
  - Team & hiring (including company organization) is discussed in lecture 13
  - Tooling is discussed in detail in lectures 6,8,11, see below




### Tooling

![img](readme.assets/Infra-Tooling3.png)

- A lot of infrastructure / tooling is required for ML models in production
  - **Training/Evaluation** tools are discussed in lecture 6
  - **Data** tools are discussed in lecture 8
    - including the feature store found in the deployment tab
  - **Deployment** tools are discussed in lecture 11
    - including some methods for performance optimization
    - including a more general view of **how to monitor** (not mainly relevant tools)





