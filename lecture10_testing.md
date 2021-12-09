



# Lecture 10: Testing

What does it mean when we have a **good test set performance**?

> If the test data and production data come from the same distribution, then **in expectation**, the performance of your model on your evaluation metrics will be the same.

This means a lot of things can still go wrong:

- **In the real world, the production distribution does not always match the offline distribution**. You could have data drift, data shift, or even malicious users trying to attack your model.
- **Expected performance does not tell the whole story**. For instance, if you are working on long-tail data distribution, then the sample of data that you use to evaluate the model offline might not tell you much about the tail of that distribution - meaning that your test set score can be misleading. On top of that, if you evaluate your model with a single metric across your entire dataset, that does not mean your model is actually performing well against all the slices of data that might be important.
- **The performance of your model is not equal to the performance of your machine learning system**. There are other things (that can go wrong with the ML system) that do not have anything to do with the model.
- Finally, the test set performance only tells you about the metrics that you are evaluating**. In the real world, you are probably not optimizing the exact metrics you care about deep down.**

Therefore, we introduce concepts and methods in this lecture to:

1. Understand at a deeper level how well your model is performing.
2. Become more confident in your model’s ability to perform well in production.
3. Understand the model’s **performance envelope** (where you should expect it to perform well and where not).

[TOC]

## Software testing

### Types of Tests

Distribute tests 70-20-10 (number of tests / effort for tests) in unit / integration / e2e tests

- **Unit tests** that test the functionality of a single piece of code (an assertion on a single function or a single class) in isolation.
  - These can be solitary (=really test only one class/function using mocks) or sociable (=depend on correct behavior of other parts)
- **Integration tests** that test how two or more units perform when used together (e.g., test if a model works well with a pre-processing function).
- **End-to-end tests** that test how the entire software system performs when all units are put together (e.g., test on realistic inputs from a real user).

### Best Practices

- Automate your tests  (typically via a CI/CD system)
  - Start with SaaS providers such as github actions, gitlab CI, CircleCI, Travis or similar.
  - If those features do not suffice (e.g. no GPU access), setup your own CI/CD, e.g. Jenkins or Buildkite...
- Make sure your tests are reliable, run fast, and go through the same code review process as the rest of your code
- Enforce that tests must pass before merging into the main branch
- When you find new production bugs, convert them to tests
- Test coverage - often overrated
- TDD - not many people stick to this method religiously, but might be a valuable tool

### Testing In Production

> Bugs are inevitable, so you might as well set up the system so that users can help you find them

> You're basically letting your monitoring system play the role that a regression suite & CI play on other teams

This does not mean to not test before production! But for complex system bugs might really be inevitable and you can mitigate the effect of bugs using...

- **Canary deployment**: roll out new version to a small percentage of your users and separately monitor that group’s behavior.
- **A/B testing:** You can run a more principled statistical test if you have particular metrics that you care about: one for the old version of the code that is currently running and another for the new version that you are trying to test.
- **Real user monitoring:** Rather than looking at aggregate metrics (i.e., click-through rate), try to follow the journey that an actual user takes through your application and build a sense of how users experience the changes.
- **Exploratory testing:** Testing in production is not something that you want to automate fully. It should involve a bit of exploration (individual users or granular metrics).



## Testing Machine Learning System

![image-20211208205415656](lecture10_testing.assets/image-20211208205415656.png)

Due to such differences, here are **common mistakes** that teams make while testing ML systems:

- Think the ML system is just a model and only test that model.
- Not test the data.
- Not build a granular enough understanding of the performance of the model before deploying it.
- Not measure the relationship between model performance metrics and business metrics.
- Rely too much on automated testing.
- Think offline testing is enough, and therefore, not monitor or test in production.

![img](lecture10_testing.assets/image17.png)

### Infrastructure Tests (training system)

- Goal
  - Avoid bugs in training pipeline
- How
  - Test training a single batch or a single epoch
  - Unit tests part of training pipeline

### Training Tests (storage & preprocessing + training system)

- Goal
  - Ensure reproducibility of training
- How
  - Pull fixed dataset, run an abbreviated training (~4-6h nightly), check performance
  - Instead of fixed dataset, maybe also data from last week?!

### Functionality Tests (prediction system)

- Goal
  - Avoid bugs in prediction system, which is a simple wrapper around the model
- How
  - Load fixed model, run prediction, check results
  - Unit test prediction code

### Evaluation Tests (training + prediction system) !!!

- Goal
  - Ensure model is ready for production
- How
  - Evaluate model on all **metrics**, all **datasets (slices)**
  - Compare performance of new model with fixed or previous model and with baselines
  - Understand the performance envelope (where does it perform well, where not?) 
  - Run for every new candidate model

#### Metrics
Relevant metrics are

- **Model metrics**: precision, recall, accuracy, L2, etc.
- **Behavioral metrics**: The goal of behavioral tests is to ensure the model has the invariances we expect. There are three types of behavioral tests: (1) *invariance tests* to assert that the change in inputs shouldn’t affect outputs, (2) *directional tests* to assert that the change in inputs should affect outputs, and (3) *minimum functionality tests* to ensure that certain inputs and outputs should always produce a given result. Behavioral testing metrics are primarily used in NLP applications and proposed in the [Beyond Accuracy paper by Ribeiro et al. (2020)](https://arxiv.org/abs/2005.04118).
- **Robustness metrics**: The goal of robustness tests is to understand the model’s performance envelope (i.e., where you should expect the model to fail). You can examine feature importance, sensitivity to staleness (=train on data from different times. When is the model too outdated?), sensitivity to data drift (=ideally we would measure the sensitivity to certain data drifts and then measure the data drift in production - in real not yet employed), and correlation between model performance and business metrics (=if model metric drops, will business metrics drop?). In general, robustness tests are still under-rated.
- **Privacy and fairness metrics**: The goal of privacy and fairness tests is to distinguish whether your model might be biased against specific classes. Helpful resources are Google’s [Fairness Indicators](https://ai.googleblog.com/2019/12/fairness-indicators-scalable.html) and [the Fairness Definitions Explained paper by Verma and Rubin (2018)](https://fairware.cs.umass.edu/papers/Verma.pdf).
- **Simulation metrics**: The goal of simulation tests is to understand how the model performance could affect the rest of the system. These are useful when your model affects the real world (for systems such as autonomous vehicles, robotics, recommendation systems, etc.). Simulation tests are hard to do well because they require a model of how the world works and a dataset of different scenarios.

#### Dataset slices

Your main validation or test set should mirror your production distribution as closely as possible as a matter of principle. However, instead of simply evaluating the aforementioned metrics on your entire dataset in aggregate, you should also evaluate these metrics **on multiple slices of data**. A slice is a mapping of your data to a specific category (e.g. only country Japan, only users between 10-20 years, etc.).

![img](lecture10_testing.assets/image7.png)

A natural question that arises is how to pick those slices. Tools like [What-If](https://pair-code.github.io/what-if-tool/) and [SliceFinder](https://arxiv.org/abs/1807.06068) help surface the slices where the model performance might be of particular interest. When should you add new evaluation datasets?

- When you collect datasets to specify specific edge cases.
- When you run your production model on multiple data modalities.
- When you augment your training set with data not found in production (synthetic data).

At a high level, you want to compare the new model to **the previous model** and **another fixed older model**. Tactically, you can (1) set thresholds on the differences between the new and the old models for most metrics, (2) set thresholds on the differences between data slices, and (3) set thresholds against the fixed older model to prevent slower performance “leaks.”

### Shadow Tests (prediction + serving system)

- Goal
  - Catch bugs in production system
  - Possible, because production might use another language -> different production pipeline, converted model (e.g. ONNX) !
- How
  - Run new model in production system **alongside** previous model. Save data and run the offline model on it. Compare.

### A/B Tests (serving system)

- Goal
  - Understand how model affects user behavior / business metrics
- How
  - Deploy model only on 1-10% of users / data. Unlike shadow tests, return predictions of new model to users. Compare metrics.

### Labeling Tests (labeling system)

- Goal
  - catch poor quality labels
- How
  - aggregate labels (maybe weighted by trust)
  - additional QC of labels (yourself or certified labelers or using pretrained model)

### Expectation Tests (storage &  preprocessing system)

- Goal
  - Catch (processed) data quality issues
- How
  - Define hard manual rules (maybe inferred by existing data) for inputs/outputs of each step in data processing pipeline
    - see library by [greatexpectations.io](https://greatexpectations.io)
  - run them whenever you run data processing pipelines

### Challenges and Recommendations Operationalizing ML Tests

#### Challenges

- The first challenge is often **organizational**. In contrast to software engineering teams for whom testing is table stakes, data science teams often struggle to implement testing and code review norms.
- The second challenge is **infrastructural.** Most CI/CD platforms don’t support GPUs, data integrations, or other required elements of testing ML systems effectively or efficiently.
- The third challenge is **tooling**, which has not yet been standardized for operations like comparing model performance and slicing datasets.
- Finally, **decision-making** for ML test performance is hard. What is “good enough” test performance is often highly contextual, which is a challenge that varies across ML systems and teams.

#### Recommendations

- Test each part of the ML system, not just the model. You build the machine that builds the model, not just the model!
- Test code, data, and model performance, not just code.
- Testing model performance is an art, not a science. There is a considerable amount of intuition that guides testing ML systems.
- Thus, the *fundamental* goal of testing model performance is to **build a granular understanding** of how well your model performs and where you don’t expect it to perform well. Using this intuition derived from testing, you can make better decisions about productionizing your model effectively.
- Build up to this gradually! You don’t need to do everything detailed in this lecture, and certainly not all at once. **Start with**:
  1. Infrastructure tests (unit test training pipeline)
  2. Evaluation tests (test new model on all metrics and all dataset (slices))
  3. Expectation tests (set thresholds on expected data processing input/output)

## Explainable and Interpretable AI

Definitions:

- **Domain predictability**: the degree to which it is possible to detect data outside the model’s domain of competence.
  - Note: Could find very little on this term?!
- **Interpretability**: the degree to which a human can consistently predict the model’s result ([Kim et al., 2016](https://beenkim.github.io/papers/KIM2016NIPS_MMD.pdf)).
- **Explainability**: the degree to which a human can understand the cause of a decision ([Miller, 2017](https://arxiv.org/abs/1706.07269)).

There are four ways to achieve this (see details below)

- Use an interpretable family of models.
- Distill the complex model to an interpretable one.
- Understand the contribution of features to the prediction.
- Understand the contribution of training data points to the prediction.

Conclusion up front: At present, **true explainability for deep learning models is not possible**:

- Current explanation methods are not faithful to the original model performance; it can be easy to cherry-pick specific examples that can overstate explainability.
- Furthermore, these methods tend to be unreliable and highly sensitive to the input.
- Finally, as described in the attention section, the full explanation is often not available to modern explainability methods.

Because of these reasons, explainability is not practically feasible for deep learning models (as of 2021). Read [Cynthia Rudin’s 2019 paper](https://arxiv.org/pdf/1811.10154.pdf) for more detail.

### Use An Interpretable Family of Models

**Simple models** like linear regression, logistic regression, generalized linear models, and decision trees. Because of the reasonably elementary math, these models are interpretable and explainable. However, they are not very powerful.

Another class of models that are interpretable is **attention models**. However, attention maps are not particularly explainable. They do not produce *complete* explanations for a model’s output, just a directional explanation. Furthermore, attention maps are not reliable explanations. Attention maps tell us only where a model is looking, not why it is looking there.

![img](lecture10_testing.assets/image3.png)

### Distill A Complex To An Interpretable One

Instead of restricting models to only interpretable families, we can fit a more complex model and interpret its decision using another **surrogate model** from an interpretable family. This technique is quite simple and fairly general to apply. In practice, however, two concerns manifest. If the surrogate itself performs well on the predictions, why not try to directly apply it rather than the more complex model? If it doesn’t perform well, how do we know that it genuinely represents the complex model’s behavior?

Another category of surrogate models is [local surrogate models](https://christophm.github.io/interpretable-ml-book/lime.html) **(LIME)**. Rather than apply the surrogate model in a global context on all the data, LIME models focus on a single point to generate an explanation for. This method is used widely, as it works for all data types (including images and text). However, defining the right perturbations and ensuring the stability of the explanations is challenging.

### Understand The Contribution of Features To The Prediction

- **Data visualization** is one such option, with plots like [partial dependence plots](https://christophm.github.io/interpretable-ml-book/pdp.html) and [individual conditional expectation](https://christophm.github.io/interpretable-ml-book/ice.html) plots
- A numerical method is **permutation feature importance**, which selects a feature, randomizes its order in the dataset, and sees how that affects performance. While this method is very easy and widely used, it doesn’t work for high-dimensional data or cases where there is feature interdependence.
- A more principled approach to explaining the contribution of individual features is **SHAP** ([Shapley Additive Explanations](https://christophm.github.io/interpretable-ml-book/shap.html)). At a high level, SHAP scores test how much changes in a single feature impact the output of a classifier when controlling for the values of the other features. This is a reliable method to apply, as it works on a variety of data and is mathematically principled. However, it can be tricky to implement and doesn’t provide explanations.
- **Gradient-based saliency maps** are a popular method for explanations and interpretations. This intuitive method selects an input, performs a forward pass, computes the gradient with respect to the pixels, and visualizes the gradients. Essentially, how much does a unit change in the value of the input’s pixels affect the prediction of the model? This is a straightforward and common method. Similar to the challenge with attention, the explanations may not be correct, and the overall method is fragile and sensitive to small changes.

### Understand The Contribution of Training Data Points To The Prediction

These techniques are not (yet) common in deep learning.

- **Prototypes and criticisms** are one such approach, though it is less applicable to deep learning. In this method, prototypes are clusters of data that explain much of the variance in the model. Criticisms are data points not explained by the prototypes.
- Another approach is to look specifically at “**influential instances**” or data points that cause major changes in the model’s predictions when removed from the data set.

![img](https://fullstackdeeplearning.com/spring2021/lecture-10-notes-media/image18.png)



### Do You Need "Explainability"?

A good question to ask yourself whether or not “explainable AI” is a real need for your applications. There are a couple of cases where this question can be useful:

1. **Regulators demand it.** In this case, there’s not much you can do besides produce some kind of explainable model. However, it can be helpful to **ask for clarification** on what explainability is judged as.
2. **Users demand it.** In some cases, users themselves may want trust or explainability in the system. Investigate the necessity for the explainability and trust to come directly from the model itself. **Can good product design inspire trust more effectively?** For example, allowing doctors to simply override models can reduce the immediate need for explainability. A big associated concern is how often users interact with the model. Infrequent interactions likely require explainable AI, as humans do not get a chance to build their feel for the system. More frequent interactions allow for the simpler objective of interpretability.
3. **Deployment demands it.** Sometimes, ML stakeholders may demand explainability as a component of ensuring confidence in ML system deployment. In this context, explainability is the wrong objective; **domain predictability is the real aim**. Rather than full-on explainability, interpretability can be helpful for deployment, especially visualizations for debugging.

### Caveats For Explainable and Interpretable AI

- If you genuinely need to explain your model’s predictions, use an interpretable model family (read more [here](https://arxiv.org/abs/1806.10574)).

- Don’t try to force-fit deep learning explainability methods; they produce cool results but are not reliable enough for production use cases.

- Specific interpretability methods like LIME and SHAP are instrumental in helping users reach interpretability thresholds faster.

- Finally, the visualization for interpretability can be pretty useful for debugging.
