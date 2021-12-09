

# Lecture 6: MLOps Infrastructure & Tooling

![img](week6_mlops_infrastructure.assets/Infra-Tooling2.png)

![img](week6_mlops_infrastructure.assets/Infra-Tooling3.png)

- Follow https://twitter.com/full_stack_dl for "tooling tuesdays", where they recommend one tool weekly
- This session focuses **only on Training/Evaluation**!

[TOC]

## Training / Evaluation

### Software Engineering

- VSCode - recommended 
  - peek documentation
  - for access to remote machine and docker containers
  - for integrated linting (mypy, pylint, ...)
  - port forwarding
- Jupyter notebook
  - see course from Jeremy Howard on course.fast.ai
  - Pro: Great as first draft 
  - Con: Hard to version, hard to test, hard to log, out-of-order execution, poor IDE
  - However, some companies use it (Netlflix (!), fast.ai, NBDev, )
- Streamlit for quickly generating apps
- Setting up 
  - conda for python, CUDA, cudnn based on `environment.yml`
  - pip-tools for actual python libraries (to lock compatible versions) based on `prod.in`, `dev.in`
  - Makefile for installing everything



### Compute

Recommendation concerning local vs. cloud, see also blog posts from Tim Dettmers

- Hobbyists: Build your own machine (maybe a 4x Turing or a 2x Ampere PC) during development. Either use the same PC or use cloud instances during training/evaluation.
- Startups: Buy a sizeable Lambda Labs machine for every ML scientist during development. Buy more shared server machines or use cloud instances during training/evaluation.
- Larger companies: Buy an even more powerful machine for every ML scientist during development. Use cloud with fast instances with proper provisioning and handling of failures during training/evaluation.

Cloud

- Amazon Web Services, Google Cloud Platform, and Microsoft Azure are the cloud heavyweights with largely similar functions and prices. There are also startups like [Lambda Labs](https://lambdalabs.com/service/gpu-cloud) and [Corewave](https://www.coreweave.com/pricing) that provide cloud GPUs. 



### Resource management

1. **Script a solution ourselves**: In theory, this is the simplest solution. We can check if a resource is free and then lock it if a particular user is using it or wants to.
2. **SLURM**: If we don't want to write the script entirely ourselves, standard cluster job schedulers like [SLURM](https://slurm.schedmd.com/documentation.html) can help us. The workflow is as follows: First, a script defines a job’s requirements. Then, the SLURM queue runner analyzes this and then executes the jobs on the correct resource.
3. **Docker/Kubernetes**: The above approach might still be too manual for your needs, in which case you can turn to Docker/Kubernetes. [Docker](https://www.docker.com/) packages the dependency stack into a lighter-than-VM package called a container (that excludes the OS). [Kubernetes](https://kubernetes.io/) lets us run these Docker containers on a cluster. In particular, [Kubeflow](https://www.kubeflow.org/) is an OSS project started by Google that allows you to spawn/manage Jupyter notebooks and manage multi-step workflows. It also has lots of plug-ins for extra processes like hyperparameter tuning and model deployment. However, Kubeflow can be a challenge to setup.
4. **Custom ML software**: There’s a lot of novel work and all-in-one solutions being developed to provision compute resources for ML development efficiently. Platforms like [AWS Sagemaker](https://aws.amazon.com/sagemaker/), [Paperspace Gradient](https://gradient.paperspace.com/), and [Determined AI](https://determined.ai/) are advancing (see below!). Newer startups like [Anyscale](https://www.anyscale.com/) and [Grid.AI](https://www.grid.ai/) (creators of PyTorch Lightning) are also tackling this. Their vision is around allowing you to seamlessly go from training models on your computer to running lots of training jobs in the cloud with a simple set of SDK commands.



### Frameworks and distributed training

![img](week6_mlops_infrastructure.assets/Infra-Tooling6.png)

- Additional frameworks
  - HuggingFace abstracts entire model architectures (not just layers), mostly in the NLP realm
- Distributed training
  - data parallelism - easy
  - model parallelism - avoid if possible by using GPU with more RAM or gradient checkpointing (=save some gradients to disk)

### Experiment management

- [TensorBoard](https://www.tensorflow.org/tensorboard): This is the default experiment tracking platform that comes with TensorFlow. As a pro, it’s easy to get started with. On the flip side, it’s not very good for tracking and comparing multiple experiments. It’s also not the best solution to store past work.
- [MLFlow](https://mlflow.org/): An OSS project from Databricks, MLFlow is a complete platform for the ML lifecycle. They have great experiment and model run management at the core of their platform. Another open-source project, [Keepsake](https://keepsake.ai/), recently came out focused solely on experiment tracking.
- Paid platforms ([Comet.ml](https://www.comet.ml/), [Weights and Biases](https://wandb.ai/), [Neptune](https://neptune.ai/)): Finally, outside vendors offer deep, thorough experiment management platforms, with tools like code diffs, report writing, data visualization, and model registering features. In our labs, we will use Weights and Biases.



### Hyperparameter tuning

- [SigOpt](https://sigopt.com/) offers an API focused exclusively on efficient, iterative hyperparameter optimization. Specify a range of values, get SigOpt’s recommended hyperparameter settings, run the model and return the results to SigOpt, and repeat the process until you’ve found the best parameters for your model.
- Rather than an API, [Ray Tune](https://docs.ray.io/en/master/tune/index.html) offers a local software (part of the broader Ray ecosystem) that integrates hyperparameter optimization with compute resource allocation. Jobs are scheduled with specific hyperparameters according to state-of-the-art methods, and underperforming jobs are automatically killed.
- Weights and Biases also has this feature! With a YAML file specification, we can specify a hyperparameter optimization job and perform a “[sweep](https://wandb.ai/site/sweeps),” during which W&B sends parameter settings to individual “agents” (our machines) and compares performance.

Note: no mentioning of hyperopt or optuna?



## All-in-one solutions


![img](week6_mlops_infrastructure.assets/Infra-Tooling7.png)

- Domino data lab rather for traditional data science, not deeplearning. Also rather expensive
- Amazon Sagemaker is ~20-40% more expensive than renting a compute like EC2 directly
- GC ML might be very interesting if you're interested in TPUs



