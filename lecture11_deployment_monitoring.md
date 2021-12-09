



# Lecture 11: Deployment and monitoring

[TOC]

## Model Deployment

### Types of Deployment

#### Batch Prediction

- Run model e.g. nightly on full database
- This works if the universe of inputs is small (e.g. 1 prediction per user, think recommendations)

![img](lecture11_deployment_monitoring.assets/image16.png)

PROS

- It is simple to implement.
- It requires relatively low latency to the user.

CONS

- It does not scale to complex input types.
- Users do not get the most up-to-date predictions.
- Models frequently become “stale” and hard to detect.

#### Model-In-Service (rarely used)

Model-in-service means that you package up your model and include it in the deployed web server. Then, the web server loads the model and calls it to make predictions.

![img](lecture11_deployment_monitoring.assets/image11.png)


PROS

- It reuses your existing infrastructure.

CONS

- The web server may be written in a different language.
- Models may change more frequently than the server code.
- Large models can eat into the resources for your webserver.
- Server hardware is not optimized for your model (e.g., no GPUs).
- Model and server may scale differently.

#### Model-As-Service (most used today)

Model-as-service means that you deploy the model separately as its own service. The client and server can interact with the model by making requests to the model service and receiving responses.

![img](lecture11_deployment_monitoring.assets/image14.png)

PROS

- It is dependable, as model bugs are less likely to crash the web app.
- It is scalable, as you can choose the optimal hardware for the model and scale it appropriately.
- It is flexible, as you can easily reuse the model across multiple applications.

CONS

- It adds latency.
- It adds infrastructural complexity.
- Most importantly, you are now on the hook to run a model service...

### Building A Model Service

#### REST APIs

- Serve prediction in response formatted HTTP requests
  - There's not yet a standard REST API requests / response
- Alternatives are GRPC (used in tensorflow servin) and GraphQL (not relevant for to model services)

#### Dependency Management
- Constrain dependencies!
  - use ONNX as a standard format to reduce dependencies. Can work great, but ML libraries develop more quickly than ONNX
- OR: Use containers, e.g. docker in combination with a container orchestration framework like kubernetes

#### Performance Optimization
- Run on GPU or CPU?
  - PRO GPU: Same hardware as in development, higher throughput
  - CON GPU: More complex to set up, expensive
- Concurrency
  - Instead of running a single model copy on your machine, you run multiple model copies on different CPUs or cores. In practice, you need to be careful about **thread tuning** - making sure that each model copy only uses the minimum number of threads required. Read [this blog post from Roblox](https://blog.roblox.com/2020/05/scaled-bert-serve-1-billion-daily-requests-cpus/) for the details.
- Model distillation - do not do yourself!
  - Model distillation is a compression technique in which a **small “student” model** is trained to reproduce the behavior of a large “teacher” model. The method was first proposed by [Bucila et al., 2006](https://www.cs.cornell.edu/~caruana/compression.kdd06.pdf) and generalized by [Hinton et al., 2015](https://arxiv.org/pdf/1503.02531.pdf).
  - Distillation can be finicky to do yourself, so it is infrequently used in practice. An exception are pretrained distilled model such as "DistilBERT".
- Model quantization
  - A straightforward method is implemented [in the TensorFlow Lite toolkit](https://www.tensorflow.org/lite/performance/quantization_spec). It turns a matrix of 32-bit floats into 8-bit integers by applying a simple “center-and-scale” transform to it
  - [PyTorch also has quantization built-in](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/) that includes three techniques: dynamic quantization, post-training static quantization, and quantization-aware training.
- Caching
  - cache the model’s frequently-used inputs
- Batching - do not do yourself!
  - ML models achieve higher throughput when making predictions in parallel (especially true for GPU inference). The last caveat is that you probably do not want to implement batching yourself.
- Sharing The GPU - do not do yourself!
  - Your model may not take up all of the GPU memory with your inference batch size. **Why not run multiple models on the same GPU?** You probably want to use a model serving solution that supports this out of the box.
- Model Serving Libraries
  - Nearly necessary for GPU inference
  - There are canonical open-source model serving libraries for both PyTorch ([TorchServe](https://pytorch.org/serve/)) and TensorFlow ([TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)). [Ray Serve](https://docs.ray.io/en/master/serve/index.html) is another promising choice. Even NVIDIA has joined the game with [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server).

#### Horizontal Scaling

In practice, there are two common methods:

- Use a container orchestration toolkit like Kubernetes
  - Build/configure k8s yourself: k8s manages containers. k8s provides a single service for requests. k8s divides traffic.
  - Use existing frameworks building on top of k8s like [KFServing](https://www.kubeflow.org/docs/components/kfserving/) - part of the [Kubeflow](https://www.kubeflow.org/) package - or [Seldon](http://seldon.io/) 
- Use a serverless option like AWS Lambda, Google Cloud Functions, Azure Functions
  - PROS
    - automatic load balancing
    - only pay for compute time!
    - can be triggered from ~200 services
  - Cons
    - limited size of deployed package (usually no issue, unless e.g. GPT3)
    - CPU-only
    - limited execution time
    - challenging to build pipelines
    - no cache (no state management)

#### Model Deployment

If serving is how you turn a model into something that can respond to requests, **deployment** is how you roll out, manage, and update these services. You probably want to be able to **roll out gradually**, **roll back instantly**, and **deploy pipelines of models**

Here, no specific ways are discussed. Your deployment libraries (like k8s based ones or AWS lambda) hopefully take care of this for you

#### Managed Options (effectively an all-in-one deployment solution)

There are managed options in the market: All major cloud providers have ones (Google AI, Amazon Sagemaker) that enable you to package your model in a predefined way and turn it into an API. Startups like [Algorithmia](https://algorithmia.com/) and [Cortex](https://www.cortex.dev/) are some alternatives

#### Takeaways

- If you are making CPU inference, you can get away with scaling by launching more servers or going serverless
- If you are using GPU inference, serving tools will save you time. It’s worth keeping an eye on startups in this space.

### Edge Deployment

![img](lecture11_deployment_monitoring.assets/image9.png)


The pros of edge prediction:

- It has low latency.
- It does not require an Internet connection.
- It satisfies data security requirements, as data does not need to leave the user’s device.

The cons of edge prediction:

- The client often has limited hardware resources available.
- Embedded and mobile frameworks are less full-featured than TensorFlow and PyTorch.
- It is challenging to update models.
- It is difficult to monitor and debug when things go wrong.

#### Tools For Edge Deployment
- TensorRT to optimize models on smaller NVIDIA GPUs (think robots)
- Apache TVM to compile them for multiple backends
- TFLite or PyTorch mobile for mobile devices
- Tensorflow.js to compile to javascript (can even be used for training)

How to ML models on mobile devices

- CoreML for Apps (only inference)
- MLKit for google
- Fritz for both ?!

#### More Efficient Models
- e.g. MobileNet, DistilBERT 

#### Mindset For Edge Deployment
-  you can make up a factor of 2-10 through distillation, quantization, and other tricks (but not more than that)
- treat tuning the model for your device as an additional risk in the deployment cycle

#### Takeaways
- Web deployment is easier, so only perform edge deployment if you need to.
- You should choose your framework to match the available hardware and corresponding mobile frameworks. Else, you can try Apache TVM to be more flexible.
- You should start considering hardware constraints at the beginning of the project and choose the architectures accordingly.


## Model Monitoring

### Why Model Degrades Post-Deployment?

Many things can go wrong with a model once it’s been trained. There are three core ways that the model’s performance can degrade:

![img](lecture11_deployment_monitoring.assets/image17.png)

### Data Drift
There are a few, [very real](https://www.technologyreview.com/2020/05/11/1001563/covid-pandemic-broken-ai-machine-learning-amazon-retail-fraud-humans-in-the-loop/), different types of data drift:

- **Instantaneous drift**: In this situation, the paradigm of the draft dramatically shifts. Examples are deploying the model in a new domain (e.g., self-driving car model in a new city), a bug in the preprocessing pipeline, or even major external shifts like COVID.
- **Gradual drift**: In this situation, the value of data gradually changes with time. For example, users’ preferences may change over time, or new concepts can get introduced to the domain.
- **Periodic drift**: Data can have fluctuating value due to underlying patterns like seasonality or time zones.
- **Temporary drift**: The most difficult to detect, drift can occur through a short-term change in the data that shifts back to normal. This could be via a short-lived malicious attack, or even simply because a user with different demographics or behaviors uses your product in a way that it’s not designed to be used.

###  What Should You Monitor?
In considering which metrics to focus on, prioritize ground-truth metrics (model and business metrics), then **approximate performance** metrics (business and input/outputs), and finally, system health metrics. The harder a metric may be to monitor, the more useful it likely is.

- The hardest and best metrics to monitor are **model performance metrics**, though these can be difficult to acquire in real-time (labels are hard to come by).
- **Business metrics** can be helpful signals of model degradation in monitoring but can easily be confounded by other impactful considerations.
- **Model inputs and predictions** are a simple way to identify high-level drift and are very easy to gather. Still, they can be difficult to assess in terms of actual performance impact, leaving it more of an art than science.
- Finally, **system performance** (e.g., GPU usage) can be a coarse method of catching serious bugs.



###  How Do You Measure Distribution Changes?
- Select a reference window
  - most practical thing to do is to use your training or evaluation data as the reference
  - Alternative is e.g. time window from last week
- Select a measurement window
  - pick one or several window sizes and slide them over the data
  - in general very problem-dependent
- Compute distance metric
  - for 1D 
    - rule-based
      - recommended! Works well, catches most bugs, easy to understand
    - statistical distance metrics
      - [KL divergence](https://machinelearningmastery.com/divergence-between-probability-distributions/) - sounds good, but not recommended since very sensitive to distribution tails and difficult to interpret
      - [KS Statistic](https://www.statisticshowto.com/kolmogorov-smirnov-test/) - recommended! `max_x |p_cum(x)-q_cum(x)|` 
        - ![image-20211209010733214](lecture11_deployment_monitoring.assets/image-20211209010733214.png)
      - [D1 distance](https://mlsys.org/Conferences/2019/doc/2019/167.pdf) - google uses it! `sum_i|p_i-q_i|`
        - ![image-20211209010815657](lecture11_deployment_monitoring.assets/image-20211209010815657.png) 
  - for multiple dimensions
    - [Maximum mean discrepancy](http://alex.smola.org/teaching/iconip2006/iconip_3.pdf)
    - Performing multiple 1D comparisons across the data: While suffering from [the multiple hypothesis testing problem](https://en.wikipedia.org/wiki/Bonferroni_correction), this is a practical approach.
    - Prioritize some features for 1D comparisons: Another option is to avoid testing all the features and only focus on those that merit comparison; for example, those features you know may have shifted in the data.
    - **[Projections](https://arxiv.org/abs/1810.11953):** In this approach, large data points are put through a dimensionality reduction process and then subject to a two-sample statistical test. Reducing the dimensionality with a domain-specific approach (e.g., mean

###  How Do You Tell If A Change Is Bad?
There’s no hard and fast rule for finding if a change in the data is bad. An easy option is to set thresholds on the test values. Don’t use a statistical test like the KS test, as they are too sensitive to small shifts. Other [options](https://blog.anomalo.com/dynamic-data-testing-f831435dba90?gi=6c18774717d2) include setting manual ranges, comparing values over time, or even applying an unsupervised model to detect outliers. In practice, fixed rules and specified ranges of test values are used most in practice
### Tools For Monitoring
There are three categories of tools useful for monitoring:

1. **System monitoring tools** like [AWS CloudWatch](https://aws.amazon.com/cloudwatch/), [Datadog](https://www.datadoghq.com/), [New Relic](https://newrelic.com/), and [honeycomb](https://www.honeycomb.io/) test traditional performance metrics
2. **Data quality tools** like [Great Expectations](https://greatexpectations.io/), [Anomalo](https://www.anomalo.com/), and [Monte Carlo](https://www.montecarlodata.com/) test if specific windows of data violate rules or assumptions.
3. **ML monitoring tools** like [Arize](https://arize.com/), [Fiddler](https://www.fiddler.ai/), and [Arthur](https://www.arthur.ai/) can also be useful, as they specifically test models.

###  Evaluation Store
**Monitoring is more central to ML than for traditional software**.

- In traditional SWE, most bugs cause loud failures, and the data that is monitored is most valuable to detect and diagnose problems. If the system is working well, the data from these metrics and monitoring systems may not be useful. The monitoring system is usually appended at the end.
- In machine learning, however, monitoring plays a different role. First off, bugs in ML systems often lead to silent degradations in performance. Furthermore, the data that is monitored in ML is literally the code used to train the next iteration of models. Because monitoring is so essential to ML systems, tightly integrating the it into the ML system architecture brings major benefits  (think "data flywheel" again). When used as such, we call it **evaluation store** in this lesson:

![img](lecture11_deployment_monitoring.assets/image8.png)

- from Train
  - register data distribution and model performance
  - warn if training data looks different to prod
- from Test
  - Register (final) model performance on all test dataset slices (maybe record those as well?)
- from Deploy
  - Run a shadow test or A/B test and compare model performances
  - Log data and approximate performance metrics
    - Fire an alert when approximate performance dips below threshold -> Trigger retraining?
    - Flag uncertain predictions / special data -> Later used to select data subset for additional training (since often cannot label all)

### Conclusion

- Something will always go wrong, and you should have a system to catch errors.
- Start by looking at data quality metrics and system metrics, as they are easiest.
- In a perfect world, the testing and monitoring should be linked, and they should help you close the data flywheel.