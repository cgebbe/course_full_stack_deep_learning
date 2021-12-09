



# Lecture 7: Troubleshooting Deep Neural Networks

[TOC]

## Why is troubleshooting hard?

Many different things can cause this:

- It can be **implementation bugs**. Most bugs in deep learning are actually invisible.
- **Hyper-parameter choices** can also cause your performance to degrade. Deep learning models are very sensitive to hyper-parameters. Even very subtle choices of learning rate and weight initialization can make a big difference.
- Performance can also be worse just because of **data/model fit**. For example, you pre-train your model on ImageNet data and fit it on self-driving car images, which are harder to learn.
- Finally, poor model performance could be caused not by your model but your **dataset construction**. Typical issues here include not having enough examples, dealing with noisy labels and imbalanced classes, splitting train and test set with different distributions.

Therefore, use a **STRATEGY**!

- **start simple:** use simplest model and data
- **implement & debug:** once model runs
- **evaluate**: apply bias-variance decomposition to decide next steps+
- **tune hyperparameters**: coarse-to-fine search
- **improve model/data**: If underfit, make model bigger. If overfit, add data or regularize

![img](lecture7_troubleshooting_dnn.assets/image4.png)

## Start simple

### Choose a simple architecture

![image-20211208130725703](lecture7_troubleshooting_dnn.assets/image-20211208130725703.png)

For multiple input modalities, a simple architecture is:

![img](lecture7_troubleshooting_dnn.assets/image7.png)

### Use sensible defaults

we recommend:

- [Adam optimizer with a “magic” learning rate value of 3e-4](https://twitter.com/karpathy/status/801621764144971776?lang=en).
- [ReLU](https://stats.stackexchange.com/questions/226923/why-do-we-use-relu-in-neural-networks-and-how-do-we-use-it) activation for fully-connected and convolutional models and [Tanh](https://stats.stackexchange.com/questions/330559/why-is-tanh-almost-always-better-than-sigmoid-as-an-activation-function) activation for LSTM models.
- [He initialization for ReLU activation function and Glorot initialization for Tanh activation function](https://datascience.stackexchange.com/questions/13061/when-to-use-he-or-glorot-normal-initialization-over-uniform-init-and-what-are).
- No regularization
- No data normalization (?)

### Normalize inputs

- subtract mean, divide by variance
- for images, scale to [0,1] or [-0.5,0.5]
  - WARNING: some pipelines do this automatically, yielding to multiplying with 1/255 twice and low gradients!

### Simplify the problem

you should consider:

- Working with a small training set around 10,000 examples.
- Using a fixed number of objects, classes, input size, etc.
- Creating a simpler synthetic training set like in research labs.

## Implement and debug

general advice for implementing your model:

- **Start with a lightweight implementation**. You want minimum possible new lines of code for the 1st version of your model. The rule of thumb is **less than 200 lines**. This doesn’t count tested infrastructure components or TensorFlow/PyTorch code.
- **Use off-the-shelf components** such as Keras if possible, since most of the stuff in Keras works well out-of-the-box. If you have to use TensorFlow, use the built-in functions, don’t do the math yourself. This would help you avoid a lot of numerical instability issues.
- **Build complicated data pipelines later**. These are important for large-scale ML systems, but you should not start with them because data pipelines themselves can be a big source of bugs. Just start with a dataset that you can load into memory.

### Getting your model to run

Common issues and causes

- **Incorrect shapes for the network tensors**: This bug is a common one and can fail silently. This happens many times because the automatic differentiation systems in the deep learning framework do silent broadcasting. Tensors become different shapes in the network and can cause a lot of problems. 
  - To address this type of problem, you should step through your model creation and inference step-by-step in a debugger, checking for correct shapes and data types of your tensors.
- **Pre-processing inputs incorrectly**: For example, you forget to normalize your inputs or apply too much input pre-processing (over-normalization and excessive data augmentation).
- **Incorrect input to the model’s loss function**: For example, you use softmax outputs to a loss that expects logits.
- **Forgot to set up train mode for the network correctly**: For example, toggling train/evaluation mode or controlling batch norm dependencies.
- **Numerical instability**: For example, you get `inf` or `NaN` as outputs. This bug often stems from using an exponent, a log, or a division operation somewhere in the code.

![img](https://fullstackdeeplearning.com/spring2021/lecture-7-notes-media/image11.png)

- debugging tools
  - for tensorflow, you could use `import ipdb; ipdb.set_trace()` during either graph creation or `session.run()`.  Or you can use `tfdb` 

### Overfit a single batch

Common bugs:

![img](https://fullstackdeeplearning.com/spring2021/lecture-7-notes-media/image14.png)

### Compare to known result

Some known results are more useful than others (see graphic below). 

Notes

- **Simple baselines** are usually underrated! They can show you whether the model learns anything at all or is at least better than a simple heuristic!
- Implementations are useful, because you can step through code line-by-line and compare with your version

![img](https://fullstackdeeplearning.com/spring2021/lecture-7-notes-media/image10.png)




## Evaluate

- `Test error = irreducible error + bias + variance + distribution shift + val overfitting`
- Use not just split from training set, but also split from test set (because often originate from different distributions!)

![img](lecture7_troubleshooting_dnn.assets/image12.png)

## Improve Model and data

### Address underfitting

- Reminder: You shouldn't have used regularization in your simple model
- Choosing a different model architecture could come earlier if you don't need to implement it yourself
- Error analysis = Manual analysis of errors

![img](https://fullstackdeeplearning.com/spring2021/lecture-7-notes-media/image13.png)

Exemplary result after

- adding more layers to ConvNet
- switching to a Resnet-101
- tuning the learning rate

![image-20211208132923702](lecture7_troubleshooting_dnn.assets/image-20211208132923702.png)



### Address overfitting

- normalization (batch norm) can also be helpful for underfitting
- early-stopping is recommended by e.g. fast.ai?!

![img](https://fullstackdeeplearning.com/spring2021/lecture-7-notes-media/image15.png)

Exemplary result after

- adding more data
- adding weight decay
- adding data augmentation
- tuning several hyperparameters

![image-20211208133148313](lecture7_troubleshooting_dnn.assets/image-20211208133148313.png)

### Address distribution shift
- **Manual** analyzing errors

![img](https://fullstackdeeplearning.com/spring2021/lecture-7-notes-media/image9.png)

Exemplary error analysis

![img](lecture7_troubleshooting_dnn.assets/image5.png)

Domain adaptation are techniques to train on a "source" distribution and generalize to another "target" distribution using only unlabeled data or limited labeled data. There are a few different types of domain adaptation:

1. **Supervised domain adaptation** (works well in production): In this case, we have limited data from the target domain to adapt to. Some example applications of the concept include **fine-tuning** a pre-trained model or adding target data to a training set.
2. **Unsupervised domain adaptation** (more researchy): In this case, we have lots of unlabeled data from the target domain. Some techniques you might see are CORAL, domain confusion, and CycleGAN.

### Re-balance datasets

If the test-validation set performance starts to look considerably better than the test performance, you may have overfit the validation set. This commonly occurs with small validation sets or lots of hyperparameter training. If this occurs, resample the validation set from the test distribution and get a fresh estimate of the performance.


## Tune hyperparameters

Some hyperparameters tend to have a bigger effect *relative to defaults*:

![img](lecture7_troubleshooting_dnn.assets/image2.png)

Various methods of actually tuning them

- Manual Hyperparameter Optimization-
- Grid Search
- Random Search
- Coarse-to-fine (random) search
- Bayesian Hyperparameter Optimization

You should probably **start with coarse-to-fine random searches** and move to Bayesian methods as your codebase matures and you’re more certain of your model. Not sure whether this is still true in 2021?!