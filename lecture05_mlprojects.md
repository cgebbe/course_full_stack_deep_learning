# Lecture 5: ML Projects

[TOC]

## Introduction

- Based on [a report from TechRepublic](https://www.techrepublic.com/article/why-85-of-ai-projects-fail/), 85% of AI projects fail. Potential reasons:
  - technical infeasible
  - never make the leap to production
  - unclear success criteria
  - poor team management





![image-20211208091920228](week5_mlprojects.assets/image-20211208091920228.png)



## Lifecycle

![img](week5_mlprojects.assets/image4.png)

- Example: Estimate 6D pose of objects such that robot can pick them

- Relevant points (apart from the obvious)
  - Note the circular nature of the steps!
  - Training: First implement baseline in e.g. OpenCV, find SoTA models and reproduce
    - If task is too hard, revisit requirements (go back to planning)
  - Deployment: Pilot in lab, write tests to prevent regression, roll out in production
    - If doesn't work in pilot, you might need to
      - improve accuracy in training
      - collect more data
      - redefine the metrics (if good model metrics doesn't lead to success downstream)
- What else
  - understand what's possible
  - If you're an expert in the field: know what to try next



## Prioritizing projects & Archetypes

### What other companies are doing

- netflix, see image below
- papers from google, facebook, nvidia, netflix, etc.
- blog-posts from early-stage companies such as Uber, Lyft, spotify, Stripe



![image-20211208093144539](week5_mlprojects.assets/image-20211208093144539.png)

Here is a list of excellent ML use cases to check out (credit to Chip Huyen’s [ML Systems Design Lecture 2 Note](https://docs.google.com/document/d/15vCMf7SbDuxST9Q-rWtx8o7qHJQN2pE5urCDFTYI1zs/edit?usp=sharing)):

- [Human-Centric Machine Learning Infrastructure at Netflix](https://www.youtube.com/watch?v=XV5VGddmP24) (Ville Tuulos, InfoQ 2019)
- [2020 state of enterprise machine learning](https://algorithmia.com/state-of-ml) (Algorithmia, 2020)
- [Using Machine Learning to Predict Value of Homes On Airbnb](https://medium.com/airbnb-engineering/using-machine-learning-to-predict-value-of-homes-on-airbnb-9272d3d4739d) (Robert Chang, Airbnb Engineering & Data Science, 2017)
- [Using Machine Learning to Improve Streaming Quality at Netflix](https://medium.com/netflix-techblog/using-machine-learning-to-improve-streaming-quality-at-netflix-9651263ef09f) (Chaitanya Ekanadham, Netflix Technology Blog, 2018)
- [150 Successful Machine Learning Models: 6 Lessons Learned at Booking.com](https://blog.acolyer.org/2019/10/07/150-successful-machine-learning-models/) (Bernardi et al., KDD, 2019)
- [How we grew from 0 to 4 million women on our fashion app, with a vertical machine learning approach](https://medium.com/hackernoon/how-we-grew-from-0-to-4-million-women-on-our-fashion-app-with-a-vertical-machine-learning-approach-f8b7fc0a89d7) (Gabriel Aldamiz, HackerNoon, 2018)
- [Machine Learning-Powered Search Ranking of Airbnb Experiences](https://medium.com/airbnb-engineering/machine-learning-powered-search-ranking-of-airbnb-experiences-110b4b1a0789) (Mihajlo Grbovic, Airbnb Engineering & Data Science, 2019)
- [From shallow to deep learning in fraud](https://eng.lyft.com/from-shallow-to-deep-learning-in-fraud-9dafcbcef743) (Hao Yi Ong, Lyft Engineering, 2018)
- [Space, Time and Groceries](https://tech.instacart.com/space-time-and-groceries-a315925acf3a) (Jeremy Stanley, Tech at Instacart, 2017)
- [Creating a Modern OCR Pipeline Using Computer Vision and Deep Learning](https://dropbox.tech/machine-learning/creating-a-modern-ocr-pipeline-using-computer-vision-and-deep-learning) (Brad Neuberg, Dropbox Engineering, 2017)
- [Scaling Machine Learning at Uber with Michelangelo](https://eng.uber.com/scaling-michelangelo/) (Jeremy Hermann and Mike Del Balso, Uber Engineering, 2019)
- [Spotify’s Discover Weekly: How machine learning finds your new music](https://hackernoon.com/spotifys-discover-weekly-how-machine-learning-finds-your-new-music-19a41ab76efe) (Sophia Ciocca, 2017)



### Assessing feasibility of projects

![image-20211208093603202](week5_mlprojects.assets/image-20211208093603202.png)

- Accuracy:
  - ~1x budget for 90% accuracy
  - ~10x budget for 99% accuracy
  - ~100x budget for 99.9% accuracy
- How to run **feasibility study**
  - Define success criteria with all stakeholders
  - Do a literature review
  - Build a labeled **benchmark dataset** (this will give you a sense of data collection costs!)
  - Build a **minimal viable product** using e.g. manual rules (this will give you a sense about problem difficulty?)
  - Are you sure, you need ML after all?



### Typical products

![image-20211208094147620](week5_mlprojects.assets/image-20211208094147620.png)

- "Data flywheel" = positive feedback cycle
  - better model -> more users -> more data -> even better model



![img](week5_mlprojects.assets/image7.png)

- Potential to improve either impact / feasibility:
  - software 2.0: use data flywheel to improve 
  - human-in-the-loop: better product design (prediction as suggestion, ask customer for feedback, make transparent), see [guidelines for human-AI interaction](https://www.microsoft.com/en-us/research/project/guidelines-for-human-ai-interaction/)
  - autonomous systems: add humans in the loop / limit scope



# Metrics

- Problem
  - most real word settings care about multiple metrics, but in ML systems you can only optimize a single number
- Solutions
  - some kind of average 
  - more complex / domain-specific formula
  - **put a threshold on (n-1) metrics, optimize the n'ths metric** (usually employed in practice)
    - thresholds can be chose via either acceptable tolerance downstream or baselines

# Baselines

- Baselines are important, because they indicate next steps!
  - If training KPI close to baseline, address overfitting
  - If training KPI far away from baseline, address underfitting
- Strive for good ("tight") baselines, as they indicate usefulness of model versus theoretical optimum
- Baseline sources
  - External
    - business requirements
    - published results (caution with comparability)
  - Internal
    - scripted baselines (e.g. OpenCV)
    - simple ML baselines (e.g. linear regression)
    - human performance
      - exists in different qualities: from e.g. Amazon Turk or domain experts
      - consider tradeoff between quality - quantity (e.g. sufficient quality for en-mass data, expert quality for failure or edge cases)

# Conclusion

![image-20211208095844180](week5_mlprojects.assets/image-20211208095844180.png)