# Introduction to Machine Learning and Spark ML Pipelines

<div style="text-align:center">
  <img src="https://gitlab.com/droff/ph/raw/477d2a011575887dfb65d36dc3ff4c116f3bf586/logos/Spark-logo.png" width="192" height="100" style="margin-right:70px">
  <img src="https://gitlab.com/droff/ph/raw/477d2a011575887dfb65d36dc3ff4c116f3bf586/logos/spark-notebook-logo.png" width="111" height="128">
</div>

# Machine learning Pipeline

In this lab we are going to learn how to teach machine learning models, how to correctly set up an experiment, how to tune model hyperparameters and how to compare models. Also we'are going to get familiar with spark.ml package as soon as all of the work we'are going to get done using this package.

* http://spark.apache.org/docs/latest/ml-guide.html
* http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.package

## Evaluation Metrics
Model training and model quality assessment is performed on independent sets of examples. As a rule, the available examples are divided into two subsets: training (train) and control (test). The choice of the proportions of the split is a compromise. Indeed, the large size of the training leads to better quality of algorithms, but more noisy estimation of the model on the control. Conversely, the large size of the test sample leads to a less noisy assessment of the quality, however, models are less accurate.

Many classification models produce estimation of belonging to the class $\tilde{h}(x) \in R$ (for example, the probability of belonging to the class 1). They then make a decision about the class of the object by comparing the estimates with a certain threshold $\theta$:

$h(x) = +1$,  if $\tilde{h}(x) \geq \theta$, $h(x) = -1$, if $\tilde{h}(x) < \theta$

In this case, we can consider metrics that are able to work with estimates of belonging to a class.
In this lab, we will work with [AUC-ROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve) metric. Detailed understanding of the operating principle of AUC-ROC metric is not required to perform the lab.
## Model Hyperparameter Tuning
In machine learning problems it is necessary to distinguish the parameters of the model and hyperparameters (structural parameters). The model parameters are adjusted during the training (e.g., weights in the linear model or the structure of the decision tree), while hyperparameters are set in advance (for example, the regularization in linear model or maximum depth of the decision tree). Each model usually has many hyperparameters, and there is no universal set of hyperparameters optimal working in all tasks, for each task one should choose a different set of hyperparameters. _Grid search_ is commonly used to optimize model hyperparameters: for each parameter several values are selected and combination of parameter values where the model shows the best quality (in terms of the metric that is being optimized) is selected. However, in this case, it is necessary to correctly assess the constructed model, namely to do the split into training and test sample. There are several ways how it can be implemented:

 - Split the available samples into training and test samples. In this case, the comparison of a large number of models in the search of parameters leads to a situation when the best model on test data does not maintain its quality on new data. We can say that there is overfitting on the test data.
 - To eliminate the problem described above, it is possible to split data into 3 disjoint sub-samples: `train`, `validation` and `test`. The `validation` set is used for models comparison, and `test` set is used for the final quality assessment and comparison of families of models with selected parameters.
 - Another way to compare models is [cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics). There are different schemes of cross-validation:
  - Leave-one-out cross-validation
  - K-fold cross-validation
  - Repeated random sub-sampling validation
  
Cross-validation is computationally expensive operation, especially if you are doing a grid search with a very large number of combinations. So there are a number of compromises:
 - the grid can be made more sparse, touching fewer values for each parameter, however, we must not forget that in such case one can skip a good combination of parameters;
 - cross-validation can be done with a smaller number of partitions or folds, but in this case the quality assessment of cross-validation becomes more noisy and increases the risk to choose a suboptimal set of parameters due to the random nature of the split;
 - the parameters can be optimized sequentially (greedy) — one after another, and not to iterate over all combinations; this strategy does not always lead to the optimal set;
 - enumerate only small number of randomly selected combinations of values of hyperparameters.
 
 ## Data

We'are going to solve binary classification problem by building the algorithm which determines whether a person makes over 50K a year. Following variables are available:
* age
* workclass
* fnlwgt
* education
* education-num
* marital-status
* occupation
* relationship
* race
* sex
* capital-gain
* capital-loss
* hours-per-week

More on this data one can read in [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names)
