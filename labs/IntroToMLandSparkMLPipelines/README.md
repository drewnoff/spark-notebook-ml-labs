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

Many classification models produce estimation of belonging to the class <img src="http://telegra.ph/file/8aacba8ccab5367659ee8.png" border="0" /> (for example, the probability of belonging to the class 1). They then make a decision about the class of the object by comparing the estimates with a certain threshold $\theta$:

 <img src="http://telegra.ph/file/dba22c4d6f6fd98795bc7.png" align="center" border="0" />


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

```scala
val spark = sparkSession

val df = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv("notebooks/spark-notebook-ml-labs/labs/IntroToMLandSparkMLPipelines/data/data.adult.csv")  
  
df.show(5)
```

```
+---+---------+------+------------+-------------+------------------+---------------+-------------+-----+------+------------+------------+--------------+----------+
|age|workclass|fnlwgt|   education|education-num|    marital-status|     occupation| relationship| race|   sex|capital-gain|capital-loss|hours-per-week|>50K,<=50K|
+---+---------+------+------------+-------------+------------------+---------------+-------------+-----+------+------------+------------+--------------+----------+
| 34|Local-gov|284843|     HS-grad|            9|     Never-married|Farming-fishing|Not-in-family|Black|  Male|         594|           0|            60|     <=50K|
| 40|  Private|190290|Some-college|           10|          Divorced|          Sales|Not-in-family|White|  Male|           0|           0|            40|     <=50K|
| 36|Local-gov|177858|   Bachelors|           13|Married-civ-spouse| Prof-specialty|    Own-child|White|  Male|           0|           0|            40|     <=50K|
| 22|  Private|184756|Some-college|           10|     Never-married|          Sales|    Own-child|White|Female|           0|           0|            30|     <=50K|
| 47|  Private|149700|   Bachelors|           13|Married-civ-spouse|   Tech-support|      Husband|White|  Male|       15024|           0|            40|      >50K|
+---+---------+------+------------+-------------+------------------+---------------+-------------+-----+------+------------+------------+--------------+----------+
only showing top 5 rows
```

Sometimes there are missing values in the data. Sometimes, in the description of the dataset one can found the description of format of missing values. Particularly in the given dataset  missing values are identified by '?' sign.

**Problem** Find all the variables with missing values. Remove from the dataset all objects with missing values in any variable.

```scala
val missingValsFeatures = df.columns.filter(column => df.filter(df(column) === "?").count > 0)

println("Features with missing values: " + missingValsFeatures.mkString(", "))

val data = missingValsFeatures.foldLeft(df)((dfstage, column) => dfstage.filter(!dfstage(column).equalTo("?")))
```

Split on training and test datasets.

```scala
val Array(training, test) = data.randomSplit(Array(0.8, 0.2), seed = 1234)
```

### MLlib Transformers and Estimators

`Transformer` transforms one `DataFrame` into another `DataFrame`.

<div style="text-align:left">
  <img src="https://gitlab.com/droff/ph/raw/master/images/Transformer.png" width="566" height="352">
</div>

`Estimator` fits on a `DataFrame` to produce a `Transformer`.

<div style="text-align:left">
  <img src="https://gitlab.com/droff/ph/raw/master/images/Estimator.png" width="681" height="327">
</div>

## Training classifiers on numeric features

Some preprocessing steps are usually required after loading and cleaning dataset. In this case, these steps will include the following:

 - At first we will work only with numeric features. So let's select them separately in the feature vector "numFeatures" using [VectorAssembler](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.feature.VectorAssembler).
 - Select the target variable (the one we want to predict, string column of labels) and map it to an ML column of label indices using [StringIndexer](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.feature.StringIndexer), give the name "labelIndex" to a new variable.
 
```scala
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer

val assembler = new VectorAssembler()
  .setInputCols(Array("age",
                      "fnlwgt", 
                      "education-num", 
                      "capital-gain", 
                      "capital-loss",
                      "hours-per-week"))
  .setOutputCol("numFeatures")

val labelIndexer = new StringIndexer()
  .setInputCol(">50K,<=50K")
  .setOutputCol("label")
  .fit(training)
```
 
```scala
 labelIndexer.transform(training).select(">50K,<=50K", "label").show(8)
```
```
 +----------+-----+
|>50K,<=50K|label|
+----------+-----+
|     <=50K|  0.0|
|     <=50K|  0.0|
|     <=50K|  0.0|
|     <=50K|  0.0|
|     <=50K|  0.0|
|     <=50K|  0.0|
|     <=50K|  0.0|
|     <=50K|  0.0|
+----------+-----+
only showing top 8 rows
```
 
```scala
 assembler.transform(training)
         .select("age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week", "numFeatures")
         .show(5, truncate=false)
```
```
+-----+--------------------------------+
|label|numFeatures                     |
+-----+--------------------------------+
|0.0  |[17.0,192387.0,5.0,0.0,0.0,45.0]|
|0.0  |[17.0,340043.0,8.0,0.0,0.0,12.0]|
|0.0  |[17.0,24090.0,9.0,0.0,0.0,35.0] |
|0.0  |[17.0,25690.0,6.0,0.0,0.0,10.0] |
|0.0  |[17.0,28031.0,5.0,0.0,0.0,16.0] |
+-----+--------------------------------+
only showing top 5 rows
```

```scala
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator


val lr = new LogisticRegression()
                .setFeaturesCol("numFeatures")
                .setLabelCol("label")
                .setRegParam(0.1)

val lrModel = lr.fit(trainData)

val testData = assembler.transform{
                labelIndexer.transform(test)
              }
              
val eval = new BinaryClassificationEvaluator()
                  .setMetricName("areaUnderROC")

println(eval.evaluate(lrModel.transform(testData)))
```
```
0.7937381854879748
```

## Model selection with MLlib
Apache Spark MLlib supports model hyperparameter tuning using tools such as `CrossValidator` and `TrainValidationSplit`. These tools require the following items:

 - Estimator: algorithm or Pipeline to tune
 - Set of ParamMaps: parameters to choose from, sometimes called a “parameter grid” to search over
 - Evaluator: metric to measure how well a fitted Model does on held-out test data
 
In this section we will need to work only with numeric features and a target variable.
At the beginning let's have a look at grid search in action.
We will consider 2 algorithms:
 - [LogisticRegression](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.classification.LogisticRegression)
 - [DecisionTreeClassifier](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.classification.DecisionTreeClassifier)
 
To start with, let's choose one parameter to optimize for each algorithm:
 - LogisticRegression — regularization parameter (*regParam*)
 - DecisonTreeClassifier — maximum depth of the tree (*maxDepth*)
 
The remaining parameters we will leave at their default values. 
To implement grid search procedure one can use
[CrossValidator](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.tuning.CrossValidator) class
combining with [ParamGridBuilder](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.tuning.ParamGridBuilder) class. 
Also we need to specify appropriate evaluator for this task, in our case we should use [BinaryClassificationEvaluator](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.evaluation.BinaryClassificationEvaluator)
(note that its default metric is areaUnderROC, so we don't neet to specify metric via `setMetricName` method call).
Set up 5-fold cross validation scheme.

<div style="font-size:large">K-fold cross-validation</div>
<div style="text-align:left">
  <img src="https://upload.wikimedia.org/wikipedia/commons/1/1c/K-fold_cross_validation_EN.jpg" width="562" height="262">
</div>
<div style="font-size:x-small">
  By Fabian Flöck (Own work) [CC BY-SA 3.0 (http://creativecommons.org/licenses/by-sa/3.0)], via Wikimedia Commons
</div>

**Problem** Try to find the optimal values of these hyperparameters for each algorithm. Plot the average cross-validation metrics for a given value of hyperparameter for each algorithm (hint: use `avgMetrics` field of resulting `CrossValidatorModel`).

```scala
import org.apache.spark.ml.classification.{LogisticRegression, DecisionTreeClassifier, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}


val lr = new LogisticRegression()
                .setFeaturesCol("numFeatures")
                .setLabelCol("label")

val lrParamGrid = new ParamGridBuilder()
  .addGrid(lr.regParam, Array(1e-2, 5e-3, 1e-3, 5e-4, 1e-4))
  .build()

val lrCV = new CrossValidator()
  .setEstimator(lr)
  .setEvaluator(new BinaryClassificationEvaluator)
  .setEstimatorParamMaps(lrParamGrid)
  .setNumFolds(5)

val lrCVModel = lrCV.fit(trainData)

println("cross-validated areaUnderROC: " + lrCVModel.avgMetrics.max)
println("test areaUnderROC: " + eval.evaluate(lrCVModel.transform(testData)))
```
```
cross-validated areaUnderROC: 0.8297755442702006
test areaUnderROC: 0.8068812315222861
```

```scala
val tree = new DecisionTreeClassifier()
                .setFeaturesCol("numFeatures")
                .setLabelCol("label")

val treeParamGrid = new ParamGridBuilder()
  .addGrid(tree.maxDepth, Array(5, 10, 20, 25, 30))
  .build()

val treeCV = new CrossValidator()
  .setEstimator(tree)
  .setEvaluator(new BinaryClassificationEvaluator)
  .setEstimatorParamMaps(treeParamGrid)
  .setNumFolds(5)

val treeCVModel = treeCV.fit(trainData)

println("cross-validated areaUnderROC: " + treeCVModel.avgMetrics.max)
println("test areaUnderROC: " + eval.evaluate(treeCVModel.transform(testData)))
```
```
cross-validated areaUnderROC: 0.7105377328054816
test areaUnderROC: 0.6934402983359256
```

```scala
lrCVModel.getEstimatorParamMaps
                            .map(paramMap => paramMap(lr.regParam))
                            .zip(lrCVModel.avgMetrics)
                            .toSeq.toDF("regParam", "AUC-ROC")
                            .collect
```

<img src="http://telegra.ph/file/62694a00f2434bca1a41a.png" width=900>
</img>

```scala
treeCVModel.getEstimatorParamMaps
                            .map(paramMap => paramMap(tree.maxDepth))
                            .zip(treeCVModel.avgMetrics)
                            .toSeq.toDF("maxDepth", "AUC-ROC")
                            .collect
```

<img src="http://telegra.ph/file/8c2ba01164df95f3525d4.png" width=900>
</img>

## Adding categorical features

Up to this point we did not use categorical features from the dataset. Let's see how additional categorical features will affect the quality of the classification. A common technique to convert categorical feature into numerical ones is [one-hot](https://en.wikipedia.org/wiki/One-hot) encoding. This can be done using [StringIndexer](http://spark.apache.org/docs/1.6.1/api/scala/index.html#org.apache.spark.ml.feature.StringIndexer) transformation followed by [OneHotEncoder](http://spark.apache.org/docs/1.6.1/api/scala/index.html#org.apache.spark.ml.feature.OneHotEncoder) transformation.

*Let's start with encoding just one new feature `occupation` and after that generalize encoding step for all categorical features and combine all processing steps using [pipeline](http://spark.apache.org/docs/1.6.1/ml-guide.html#pipeline)*

```scala
data.groupBy("occupation").count.show(truncate=false)
println(data.select("occupation").distinct.count)
```
```
+-----------------+-----+
|occupation       |count|
+-----------------+-----+
|Sales            |1840 |
|Exec-managerial  |2017 |
|Prof-specialty   |2095 |
|Handlers-cleaners|674  |
|Farming-fishing  |481  |
|Craft-repair     |2057 |
|Transport-moving |799  |
|Priv-house-serv  |90   |
|Protective-serv  |343  |
|Other-service    |1617 |
|Tech-support     |464  |
|Machine-op-inspct|1023 |
|Armed-Forces     |3    |
|Adm-clerical     |1844 |
+-----------------+-----+

14
```

```scala
import org.apache.spark.ml.feature.OneHotEncoder

val occupationIndexer = new StringIndexer()
  .setInputCol("occupation")
  .setOutputCol("occupationIndex")
  .fit(training)

val indexedTrainData = occupationIndexer.transform(training)

val occupationEncoder = new OneHotEncoder()
  .setInputCol("occupationIndex")
  .setOutputCol("occupationVec")

val oheEncodedTrainData = occupationEncoder.transform(indexedTrainData)

oheEncodedTrainData.select("occupation", "occupationVec").show(5, truncate=false)
```
```
+---------------+--------------+
|occupation     |occupationVec |
+---------------+--------------+
|Other-service  |(13,[5],[1.0])|
|Adm-clerical   |(13,[4],[1.0])|
|Exec-managerial|(13,[2],[1.0])|
|Other-service  |(13,[5],[1.0])|
|Other-service  |(13,[5],[1.0])|
+---------------+--------------+
only showing top 5 rows
```

```scala
val assembler = new VectorAssembler()
  .setInputCols(Array("age",
                      "fnlwgt", 
                      "education-num", 
                      "capital-gain", 
                      "capital-loss",
                      "hours-per-week",
                      "occupationVec"))
  .setOutputCol("features")


val trainDataWithOccupation = assembler.transform{
                                labelIndexer.transform(oheEncodedTrainData)
                              }.select("label", "features")
```

*For the sake of brevity, from now let's use only LogisticRegression model.*

```scala
val lr = new LogisticRegression()
  .setFeaturesCol("features")

val lrParamGrid = new ParamGridBuilder()
  .addGrid(lr.regParam, Array(1e-2, 5e-3, 1e-3, 5e-4, 1e-4))
  .build()

val lrCV = new CrossValidator()
  .setEstimator(lr)
  .setEvaluator(new BinaryClassificationEvaluator)
  .setEstimatorParamMaps(lrParamGrid)
  .setNumFolds(5)

val lrCVModel = lrCV.fit(trainDataWithOccupation)

val testDataWithOccupation = assembler.transform{
                                labelIndexer.transform(occupationEncoder.transform(occupationIndexer.transform(test)))
                              }.select("label", "features")

println("cross-validated areaUnderROC: " + lrCVModel.avgMetrics.max)
println("test areaUnderROC: " + eval.evaluate(lrCVModel.transform(testDataWithOccupation)))
```
```
cross-validated areaUnderROC: 0.8447936545404254
test areaUnderROC: 0.823490779891881
```

Adding `occupation` categorical variable yielded an increase in quality.

## Pipelines
