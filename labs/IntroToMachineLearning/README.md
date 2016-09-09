# Introduction To Machine Learning
In this lab we are going to learn how to teach machine learning models, how to correctly set up an experiment, how to tune model hyperparameters and how to compare models. Also we'are going to get familiar with **spark.ml** package as soon as all of the work we'are going to get done using this package.
### Data
We'are going to solve binary classification problem by building the algorithm which determines whether a person makes over 50K a year. Following features are available:
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
## Assignment
To load data from csv file direct to Spark's Dataframe we will use [spark-csv](http://spark-packages.org/package/databricks/spark-csv) package.
To add spark-csv package to spark notebook one could add "com.databricks:spark-csv_2.10:1.4.0" (or "com.databricks:spark-csv_2.11:1.4.0" for Scala 2.11) dependency into customDeps conf section. Alternatively one could specify this dependency in `--packages` command line option while submiting spark application to a cluster (`spark-submit`) or launching spark shell (`spark-shell`). 
Load `data.adult.csv` dataset. Print several rows from the dataset.

```scala
import org.apache.spark.sql.SQLContext

val sqlContext = new SQLContext(sc)

val data = sqlContext.read
    .format("com.databricks.spark.csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load("notebooks/labs/IntroToMachineLearning/data.adult.csv")
```


><pre>
> import org.apache.spark.sql.SQLContext
> sqlContext: org.apache.spark.sql.SQLContext = org.apache.spark.sql.SQLContext@47a0547b
> data: org.apache.spark.sql.DataFrame = [age: int, workclass: string, fnlwgt: int, education: string, education-num: int, marital-status: string, occupation: string, relationship: string, race: string, sex: string, capital-gain: int, capital-loss: int, hours-per-week: int, >50K,<=50K: string]
></pre>




```scala
data.limit(10)
```


><pre>
> res3: org.apache.spark.sql.DataFrame = [age: int, workclass: string, fnlwgt: int, education: string, education-num: int, marital-status: string, occupation: string, relationship: string, race: string, sex: string, capital-gain: int, capital-loss: int, hours-per-week: int, >50K,<=50K: string]
></pre>



Sometimes there are missing values in the data. Sometimes, in the description of the dataset one can found the description of format of missing values. Particularly in the given dataset  missing values are identified by '?' sign.
**Problem** Find all the features with missing values. Remove from the dataset all objects with missing values.

```scala
val missingValsFeatures = data.columns.filter(col => data.filter(data(col) === "?").count > 0)

println("Features with missing values: " + missingValsFeatures.mkString(", "))

val cleanData = missingValsFeatures.foldLeft(data)((df, col) => df.filter(df(col) !== "?"))
cleanData.limit(10)
```


><pre>
> Features with missing values: workclass, occupation
> missingValsFeatures: Array[String] = Array(workclass, occupation)
> cleanData: org.apache.spark.sql.DataFrame = [age: int, workclass: string, fnlwgt: int, education: string, education-num: int, marital-status: string, occupation: string, relationship: string, race: string, sex: string, capital-gain: int, capital-loss: int, hours-per-week: int, >50K,<=50K: string]
> res5: org.apache.spark.sql.DataFrame = [age: int, workclass: string, fnlwgt: int, education: string, education-num: int, marital-status: string, occupation: string, relationship: string, race: string, sex: string, capital-gain: int, capital-loss: int, hours-per-week: int, >50K,<=50K: string]
></pre>



Some preprocessing steps are usually required after loading and cleaning dataset. In this case, these steps will include the following:

 - Select the target variable (the one we want to predict, string column of labels) and map it to an ML column of label indices using [StringIndexer](http://spark.apache.org/docs/1.6.1/api/scala/index.html#org.apache.spark.ml.feature.StringIndexer), give the name "label" to a new variable.
 - Note that not all features are numeric. At first we will work only with numeric features. So let's select them separately in the feature vector "features".

```scala
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
```


><pre>
> import org.apache.spark.ml.feature.VectorAssembler
> import org.apache.spark.ml.feature.StringIndexer
></pre>




```scala
val assembler = new VectorAssembler()
  .setInputCols(Array("fnlwgt", 
                      "education-num", 
                      "capital-gain", 
                      "capital-loss",
                      "hours-per-week"))
  .setOutputCol("features")

val labelIndexer = new StringIndexer()
  .setInputCol(">50K,<=50K")
  .setOutputCol("label")
  .fit(cleanData)

val vecIdxData = assembler.transform{
                labelIndexer.transform(cleanData)
              }.select("label", "features")
```


><pre>
> assembler: org.apache.spark.ml.feature.VectorAssembler = vecAssembler_74ac2027368e
> labelIndexer: org.apache.spark.ml.feature.StringIndexerModel = strIdx_da8c63a32a55
> vecIdxData: org.apache.spark.sql.DataFrame = [label: double, features: vector]
></pre>



## Training classifiers on numeric features

In this section we will need to work only with numeric features and a target variable.
At the beginning let's have a look at grid search in action.
We will consider 3 algorithms:
 - [LogisticRegression](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.classification.LogisticRegression)
 - [DecisionTreeClassifier](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.classification.DecisionTreeClassifier)
 - [RandomForestClassifier](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.classification.RandomForestClassifier)
 
To start with, let's choose one parameter to optimize for each algorithm:
 - LogisticRegression — regularization parameter (*regParam*)
 - DecisonTreeClassifier — maximum depth of the tree (*maxDepth*)
 - RandomForestClassifier — maximum number of bins used for discretizing continuous features (*maxBins*)
 
The remaining parameters we will leave at their default values. 
To implement grid search procedure one can use
[CrossValidator](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.tuning.CrossValidator) class
combining with [ParamGridBuilder](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.tuning.ParamGridBuilder) class. 
Also we need to specify appropriate evaluator for this task, in our case we should use [BinaryClassificationEvaluator](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.evaluation.BinaryClassificationEvaluator)
(note that its default metric is areaUnderROC, so we don't neet to specify metric via `setMetricName` method call).
Set up 5-fold cross validation scheme.

**Problem** Try to find the optimal values of these hyperparameters for each algorithm. Plot the average cross-validation metrics for a given value of hyperparameter for each algorithm (hint: use `avgMetrics` field of resulting `CrossValidatorModel`).

```scala
import org.apache.spark.ml.classification.{LogisticRegression, DecisionTreeClassifier, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
```


><pre>
> import org.apache.spark.ml.classification.{LogisticRegression, DecisionTreeClassifier, RandomForestClassifier}
> import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
> import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
></pre>




```scala
val lr = new LogisticRegression()
val tree = new DecisionTreeClassifier()
val rf = new RandomForestClassifier()

val lrParamGrid = new ParamGridBuilder()
  .addGrid(lr.regParam, Array(5e-3, 2e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5))
  .build()

val treeParamGrid = new ParamGridBuilder()
  .addGrid(tree.maxDepth, Array(5, 10, 20, 25, 30))
  .build()

val rfParamGrid = new ParamGridBuilder()
  .addGrid(rf.maxBins, Array(16, 32, 64, 128, 256))
  .build()

val lrCV = new CrossValidator()
  .setEstimator(lr)
  .setEvaluator(new BinaryClassificationEvaluator)
  .setEstimatorParamMaps(lrParamGrid)
  .setNumFolds(5)

val treeCV = new CrossValidator()
  .setEstimator(tree)
  .setEvaluator(new BinaryClassificationEvaluator)
  .setEstimatorParamMaps(treeParamGrid)
  .setNumFolds(5)

val rfCV = new CrossValidator()
  .setEstimator(rf)
  .setEvaluator(new BinaryClassificationEvaluator)
  .setEstimatorParamMaps(rfParamGrid)
  .setNumFolds(5)

val lrCVModel = lrCV.fit(vecIdxData)
val treeCVModel = treeCV.fit(vecIdxData)
val rfCVModel = rfCV.fit(vecIdxData)
```


><pre>
> lr: org.apache.spark.ml.classification.LogisticRegression = logreg_9f971b2bad00
> tree: org.apache.spark.ml.classification.DecisionTreeClassifier = dtc_5ed3fa18a5cd
> rf: org.apache.spark.ml.classification.RandomForestClassifier = rfc_dbcb4625047a
> lrParamGrid: Array[org.apache.spark.ml.param.ParamMap] = 
> Array({
> 	logreg_9f971b2bad00-regParam: 0.005
> }, {
> 	logreg_9f971b2bad00-regParam: 0.002
> }, {
> 	logreg_9f971b2bad00-regParam: 0.001
> }, {
> 	logreg_9f971b2bad00-regParam: 5.0E-4
> }, {
> 	logreg_9f971b2bad00-regParam: 1.0E-4
> }, {
> 	logreg_9f971b2bad00-regParam: 5.0E-5
> }, {
> 	logreg_9f971b2bad00-regParam: 1.0E-5
> })
> treeParamGrid: Array[org.apache.spark.ml.param.ParamMap] = 
> Array({
> 	dtc_5ed3fa18a5cd-maxDepth: 5
> }, {
> 	dtc_5ed3fa18a5cd-maxDepth: 10
> }, {
> 	dtc_5ed3fa18a5cd-maxDepth: 20
> }, {
> 	dtc_5ed3fa18a5c...
></pre>




```scala
case class LRAvgMetric(regParam: Double, avgMetric: Double)
case class TreeAvgMetric(maxDepth: Double, avgMetric: Double)
case class RFAvgMetric(maxBins: Double, avgMetric: Double)

val lrAvgMetrics = lrCVModel.getEstimatorParamMaps
                            .map(paramMap => paramMap(lr.regParam))
                            .zip(lrCVModel.avgMetrics)
                            .map(p => LRAvgMetric(p._1, p._2))
val treeAvgMetrics = treeCVModel.getEstimatorParamMaps
                            .map(paramMap => paramMap(tree.maxDepth))
                            .zip(treeCVModel.avgMetrics)
                            .map(p => TreeAvgMetric(p._1, p._2))
val rfAvgMetrics = rfCVModel.getEstimatorParamMaps
                            .map(paramMap => paramMap(rf.maxBins))
                            .zip(rfCVModel.avgMetrics)
                            .map(p => RFAvgMetric(p._1, p._2))
```


><pre>
> defined class LRAvgMetric
> defined class TreeAvgMetric
> defined class RFAvgMetric
> lrAvgMetrics: Array[LRAvgMetric] = Array(LRAvgMetric(0.005,0.7965059769394554), LRAvgMetric(0.002,0.7969363092449264), LRAvgMetric(0.001,0.796929835627711), LRAvgMetric(5.0E-4,0.796808917496667), LRAvgMetric(1.0E-4,0.7966207927757235), LRAvgMetric(5.0E-5,0.7965996554856329), LRAvgMetric(1.0E-5,0.7965791958395751))
> treeAvgMetrics: Array[TreeAvgMetric] = Array(TreeAvgMetric(5.0,0.41176499148651047), TreeAvgMetric(10.0,0.530571725844926), TreeAvgMetric(20.0,0.6263032570262157), TreeAvgMetric(25.0,0.6258948084014668), TreeAvgMetric(30.0,0.6249428711637058))
> rfAvgMetrics: Array[RFAvgMetric] = Array(RFAvgMetric(16.0,0.8004135537913268), RFAvgMetric(32.0,0.8047945957499629), RFAvgMetric(64.0,0.8084418765444344), RF...
></pre>




```scala
CustomC3Chart(lrAvgMetrics,
              """{ data: { x: 'regParam', 
                         },
                   axis: {
                      y: {
                        label: 'AUC-ROC'
                      },
                      x: {
                        label: 'regParam'
                        }
                   } 
                  }""")

```


><pre>
> res54: notebook.front.widgets.CustomC3Chart[Array[LRAvgMetric]] = <CustomC3Chart widget>
></pre>

<img src='https://github.com/drewnoff/spark-notebook-ml-labs/blob/master/labs/IntroToMachineLearning/images/lrAvgMetrics.png?raw=true' alt='plot' height='252' width='978'></img>


```scala
CustomC3Chart(treeAvgMetrics,
              """{ data: { x: 'maxDepth', 
                         },
                   axis: {
                      y: {
                        label: 'AUC-ROC'
                      },
                      x: {
                        label: 'maxDepth'
                        }
                   } 
                  }""")
```


><pre>
> res56: notebook.front.widgets.CustomC3Chart[Array[TreeAvgMetric]] = <CustomC3Chart widget>
></pre>

<img src='https://github.com/drewnoff/spark-notebook-ml-labs/blob/master/labs/IntroToMachineLearning/images/treeAvgMetrics.png?raw=true' alt='plot' height='252' width='978'></img>


```scala
CustomC3Chart(rfAvgMetrics,
              """{ data: { x: 'maxBins', 
                         },
                   axis: {
                      y: {
                        label: 'AUC-ROC'
                      },
                      x: {
                        label: 'maxBins'
                        }
                   } 
                  }""")

```


><pre>
> res58: notebook.front.widgets.CustomC3Chart[Array[RFAvgMetric]] = <CustomC3Chart widget>
></pre>

<img src='https://github.com/drewnoff/spark-notebook-ml-labs/blob/master/labs/IntroToMachineLearning/images/rfAvgMetrics.png?raw=true' alt='plot' height='252' width='978'></img>

What can you say about the resulting graphs?
Also let's choose the number of trees in RandomForestClassifier algorithm. In general, RandomForest is not overfitting while increasing the number of trees, so with increase of the number of trees its quality will not become worse. Therefore, we will select the number of trees at which cross-validation score  stabilizes.

```scala
val rf = new RandomForestClassifier()
  .setMaxBins(128)
val rfParamGrid = new ParamGridBuilder()
  .addGrid(rf.numTrees, Array(20, 40, 80, 120, 160, 200, 250))
  .build()

val rfCV = new CrossValidator()
  .setEstimator(rf)
  .setEvaluator(new BinaryClassificationEvaluator)
  .setEstimatorParamMaps(rfParamGrid)
  .setNumFolds(5)

val rfCVModel = rfCV.fit(vecIdxData)
```


><pre>
> rf: org.apache.spark.ml.classification.RandomForestClassifier = rfc_e0e9e70eaf93
> rfParamGrid: Array[org.apache.spark.ml.param.ParamMap] = 
> Array({
> 	rfc_e0e9e70eaf93-numTrees: 20
> }, {
> 	rfc_e0e9e70eaf93-numTrees: 40
> }, {
> 	rfc_e0e9e70eaf93-numTrees: 80
> }, {
> 	rfc_e0e9e70eaf93-numTrees: 120
> }, {
> 	rfc_e0e9e70eaf93-numTrees: 160
> }, {
> 	rfc_e0e9e70eaf93-numTrees: 200
> }, {
> 	rfc_e0e9e70eaf93-numTrees: 250
> })
> rfCV: org.apache.spark.ml.tuning.CrossValidator = cv_e932a86e0937
> rfCVModel: org.apache.spark.ml.tuning.CrossValidatorModel = cv_e932a86e0937
></pre>




```scala
case class RFAvgMetric(numTrees: Double, avgMetric: Double)
val rfAvgMetrics = rfCVModel.getEstimatorParamMaps
                            .map(paramMap => paramMap(rf.numTrees))
                            .zip(rfCVModel.avgMetrics)
                            .map(p => RFAvgMetric(p._1, p._2))
```


><pre>
> defined class RFAvgMetric
> rfAvgMetrics: Array[RFAvgMetric] = Array(RFAvgMetric(20.0,0.8099982134171134), RFAvgMetric(40.0,0.8115014223601622), RFAvgMetric(80.0,0.8123633350366459), RFAvgMetric(120.0,0.812934951608542), RFAvgMetric(160.0,0.8133463794945599), RFAvgMetric(200.0,0.812897374632533), RFAvgMetric(250.0,0.8133296213014655))
></pre>




```scala
CustomC3Chart(rfAvgMetrics,
              """{ data: { x: 'numTrees', 
                         },
                   axis: {
                      y: {
                        label: 'AUC-ROC'
                      },
                      x: {
                        label: 'numTrees'
                        }
                   } 
                  }""")
```


><pre>
> res62: notebook.front.widgets.CustomC3Chart[Array[RFAvgMetric]] = <CustomC3Chart widget>
></pre>

<img src='https://github.com/drewnoff/spark-notebook-ml-labs/blob/master/labs/IntroToMachineLearning/images/rfAvgMetrics2.png?raw=true' alt='plot' height='252' width='978'></img>

One can see that there is a stabilization of the quality at about 160 trees in random forest.
Some algorithms are sensitive to the scale of the features. Let's look at the features to make sure that raw features can have a pretty big difference in scale.
**Problem** Build histograms for features *age*, *fnlwgt*, *capital-gain*.

```scala
val ageRdd = cleanData.select("age").rdd.map(r => r.getAs[Integer](0).toDouble)
val fnlwgtRdd = cleanData.select("fnlwgt").rdd.map(r => r.getAs[Integer](0).toDouble)
val cgainRdd = cleanData.select("capital-gain").rdd.map(r => r.getAs[Integer](0).toDouble)

val ageHist = ageRdd.histogram(10)
val fnlwgtHist = fnlwgtRdd.histogram(20)
val cgainHist = cgainRdd.histogram(50)

case class AgeHistPoint(ageBucket: Double, age: Long)
case class FnlwgtHistPoint(fnlwgtBucket: Double, fnlwgt: Long)
case class CgainHistPoint(cgainBucket: Double, cgain: Long)

val ageHistData = ageHist._1.zip(ageHist._2).map(pp => AgeHistPoint(pp._1, pp._2))
val fnlwgtHistData = fnlwgtHist._1.zip(fnlwgtHist._2).map(pp => FnlwgtHistPoint(pp._1, pp._2))
val cgainHistData = cgainHist._1.zip(cgainHist._2).map(pp => CgainHistPoint(pp._1, pp._2))
```


><pre>
> ageRdd: org.apache.spark.rdd.RDD[Double] = MapPartitionsRDD[134008] at map at <console>:38
> fnlwgtRdd: org.apache.spark.rdd.RDD[Double] = MapPartitionsRDD[134017] at map at <console>:39
> cgainRdd: org.apache.spark.rdd.RDD[Double] = MapPartitionsRDD[134026] at map at <console>:40
> ageHist: (Array[Double], Array[Long]) = (Array(17.0, 24.3, 31.6, 38.9, 46.2, 53.5, 60.8, 68.1, 75.4, 82.7, 90.0),Array(2466, 2776, 2966, 3066, 1931, 1203, 667, 183, 62, 27))
> fnlwgtHist: (Array[Double], Array[Long]) = (Array(19302.0, 92572.15, 165842.3, 239112.45, 312382.6, 385652.75, 458922.9, 532193.05, 605463.2, 678733.35, 752003.5, 825273.65, 898543.8, 971813.95, 1045084.1, 1118354.25, 1191624.4, 1264894.55, 1338164.7, 1411434.85, 1484705.0),Array(2389, 4292, 4922, 1941, 1069, 459, 149, 66, 27, 17, 4, 0, 2, 4, ...
></pre>




```scala
CustomC3Chart(ageHistData,
             chartOptions = """
             { data: { x: 'ageBucket', 
                       type: 'bar'},
               bar: {
                     width: {ratio: 0.9}
                    },
              axis: {
                    y: {
                      label: 'Count'
                      }
                   }
             }
             """)
```

><pre>
> res65: notebook.front.widgets.CustomC3Chart[Array[AgeHistPoint]] = <CustomC3Chart widget>
></pre>

<img src='https://github.com/drewnoff/spark-notebook-ml-labs/blob/master/labs/IntroToMachineLearning/images/ageHistData.png?raw=true' alt='plot' height='252' width='978'></img>



```scala
CustomC3Chart(fnlwgtHistData,
             chartOptions = """
             { data: { x: 'fnlwgtBucket', 
                       type: 'bar',
                       colors: {fnlwgt: 'green'}},
               bar: {
                     width: {ratio: 0.9}
                    },
              axis: {
                    y: {
                      label: 'Count'
                      }
                   }
             }
             """)
```


><pre>
> res67: notebook.front.widgets.CustomC3Chart[Array[FnlwgtHistPoint]] = <CustomC3Chart widget>
></pre>

<img src='https://github.com/drewnoff/spark-notebook-ml-labs/blob/master/labs/IntroToMachineLearning/images/fnlwgtHistData.png?raw=true' alt='plot' height='252' width='978'></img>



```scala
CustomC3Chart(cgainHistData,
             chartOptions = """
             { data: { x: 'cgainBucket', 
                       type: 'bar',
                       colors: {cgain: 'purple'}},
               bar: {
                     width: {ratio: 0.9}
                    },
              axis: {
                    y: {
                      label: 'Count'
                      }
                   }
             }
             """)
```


><pre>
> res69: notebook.front.widgets.CustomC3Chart[Array[CgainHistPoint]] = <CustomC3Chart widget>
></pre>

<img src='https://github.com/drewnoff/spark-notebook-ml-labs/blob/master/labs/IntroToMachineLearning/images/cgainHistData.png?raw=true' alt='plot' height='252' width='978'></img>


Now when you see the histograms you can answer the following questions. What is special about each feature? Does it affect the quality of algorithms? How can we improve the quality of the algorithms?
One can improve the quality of algorithms by feature scaling. Feature scaling can be performed, for example, one of the following ways:
 - $x_{new} = \dfrac{x - \mu}{\sigma}$, where $\mu, \sigma$ — sample mean and sample standard deviation
 - $x_{new} = \dfrac{x - x_{min}}{x_{max} - x_{min}}$, where $[x_{min}, x_{max}]$ — the range of values
 
Similar scaling schemes implemented in the classes [StandardScaler](http://spark.apache.org/docs/1.6.1/api/scala/index.html#org.apache.spark.ml.feature.StandardScaler) and [MinMaxScaler](http://spark.apache.org/docs/1.6.1/api/scala/index.html#org.apache.spark.ml.feature.MinMaxScaler).
**Problem** Scale all the numeric features using one of these methods and repeat hyperparameters tuning step.
*for the sake of brevity, from now I will use only LogisticRegression model.*

```scala
import org.apache.spark.ml.feature.StandardScaler

val scaler = new StandardScaler()
  .setInputCol("features")
  .setOutputCol("scaledFeatures")
  .setWithStd(true)
  .setWithMean(true)

val scalerModel = scaler.fit(vecIdxData)
val scaledData = scalerModel.transform(vecIdxData)
```


><pre>
> import org.apache.spark.ml.feature.StandardScaler
> scaler: org.apache.spark.ml.feature.StandardScaler = stdScal_1b221a6d19d2
> scalerModel: org.apache.spark.ml.feature.StandardScalerModel = stdScal_1b221a6d19d2
> scaledData: org.apache.spark.sql.DataFrame = [label: double, features: vector, scaledFeatures: vector]
></pre>




```scala
scaledData.limit(5)
```


><pre>
> res72: org.apache.spark.sql.DataFrame = [label: double, features: vector, scaledFeatures: vector]
></pre>




```scala
val lr = new LogisticRegression()
  .setFeaturesCol("scaledFeatures")

val lrParamGrid = new ParamGridBuilder()
  .addGrid(lr.regParam, Array(5e-3, 2e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5))
  .build()

val lrCV = new CrossValidator()
  .setEstimator(lr)
  .setEvaluator(new BinaryClassificationEvaluator)
  .setEstimatorParamMaps(lrParamGrid)
  .setNumFolds(5)

val lrCVModel = lrCV.fit(scaledData)
```


><pre>
> lr: org.apache.spark.ml.classification.LogisticRegression = logreg_c590cc04ea2c
> lrParamGrid: Array[org.apache.spark.ml.param.ParamMap] = 
> Array({
> 	logreg_c590cc04ea2c-regParam: 0.005
> }, {
> 	logreg_c590cc04ea2c-regParam: 0.002
> }, {
> 	logreg_c590cc04ea2c-regParam: 0.001
> }, {
> 	logreg_c590cc04ea2c-regParam: 5.0E-4
> }, {
> 	logreg_c590cc04ea2c-regParam: 1.0E-4
> }, {
> 	logreg_c590cc04ea2c-regParam: 5.0E-5
> }, {
> 	logreg_c590cc04ea2c-regParam: 1.0E-5
> })
> lrCV: org.apache.spark.ml.tuning.CrossValidator = cv_bdbb219bd167
> lrCVModel: org.apache.spark.ml.tuning.CrossValidatorModel = cv_bdbb219bd167
></pre>




```scala
lrCVModel.avgMetrics.max
```


><pre>
> res75: Double = 0.7969366525522104
></pre>

> 0.7969366525522104

You can also perform grid search on several hyperparameters and find the optimum combination for each algorithm. Here is just one example:
 - LogisticRegression — regularization parameter (*regParam*) and ElasticNet mixing parameter (*elasticNetParam*)
 - DecisonTreeClassifier — maximum depth of the tree (*maxDepth*) and criterion for information gain calculation (*impurity*)
 - RandomForestClassifier — criterion for information gain calculation (*impurity*) and fraction of the training data used for learning each decision tree (*subsamplingRate*)

```scala
val lr = new LogisticRegression()
  .setFeaturesCol("scaledFeatures")

val lrParamGrid = new ParamGridBuilder()
  .addGrid(lr.regParam, Array(5e-3, 2e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5))
  .addGrid(lr.elasticNetParam, Array(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
  .build()

val lrCV = new CrossValidator()
  .setEstimator(lr)
  .setEvaluator(new BinaryClassificationEvaluator)
  .setEstimatorParamMaps(lrParamGrid)
  .setNumFolds(5)

val lrCVModel = lrCV.fit(scaledData)
```


><pre>
> lr: org.apache.spark.ml.classification.LogisticRegression = logreg_d208be8efefb
> lrParamGrid: Array[org.apache.spark.ml.param.ParamMap] = 
> Array({
> 	logreg_d208be8efefb-elasticNetParam: 0.0,
> 	logreg_d208be8efefb-regParam: 0.005
> }, {
> 	logreg_d208be8efefb-elasticNetParam: 0.0,
> 	logreg_d208be8efefb-regParam: 0.002
> }, {
> 	logreg_d208be8efefb-elasticNetParam: 0.0,
> 	logreg_d208be8efefb-regParam: 0.001
> }, {
> 	logreg_d208be8efefb-elasticNetParam: 0.0,
> 	logreg_d208be8efefb-regParam: 5.0E-4
> }, {
> 	logreg_d208be8efefb-elasticNetParam: 0.0,
> 	logreg_d208be8efefb-regParam: 1.0E-4
> }, {
> 	logreg_d208be8efefb-elasticNetParam: 0.0,
> 	logreg_d208be8efefb-regParam: 5.0E-5
> }, {
> 	logreg_d208be8efefb-elasticNetParam: 0.0,
> 	logreg_d208be8efefb-regParam: 1.0E-5
> }, {
> 	logreg_d208be8efefb-elasticNetParam: 0.2,
> 	logreg_d...
></pre>




```scala
lrCVModel.avgMetrics.max
```


><pre>
> res78: Double = 0.7970238858408468
></pre>

> 0.7970238858408468

## Adding categorical features 
Up to this point we did not use categorical features from the dataset. Let's see how additional categorical features will affect the quality of the classification. A common technique to convert categorical feature into numerical ones is one-hot encoding. This can be done using [StringIndexer](http://spark.apache.org/docs/1.6.1/api/scala/index.html#org.apache.spark.ml.feature.StringIndexer) transformation followed by [OneHotEncoder](http://spark.apache.org/docs/1.6.1/api/scala/index.html#org.apache.spark.ml.feature.OneHotEncoder) transformation.
*I'm going to start with encoding just one new feature `occupation` and after that generalize encoding step for all categorical features and combine all processing steps using [pipeline](http://spark.apache.org/docs/1.6.1/ml-guide.html#pipeline)*

```scala
import org.apache.spark.ml.feature.OneHotEncoder

val indexer = new StringIndexer()
  .setInputCol("occupation")
  .setOutputCol("occupationIndex")
  .fit(cleanData)
val indexedData = indexer.transform(cleanData)

val encoder = new OneHotEncoder()
  .setInputCol("occupationIndex")
  .setOutputCol("occupationVec")
val oheEncodeData = encoder.transform(indexedData)

oheEncodeData.select("occupation", "occupationVec").limit(3)
```


><pre>
> import org.apache.spark.ml.feature.OneHotEncoder
> indexer: org.apache.spark.ml.feature.StringIndexerModel = strIdx_87c12d392f1d
> indexedData: org.apache.spark.sql.DataFrame = [age: int, workclass: string, fnlwgt: int, education: string, education-num: int, marital-status: string, occupation: string, relationship: string, race: string, sex: string, capital-gain: int, capital-loss: int, hours-per-week: int, >50K,<=50K: string, occupationIndex: double]
> encoder: org.apache.spark.ml.feature.OneHotEncoder = oneHot_a57cb13b9cf0
> oheEncodeData: org.apache.spark.sql.DataFrame = [age: int, workclass: string, fnlwgt: int, education: string, education-num: int, marital-status: string, occupation: string, relationship: string, race: string, sex: string, capital-gain: int, capital-loss: int, hours-per-w...
></pre>




```scala
val assembler = new VectorAssembler()
  .setInputCols(Array("fnlwgt", 
                      "education-num", 
                      "capital-gain", 
                      "capital-loss",
                      "hours-per-week",
                      "occupationVec"))
  .setOutputCol("features")

val labelIndexer = new StringIndexer()
  .setInputCol(">50K,<=50K")
  .setOutputCol("label")
  .fit(cleanData)

val vecIdxData = assembler.transform{
                labelIndexer.transform(oheEncodeData)
              }.select("label", "features")
```


><pre>
> assembler: org.apache.spark.ml.feature.VectorAssembler = vecAssembler_d91b57cd47dc
> labelIndexer: org.apache.spark.ml.feature.StringIndexerModel = strIdx_f78720bf1da1
> vecIdxData: org.apache.spark.sql.DataFrame = [label: double, features: vector]
></pre>




```scala
val scaler = new StandardScaler()
  .setInputCol("features")
  .setOutputCol("scaledFeatures")
  .setWithStd(true)
  .setWithMean(false) // setWithMean(true) doesn't work with sparse features

val scalerModel = scaler.fit(vecIdxData)
val scaledData = scalerModel.transform(vecIdxData)
```


><pre>
> scaler: org.apache.spark.ml.feature.StandardScaler = stdScal_b0eccf6ae0a0
> scalerModel: org.apache.spark.ml.feature.StandardScalerModel = stdScal_b0eccf6ae0a0
> scaledData: org.apache.spark.sql.DataFrame = [label: double, features: vector, scaledFeatures: vector]
></pre>




```scala
val lr = new LogisticRegression()
  .setFeaturesCol("scaledFeatures")

val lrParamGrid = new ParamGridBuilder()
  .addGrid(lr.regParam, Array(5e-3, 2e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5))
  .build()

val lrCV = new CrossValidator()
  .setEstimator(lr)
  .setEvaluator(new BinaryClassificationEvaluator)
  .setEstimatorParamMaps(lrParamGrid)
  .setNumFolds(5)

val lrCVModel = lrCV.fit(scaledData)
```


><pre>
> lr: org.apache.spark.ml.classification.LogisticRegression = logreg_62491ddebad1
> lrParamGrid: Array[org.apache.spark.ml.param.ParamMap] = 
> Array({
> 	logreg_62491ddebad1-regParam: 0.005
> }, {
> 	logreg_62491ddebad1-regParam: 0.002
> }, {
> 	logreg_62491ddebad1-regParam: 0.001
> }, {
> 	logreg_62491ddebad1-regParam: 5.0E-4
> }, {
> 	logreg_62491ddebad1-regParam: 1.0E-4
> }, {
> 	logreg_62491ddebad1-regParam: 5.0E-5
> }, {
> 	logreg_62491ddebad1-regParam: 1.0E-5
> })
> lrCV: org.apache.spark.ml.tuning.CrossValidator = cv_5281642b0b6c
> lrCVModel: org.apache.spark.ml.tuning.CrossValidatorModel = cv_5281642b0b6c
></pre>




```scala
lrCVModel.avgMetrics.max
```


><pre>
> res85: Double = 0.8188199134964154
></pre>

> 0.8188199134964154

Adding one categorical yielded a significant increase in quality.
## Pipelines
Using [pipelines](http://spark.apache.org/docs/1.6.1/ml-guide.html#pipeline) one can combine all the processing steps into one pipeline and perform grid search against hyperparameters of all tunable steps included in the pipeline. Also it's easy to extend given pipeline with new steps. Let's see how we can combine all the steps made so far into one pipeline.

```scala
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer,
                                    IndexToString, 
                                    VectorIndexer,
                                    OneHotEncoder,
                                    VectorAssembler,
                                    StandardScaler}
import org.apache.spark.ml.classification.LogisticRegression

val labelIndexer = new StringIndexer()
  .setInputCol(">50K,<=50K")
  .setOutputCol("label")
  .fit(cleanData)

val featureIndexer = new StringIndexer()
  .setInputCol("occupation")
  .setOutputCol("occupationIndex")
  .fit(cleanData)

val encoder = new OneHotEncoder()
  .setInputCol("occupationIndex")
  .setOutputCol("occupationVec")

val assembler = new VectorAssembler()
  .setInputCols(Array("fnlwgt", 
                      "education-num", 
                      "capital-gain", 
                      "capital-loss",
                      "hours-per-week",
                      "occupationVec"))
  .setOutputCol("features")

val scaler = new StandardScaler()
  .setInputCol("features")
  .setOutputCol("scaledFeatures")
  .setWithStd(true)
  .setWithMean(false)

val lr = new LogisticRegression()
  .setFeaturesCol("scaledFeatures")

// Convert predicted labels back to original labels.
val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labels)

// Chain indexers, assembler, scaler and classifier and converter in a Pipeline
val pipeline = new Pipeline()
  .setStages(Array(labelIndexer, 
                   featureIndexer, 
                   encoder, 
                   assembler, 
                   scaler,
                   lr,
                   labelConverter))
```


><pre>
> import org.apache.spark.ml.Pipeline
> import org.apache.spark.ml.feature.{StringIndexer, IndexToString, VectorIndexer, OneHotEncoder, VectorAssembler, StandardScaler}
> import org.apache.spark.ml.classification.LogisticRegression
> labelIndexer: org.apache.spark.ml.feature.StringIndexerModel = strIdx_df14d97da7d4
> featureIndexer: org.apache.spark.ml.feature.StringIndexerModel = strIdx_05f7cb9f76b5
> encoder: org.apache.spark.ml.feature.OneHotEncoder = oneHot_46d21d333425
> assembler: org.apache.spark.ml.feature.VectorAssembler = vecAssembler_6ddfcefb6f64
> scaler: org.apache.spark.ml.feature.StandardScaler = stdScal_4632e654cc21
> lr: org.apache.spark.ml.classification.LogisticRegression = logreg_e6efb6e4c718
> labelConverter: org.apache.spark.ml.feature.IndexToString = idxToStr_72fd24ab9266
> pipeline: o...
></pre>




```scala
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder,
                                   CrossValidator}

val lrParamGrid = new ParamGridBuilder()
  .addGrid(lr.regParam, Array(5e-3, 2e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5))
  .build()

val cv = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(new BinaryClassificationEvaluator)
  .setEstimatorParamMaps(lrParamGrid)
  .setNumFolds(5)

val cvModel = cv.fit(cleanData)
```


><pre>
> import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
> import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
> lrParamGrid: Array[org.apache.spark.ml.param.ParamMap] = 
> Array({
> 	logreg_e6efb6e4c718-regParam: 0.005
> }, {
> 	logreg_e6efb6e4c718-regParam: 0.002
> }, {
> 	logreg_e6efb6e4c718-regParam: 0.001
> }, {
> 	logreg_e6efb6e4c718-regParam: 5.0E-4
> }, {
> 	logreg_e6efb6e4c718-regParam: 1.0E-4
> }, {
> 	logreg_e6efb6e4c718-regParam: 5.0E-5
> }, {
> 	logreg_e6efb6e4c718-regParam: 1.0E-5
> })
> cv: org.apache.spark.ml.tuning.CrossValidator = cv_df3d0a61c4b1
> cvModel: org.apache.spark.ml.tuning.CrossValidatorModel = cv_df3d0a61c4b1
></pre>




```scala
cvModel.avgMetrics.max
```


><pre>
> res89: Double = 0.8188199134964158
></pre>

> 0.8188199134964158

Now let's extend our pipeline by adding one-hot encoding step for each categorical feature.

```scala
val categCols = Array("workclass", "education", "marital-status", "occupation", "relationship", "race", "sex")

val featureIndexers: Array[org.apache.spark.ml.PipelineStage] = categCols.map(
  cname => new StringIndexer()
    .setInputCol(cname)
    .setOutputCol(s"${cname}_index")
)

val oneHotEncoders = categCols.map(
    cname => new OneHotEncoder()
     .setInputCol(s"${cname}_index")
     .setOutputCol(s"${cname}_vec")
)

val assembler = new VectorAssembler()
  .setInputCols(Array("fnlwgt", 
                      "education-num", 
                      "capital-gain", 
                      "capital-loss",
                      "hours-per-week") ++
                categCols.map(cname => s"${cname}_vec"))
  .setOutputCol("features")

val pipeline = new Pipeline()
  .setStages(Array(labelIndexer) ++
             featureIndexers ++
             oneHotEncoders ++
             Array(assembler, 
                   scaler,
                   lr,
                   labelConverter))
```


><pre>
> categCols: Array[String] = Array(workclass, education, marital-status, occupation, relationship, race, sex)
> featureIndexers: Array[org.apache.spark.ml.PipelineStage] = Array(strIdx_6af394e95c82, strIdx_efe5d50c8e14, strIdx_a1ee52409664, strIdx_05a97ffa4613, strIdx_54d7b42ce869, strIdx_5f0505479fda, strIdx_eaab84cd46dd)
> oneHotEncoders: Array[org.apache.spark.ml.feature.OneHotEncoder] = Array(oneHot_5840dee9ce71, oneHot_2b9299c7b669, oneHot_f05fced49583, oneHot_165aa2b8d775, oneHot_568553ff7b54, oneHot_1a5ef0c0ca89, oneHot_7de5c378b483)
> assembler: org.apache.spark.ml.feature.VectorAssembler = vecAssembler_78aeab014ef5
> pipeline: org.apache.spark.ml.Pipeline = pipeline_036c7ddb773e
></pre>




```scala
val cv = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(new BinaryClassificationEvaluator)
  .setEstimatorParamMaps(lrParamGrid)
  .setNumFolds(5)

val cvModel = cv.fit(cleanData)
```


><pre>
> cv: org.apache.spark.ml.tuning.CrossValidator = cv_f1c621998dd4
> cvModel: org.apache.spark.ml.tuning.CrossValidatorModel = cv_f1c621998dd4
></pre>




```scala
cvModel.avgMetrics.max
```


><pre>
> res19: Double = 0.9010213957983958
></pre>

> 0.9010213957983958

We have obtained a significant boost in quality of classification.
You can continue to modify and expand the pipeline by adding new steps of data transformation.
For example, one can try to add [QuantileDiscretizer](http://spark.apache.org/docs/1.6.1/api/scala/index.html#org.apache.spark.ml.feature.QuantileDiscretizer)
transformation applied to some numeric feature such as `fnlwgt` and add `numBuckets` parameter values
to pipeline parameters grid and see how it will affect cross validation score.
And another nice thing about pipeline is that then you have trained model from pipeline you can easily apply it to new data without manualy performing all that preprocessing steps.

```scala
// Attention: here we're just pretending what testData is the new data we haven't seen yet
// but that's not true because we've already trained on this data, 
// however that's fine for illustration purposes
val testData = data.sample(false, 0.01)
val testCleanData = missingValsFeatures.foldLeft(testData)((df, col) => df.filter(df(col) !== "?"))

// cvModel uses the best model found
cvModel.transform(testCleanData)
       .select(">50K,<=50K", 
               "features", // scaledFeatures actually
               "probability",
               "prediction",
               "predictedLabel")
       .limit(10)
```


><pre>
> testData: org.apache.spark.sql.DataFrame = [age: int, workclass: string, fnlwgt: int, education: string, education-num: int, marital-status: string, occupation: string, relationship: string, race: string, sex: string, capital-gain: int, capital-loss: int, hours-per-week: int, >50K,<=50K: string]
> testCleanData: org.apache.spark.sql.DataFrame = [age: int, workclass: string, fnlwgt: int, education: string, education-num: int, marital-status: string, occupation: string, relationship: string, race: string, sex: string, capital-gain: int, capital-loss: int, hours-per-week: int, >50K,<=50K: string]
> res38: org.apache.spark.sql.DataFrame = [>50K,<=50K: string, features: vector, probability: vector, prediction: double, predictedLabel: string]
></pre>



We've learned how to teach machine learning models, perform hyperparameters tuning and compare models using cross-validation. We've learned how to use spark.ml package and it's powerful pipeline conception.
