# Bag of Words Meets Bags of Popcorn
In this lab we're going to work with IMDB Movies Reviews dataset from kaggle competition [Bag of Words Meets Bags of Popcorn](https://www.kaggle.com/c/word2vec-nlp-tutorial/data).

<div style="text-align:center">
  <img src="http://i.imgur.com/QZgxFic.png">
</div>

The task is to determine whether the given movie review is positive or negative. This is one example of the problem of text [sentiment analysis](https://en.wikipedia.org/wiki/Sentiment_analysis). Here is one example of review from the dataset:

    When I saw this film in the 1950s, I wanted to be a scientist too. There was something magical and useful in Science. I took a girl - friend along to see it a second time. I don't think she was as impressed as I was! This film was comical yet serious, at a time when synthetic fibres were rather new. Lessons from this film could be applied to issues relating to GM experimentation of today.
Load labeledTrainData.tsv dataset. To load data from csv file direct to Spark's Dataframe one can use [spark-csv](http://spark-packages.org/package/databricks/spark-csv) package.
To add spark-csv package to spark notebook one could add "com.databricks:spark-csv_2.10:1.4.0" (or "com.databricks:spark-csv_2.11:1.4.0" for Scala 2.11) dependency into customDeps conf section. Alternatively one could specify this dependency in `--packages` command line option while submiting spark application to a cluster (`spark-submit`) or launching spark shell (`spark-shell`).
For tsv format use appropriate value of `delimiter` option.

```scala
import org.apache.spark.sql.SQLContext

val sqlContext = new SQLContext(sc)

val data = sqlContext.read
    .format("com.databricks.spark.csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .option("delimiter", "\t")
    .load("notebooks/labs/BagOfWordsMeetsBagsOfPopcorn/labeledTrainData.tsv")
```


><pre>
> import org.apache.spark.sql.SQLContext
> sqlContext: org.apache.spark.sql.SQLContext = org.apache.spark.sql.SQLContext@d74453b
> data: org.apache.spark.sql.DataFrame = [id: string, sentiment: int, review: string]
></pre>




```scala
data.limit(5).show
```


><pre>
> +-------+---------+--------------------+
> |     id|sentiment|              review|
> +-------+---------+--------------------+
> | 5814_8|        1|With all this stu...|
> | 7759_3|        0|The film starts w...|
> | 8196_8|        1|I dont know why p...|
> | 7166_2|        0|This movie could ...|
> |10633_1|        0|I watched this vi...|
> +-------+---------+--------------------+
></pre>



How many positive and negative reviews in this dataset?

```scala
data.groupBy("sentiment").count.show
```


><pre>
> +---------+-----+
> |sentiment|count|
> +---------+-----+
> |        0| 6387|
> |        1| 6990|
> +---------+-----+
></pre>



As we can see, almost half of the reviews are positive and the other half of the reviews are negative. Such datasets are called balanced. But let's make things a bit more interesting and remove three quarters of positive reviews from the dataset and thus we will make the dataset unbalanced.

```scala
val unbalancedData = data.filter(data("sentiment") === 1)
                         .sample(false, 0.25)
                         .unionAll(data.filter(data("sentiment") === 0))
unbalancedData.groupBy("sentiment").count.show
```


><pre>
> +---------+-----+
> |sentiment|count|
> +---------+-----+
> |        0| 6387|
> |        1| 1784|
> +---------+-----+
> 
> unbalancedData: org.apache.spark.sql.DataFrame = [id: string, sentiment: int, review: string]
></pre>



For model quality assessment we will be using train test split with 75% of the data is used for training and 25% for testing. Two important notes:
 - It is good to have a reproducible split on train and test data (hint: use seed param).
 - it is good to preserve the percentage of samples for each class in each split/fold especially in the case of a highly unbalanced classes (follow [the ticket](https://issues.apache.org/jira/browse/SPARK-8971)).

```scala
// Split the data into training and test sets (25% held out for testing)
val Array(trainingData, testData) = unbalancedData.randomSplit(Array(0.75, 0.25), seed=547)
```


><pre>
> trainingData: org.apache.spark.sql.DataFrame = [id: string, sentiment: int, review: string]
> testData: org.apache.spark.sql.DataFrame = [id: string, sentiment: int, review: string]
></pre>




```scala
println(trainingData.groupBy("sentiment").count.show)
println(testData.groupBy("sentiment").count.show)
```


><pre>
> +---------+-----+
> |sentiment|count|
> +---------+-----+
> |        0| 4757|
> |        1| 1304|
> +---------+-----+
> 
> ()
> +---------+-----+
> |sentiment|count|
> +---------+-----+
> |        0| 1630|
> |        1|  452|
> +---------+-----+
> 
> ()
></pre>



One of the difficulties of this task is textual representation of the data because there is no universal method of feature extraction from the texts.
In the course of the lab we will get a few feature representations of the data which will be compared with each other.
## Bag of words
First we will try the simplest approach, namely [bag-of-words](https://en.wikipedia.org/wiki/Bag-of-words_model). With bag-of-words each text will be represented as a vector of numbers with the size equal to the size of the dictionary. On each position of the vector there will be a counter which represents how many times corresponding word was found in this text. This representation one can obtain using [CountVectorizer](http://spark.apache.org/docs/1.6.1/api/scala/index.html#org.apache.spark.ml.feature.CountVectorizer).

But before making features from our data we have to perform data cleaning and text preprocessing steps.
There is a good point about data cleaning and text preprocessing in corresponding [tutorial](https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words):

    When considering how to clean the text, we should think about the data problem we are trying to solve. For many problems, it makes sense to remove punctuation. On the other hand, in this case, we are tackling a sentiment analysis problem, and it is possible that "!!!" or ":-(" could carry sentiment, and should be treated as words.
    
Removing [stop words](https://en.wikipedia.org/wiki/Stop_words) while constructing bag-of-words is also fa good practice.

All these steps can be implemented using sequence of the following feature transformers:
[RegexTokenizer](http://spark.apache.org/docs/1.6.1/api/scala/index.html#org.apache.spark.ml.feature.RegexTokenizer)
followed by [StopWordsRemover](http://spark.apache.org/docs/1.6.1/api/scala/index.html#org.apache.spark.ml.feature.StopWordsRemover)
followed by [CountVectorizer](http://spark.apache.org/docs/1.6.1/api/scala/index.html#org.apache.spark.ml.feature.CountVectorizer).
`RegexTokenizer` performs splitting/tokenization based on regular expression matching. To perform tokenization rather than splitting one neet to set parameter `gaps` to `false`.

`StopWordsRemover` comes with provided list of stop words. Alternatively one can provide its own stop words list.

```scala
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover, CountVectorizer}

val regexTokenizer = new RegexTokenizer()
  .setInputCol("review")
  .setOutputCol("tokens")
  .setPattern("(\\w+|[!?]|:-?\\)|:-?\\()")
  .setGaps(false)

val remover = new StopWordsRemover()
  .setInputCol("tokens")
  .setOutputCol("filteredTokens")

val countVec = new CountVectorizer()
  .setInputCol("filteredTokens")
  .setOutputCol("features")


// Chain tokenizer, stop words remover and CountVectorizer in a Pipeline
val pipeline = new Pipeline()
  .setStages(Array(regexTokenizer, 
                   remover, 
                   countVec))

val transformModel = pipeline.fit(unbalancedData)
```


><pre>
> import org.apache.spark.ml.Pipeline
> import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover, CountVectorizer}
> regexTokenizer: org.apache.spark.ml.feature.RegexTokenizer = regexTok_29b52045461b
> remover: org.apache.spark.ml.feature.StopWordsRemover = stopWords_65cbffca6cd9
> countVec: org.apache.spark.ml.feature.CountVectorizer = cntVec_b443f08a9fb4
> pipeline: org.apache.spark.ml.Pipeline = pipeline_e0f44a347840
> transformModel: org.apache.spark.ml.PipelineModel = pipeline_e0f44a347840
></pre>




```scala
val trainBagOfWords = transformModel.transform(trainingData).select("id", "sentiment", "features")
val testBagOfWords = transformModel.transform(testData).select("id", "sentiment", "features")

trainBagOfWords.limit(1).show
```


><pre>
> +-------+---------+--------------------+
> |     id|sentiment|            features|
> +-------+---------+--------------------+
> |10023_9|        1|(41325,[0,2,9,11,...|
> +-------+---------+--------------------+
> 
> trainBagOfWords: org.apache.spark.sql.DataFrame = [id: string, sentiment: int, features: vector]
> testBagOfWords: org.apache.spark.sql.DataFrame = [id: string, sentiment: int, features: vector]
></pre>




```scala
import org.apache.spark.mllib.linalg.SparseVector

val featureSpaceDim = trainBagOfWords.select("features").first.getAs[SparseVector](0).size

println(s"We've obtained $featureSpaceDim-dimensional feature space.")
```


><pre>
> We've obtained 41199-dimensional feature space.
> import org.apache.spark.mllib.linalg.SparseVector
> featureSpaceDim: Int = 41199
></pre>



Now after we've obtained some representation of our text,  the next step is to train the classification algorithms and to compare them with each other. This requires understanding what are the metrics should be used to compare algorithms. We can consider, for example, the following metrics:

- accuracy: $$ Accuracy = \frac{1}{l}\sum_{i=1}^l[y_i = \hat{y}_i]$$ where $y_i$ — the true object class $x_i$, $\hat{y}_i$ — he predicted class of the object.
- precision: $$Precision = \frac{TP}{TP + FP}$$
- recall: $$Recall = \frac{TP}{TP + FN}$$

where *TP*, *FP*, *FN* and *TN* — the elements of a confusion matrix:

| | y = 1 | y = 0 |
|------|------|
|   a(x) = 1  | TP| FP |
|   a(x) = 0  | FN | TN |

Please note that accuracy and recall are calculated relative to a fixed class.

Often, a classifier returns some *score* $b(x)$ of belonging to a given class, which is compared with fixed threshold *t*. Thus the classifier has the form $a(x) = [b(x) > t]$ and one can tune the threshold depending on specific needs. For example, there may be some cases where the threshold might need to be tuned so that it only predicts a class when the score is very high. Threshold tuning affects the quality of classification:
 - the higher $t$, the higher the precision, the lower the recall,
 - the lower $t$, the higher the recall, the lower the precesion.


### Precision-Recall curve
Interesting to know what will be the quality at all the different possible thresholds. So we can just compute precision-recall pairs for different thresholds. This will be precision-recall curve. We can achieve that using `BinaryClassificationMetrics` class from `org.apache.spark.mllib.evaluation` package. We can plot this curve with recall values on X-axis and precision values on Y-axis. This gives a good visualization of the quality of the algorithm.

### ROC curve
*ROC* curve is another method of visualizing the dependence of the quality of the algorithm from the threshold. In this case:
  - X-axis: $FPR = \frac{FP}{FP + TN}$
  - Y-axis: $TPR = \frac{TP}{TP + FN}$
 
Where *FPR* is false positive rate and *TPR* is true positive rate. Again `BinaryClassificationMetrics` provides appropriate method to compute this.

In addition, it is possible to measure the area under the curves: *auc_pr* and *auc_roc*, respectively.
**Problem** What are disadvantages of using `accuracy` metric in case of unbalanced data? Train Logistic Regression and Random Forest with 200 trees on bag-of-words and build `precision-recall` and `ROC` curves on test data. Also compute *auc_pr* and *auc_roc*. Compare training times of the algorithms. Is there a significant difference in the quality of algorithms? Which method seems less applicable in this problem and why?

```scala
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.feature.{StringIndexer, IndexToString}
import org.apache.spark.ml.tuning.TrainValidationSplit


val labelIndexer = new StringIndexer()
  .setInputCol("sentiment")
  .setOutputCol("label")
  .fit(unbalancedData)

// Convert predicted labels back to original labels.
val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedSentiment")
  .setLabels(labelIndexer.labels)

// Chain indexer, classifier and converter in a Pipeline
val lr = new LogisticRegression()
val lrPipeline = new Pipeline()
  .setStages(Array(labelIndexer, lr, labelConverter))

val rf = new RandomForestClassifier()
  .setNumTrees(200)
val rfPipeline = new Pipeline()
  .setStages(Array(labelIndexer, rf, labelConverter))
```


><pre>
> import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassifier}
> import org.apache.spark.ml.feature.{StringIndexer, IndexToString}
> import org.apache.spark.ml.tuning.TrainValidationSplit
> labelIndexer: org.apache.spark.ml.feature.StringIndexerModel = strIdx_f46f07b833ba
> labelConverter: org.apache.spark.ml.feature.IndexToString = idxToStr_b6143fe121fa
> lr: org.apache.spark.ml.classification.LogisticRegression = logreg_e7ba5c026307
> lrPipeline: org.apache.spark.ml.Pipeline = pipeline_8b278f8fe49b
> rf: org.apache.spark.ml.classification.RandomForestClassifier = rfc_7518313860ff
> rfPipeline: org.apache.spark.ml.Pipeline = pipeline_d5c9534ffb26
></pre>




```scala
// train classifier
val lrModel = lrPipeline.fit(trainBagOfWords)
```


><pre>
> lrModel: org.apache.spark.ml.PipelineModel = pipeline_f3fba53424ed
></pre>




```scala
val rfModel = rfPipeline.fit(trainBagOfWords)
```


><pre>
> rfModel: org.apache.spark.ml.PipelineModel = pipeline_d0cd3fc16bef
></pre>



It's clear what Random Forest takes much more time to train. 

```scala
// Make predictions.
val lrPredictions = lrModel.transform(testBagOfWords)
lrPredictions.select("sentiment",
                     "label",
                     "probability",
                     "rawPrediction",
                     "prediction",
                     "predictedSentiment")
             .sample(false, 10.0 / testBagOfWords.count)
             .show
```


><pre>
> +---------+-----+--------------------+--------------------+----------+------------------+
> |sentiment|label|         probability|       rawPrediction|prediction|predictedSentiment|
> +---------+-----+--------------------+--------------------+----------+------------------+
> |        1|  1.0|[0.04101322881915...|[-3.1519826116107...|       1.0|                 1|
> |        0|  0.0|[0.99999990882226...|[16.2104549579991...|       0.0|                 0|
> |        0|  0.0|[0.99999793464449...|[13.0902061254842...|       0.0|                 0|
> +---------+-----+--------------------+--------------------+----------+------------------+
> 
> lrPredictions: org.apache.spark.sql.DataFrame = [id: string, sentiment: int, features: vector, label: double, rawPrediction: vector, probability: vector, prediction: double, predictedSentiment: string]
></pre>




```scala
val rfPredictions = rfModel.transform(testBagOfWords)
rfPredictions.select("sentiment",
                     "label",
                     "probability",
                     "rawPrediction",
                     "prediction",
                     "predictedSentiment")
             .sample(false, 10.0 / testBagOfWords.count)
             .show
```


><pre>
> +---------+-----+--------------------+--------------------+----------+------------------+
> |sentiment|label|         probability|       rawPrediction|prediction|predictedSentiment|
> +---------+-----+--------------------+--------------------+----------+------------------+
> |        1|  1.0|[0.77030053251982...|[154.060106503964...|       0.0|                 0|
> |        0|  0.0|[0.80400207191763...|[160.800414383527...|       0.0|                 0|
> |        0|  0.0|[0.79261413809234...|[158.522827618468...|       0.0|                 0|
> |        0|  0.0|[0.78447662234151...|[156.895324468303...|       0.0|                 0|
> |        0|  0.0|[0.79438708663826...|[158.877417327653...|       0.0|                 0|
> |        0|  0.0|[0.78716768931477...|[157.433537862954...|       0.0|                 0|
> |        0|  0.0|[0.78493983704925...|[156.987967409850...|       0.0|                 0|
> |        0|  0.0|[0.78145003450549...|[156.290006901098...|       0.0|                 0|
> |        0|  0.0|[0.76871914160587...|[153.743828321175...|       0.0|                 0|
> |        0|  0.0|[0.80800667428922...|[161.601334857844...|       0.0|                 0|
> |        0|  0.0|[0.80289749495636...|[160.579498991272...|       0.0|                 0|
> +---------+-----+--------------------+--------------------+----------+------------------+
> 
> rfPredictions: org.apache.spark.sql.DataFrame = [id: string, sentiment: int, features: vector, label: double, rawPrediction: vector, probability: vector, prediction: double, predictedSentiment: string]
></pre>




```scala
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.DenseVector

// prepare labels and predictions for metric model
val lrPredictionAndLabels = lrPredictions
  .select("rawPrediction", "label")
  .map(r => (r.getAs[DenseVector](0)(1), r.getAs[Double](1)))

val rfPredictionAndLabels = rfPredictions
  .select("rawPrediction", "label")
  .map(r => (r.getAs[DenseVector](0)(1), r.getAs[Double](1)))

// Instantiate metrics object
val lrMetrics = new BinaryClassificationMetrics(lrPredictionAndLabels)
val rfMetrics = new BinaryClassificationMetrics(rfPredictionAndLabels)

// Obtain precision-recall curve
val lrPrecisionRecall = lrMetrics.pr
val rfPrecisionRecall = rfMetrics.pr

// Obtain roc curve
val lrROC = lrMetrics.roc
val rfROC = rfMetrics.roc
```


><pre>
> import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
> import org.apache.spark.mllib.linalg.DenseVector
> lrPredictionAndLabels: org.apache.spark.rdd.RDD[(Double, Double)] = MapPartitionsRDD[423] at map at <console>:114
> rfPredictionAndLabels: org.apache.spark.rdd.RDD[(Double, Double)] = MapPartitionsRDD[444] at map at <console>:118
> lrMetrics: org.apache.spark.mllib.evaluation.BinaryClassificationMetrics = org.apache.spark.mllib.evaluation.BinaryClassificationMetrics@6aafa0a4
> rfMetrics: org.apache.spark.mllib.evaluation.BinaryClassificationMetrics = org.apache.spark.mllib.evaluation.BinaryClassificationMetrics@1e67d414
> lrPrecisionRecall: org.apache.spark.rdd.RDD[(Double, Double)] = UnionRDD[455] at union at BinaryClassificationMetrics.scala:108
> rfPrecisionRecall: org.apache.s...
></pre>




```scala
case class RecallPrecisionPoint(lrRecall: Double,
                                lrPrecision: Double,
                                rfRecall: Double,
                                rfPrecision: Double)
val sampleFraction = 0.2
val recallPrecisionPoints = lrPrecisionRecall
                              .sample(false, sampleFraction)
                              .collect.zip(rfPrecisionRecall.sample(false, sampleFraction).collect)
                              .map{
  p => RecallPrecisionPoint(p._1._1, p._1._2, p._2._1, p._2._2)}

case class ROCPoint(lrFPR: Double,
                    lrTPR: Double,
                    rfFPR: Double,
                    rfTPR: Double)
val rocPoints = lrROC
                  .sample(false,0.2)
                  .collect.zip(rfROC.sample(false, sampleFraction).collect)
                  .map{
  p => ROCPoint(p._1._1, p._1._2, p._2._1, p._2._2)}
```


><pre>
> defined class RecallPrecisionPoint
> sampleFraction: Double = 0.2
> recallPrecisionPoints: Array[RecallPrecisionPoint] = Array(RecallPrecisionPoint(0.0,1.0,0.00663716814159292,1.0), RecallPrecisionPoint(0.015486725663716814,0.875,0.011061946902654867,1.0), RecallPrecisionPoint(0.024336283185840708,0.9166666666666666,0.01991150442477876,1.0), RecallPrecisionPoint(0.033185840707964605,0.8823529411764706,0.03761061946902655,1.0), RecallPrecisionPoint(0.046460176991150445,0.875,0.03761061946902655,0.9444444444444444), RecallPrecisionPoint(0.05309734513274336,0.8888888888888888,0.04424778761061947,0.9090909090909091), RecallPrecisionPoint(0.06415929203539823,0.90625,0.05309734513274336,0.9230769230769231), RecallPrecisionPoint(0.07964601769911504,0.9230769230769231,0.07964601769911504,0.9), Reca...
></pre>




```scala
CustomC3Chart(recallPrecisionPoints,
              """{ data: { xs: {
                            'lrPrecision': 'lrRecall',
                            'rfPrecision': 'rfRecall',
                         }
                   },
                   axis: {
                      y: {
                        label: 'precision'
                      },
                      x: {
                         label: 'recall',
                         tick: {
                            count: 5
                         }
                      }
                   },
                   point: {
                        show: false
                   }
                  }""")

```


><pre>
> res26: notebook.front.widgets.CustomC3Chart[Array[RecallPrecisionPoint]] = <CustomC3Chart widget>
></pre>




```scala
CustomC3Chart(rocPoints,
              """{ data: { xs: {
                            'lrTPR': 'lrFPR',
                            'rfTPR': 'rfFPR',
                         }
                   },
                   axis: {
                      y: {
                        label: 'TPR'
                      },
                      x: {
                         label: 'FPR',
                         tick: {
                            count: 5
                         }
                      }
                   },
                   point: {
                        show: false
                   }
                  }""")

```


><pre>
> res28: notebook.front.widgets.CustomC3Chart[Array[ROCPoint]] = <CustomC3Chart widget>
></pre>




```scala
println("Area under precision-recall lr curve = " + lrMetrics.areaUnderPR)
println("Area under precision-recall rf curve = " + rfMetrics.areaUnderPR)

println("Area under roc lr curve = " + lrMetrics.areaUnderROC)
println("Area under roc rf curve = " + rfMetrics.areaUnderROC)
```


><pre>
> Area under precision-recall lr curve = 0.6640000664694756
> Area under precision-recall rf curve = 0.7056397083564855
> Area under roc lr curve = 0.8686817959715503
> Area under roc rf curve = 0.8963488788750762
></pre>



It's easy to see what it's hard to achieve good precision with high recall related to positive reviews. What's because of low fraction of positive reviews in our unbalanced dataset. The quality of classification would be different if we will calculate metrics relating to negative reviews.

**Problem**. Find the maximum accuracy of each classifier at level of recall of at least 0.8:
  - while predicting positive reviews
  - while predicting negative reviews

```scala
// measuring predictions of positive reviews
val lrPosPredictionAndLabels = lrPredictions
  .select("rawPrediction", "label")
  .map(r => (r.getAs[DenseVector](0)(1), r.getAs[Double](1)))
val rfPosPredictionAndLabels = rfPredictions
  .select("rawPrediction", "label")
  .map(r => (r.getAs[DenseVector](0)(1), r.getAs[Double](1)))
val lrPosMetrics = new BinaryClassificationMetrics(lrPosPredictionAndLabels)
val rfPosMetrics = new BinaryClassificationMetrics(rfPosPredictionAndLabels)

// measuring predictions of negative reviews
val lrNegPredictionAndLabels = lrPredictions
  .select("rawPrediction", "label")
  .map(r => (r.getAs[DenseVector](0)(0), 1.0 - r.getAs[Double](1)))
val rfNegPredictionAndLabels = rfPredictions
  .select("rawPrediction", "label")
  .map(r => (r.getAs[DenseVector](0)(0), 1.0 - r.getAs[Double](1)))
val lrNegMetrics = new BinaryClassificationMetrics(lrNegPredictionAndLabels)
val rfNegMetrics = new BinaryClassificationMetrics(rfNegPredictionAndLabels)


// Obtain precision-recall curves
val lrPosPrecisionRecall = lrPosMetrics.pr
val rfPosPrecisionRecall = rfPosMetrics.pr
val lrNegPrecisionRecall = lrNegMetrics.pr
val rfNegPrecisionRecall = rfNegMetrics.pr
```


><pre>
> lrPosPredictionAndLabels: org.apache.spark.rdd.RDD[(Double, Double)] = MapPartitionsRDD[521] at map at <console>:113
> rfPosPredictionAndLabels: org.apache.spark.rdd.RDD[(Double, Double)] = MapPartitionsRDD[542] at map at <console>:116
> lrPosMetrics: org.apache.spark.mllib.evaluation.BinaryClassificationMetrics = org.apache.spark.mllib.evaluation.BinaryClassificationMetrics@2bc74f0e
> rfPosMetrics: org.apache.spark.mllib.evaluation.BinaryClassificationMetrics = org.apache.spark.mllib.evaluation.BinaryClassificationMetrics@21506af4
> lrNegPredictionAndLabels: org.apache.spark.rdd.RDD[(Double, Double)] = MapPartitionsRDD[563] at map at <console>:123
> rfNegPredictionAndLabels: org.apache.spark.rdd.RDD[(Double, Double)] = MapPartitionsRDD[584] at map at <console>:126
> lrNegMetrics: org.apache.spark....
></pre>




```scala
println("=== Positive reviews prediction ===")
println("Logistic regression max accuracy at recall >= 0.8: " + 
        lrPosMetrics.pr.filter(_._1 >= 0.8).map(_._2).max)
println("Random Forest max accuracy at recall >= 0.8: " + 
        rfPosMetrics.pr.filter(_._1 >= 0.8).map(_._2).max)

println("=== Negative reviews prediction ===")
println("Logistic regression max accuracy at recall >= 0.8: " + 
        lrNegMetrics.pr.filter(_._1 >= 0.8).map(_._2).max)
println("Random Forest max accuracy at recall >= 0.8: " + 
        rfNegMetrics.pr.filter(_._1 >= 0.8).map(_._2).max)
```


><pre>
> === Positive reviews prediction ===
> Logistic regression max accuracy at recall >= 0.8: 0.5370919881305638
> Random Forest max accuracy at recall >= 0.8: 0.5647425897035881
> === Negative reviews prediction ===
> Logistic regression max accuracy at recall >= 0.8: 0.9388489208633094
> Random Forest max accuracy at recall >= 0.8: 0.9525200876552228
></pre>



Also we can conclude what complex, slow and heavy Random Forest with 200 decision trees doesn't perform much better then easy and fast Logistic Regression. That's because decision tree based models don't suite well for *sparse features* (bag-of-words is an example of sparse features).
Up to this point we have not performed hyperparameters tuning. From now let's use only LogisticRegression model and perform cross-validation or train-validation split (it is less expensive, but will not produce as reliable results) to find optimal `regularization parameter (regParam)` for Logistic Regression with respect to 'roc_auc' metric. Also searching for optimal value of regularization parameter on a logarithmic scale is good idea.
## Feature selection and dimensionality reduction
At this stage it can be concluded that the proposed text encoding may not be the best. Not every algorithm can be applied in this problem due to the large feature space. In addition, there is a lot of noise in our encoded data, because all the words have been taken to build the vocabulary, i.e., were taken even those words which were found only in a single review (think about typos). So it seems that it would be nice to reduce the dimensionality of the data and to get rid of the noise. One can perform feature selection and dimensionality reduction in multiple ways.

### Term frequency
Try to create a sample that will consist of only the most "important" words. It seems that the occurrence of the most frequent words in the review, for example, *good*, *bad*, etc. are quite good indicators. This can be done by discarding the rare words by frequency. One can specify the minimum number (or fraction) of different reviews a word must appear in to be included in the vocabulary by setting `minDF` parameter of `CountVectorizer`.

### Feature Importance
Use trained random forest to obtain its importance estimation for each feature and select most important features (words) using this estimations.

### Hashing trick

A different approach from the above two is [hashing](https://en.wikipedia.org/wiki/Feature_hashing) or hashing trick: get the hash of each word and after that, for example, perform bag-of-words over the space of obtained hashes. This allows you to tune the size of the feature space: the lower the sapce, the higher the frequency of collisions. Also it allows to handle previously unseen words. This approach is implemented in [HashingTF](http://spark.apache.org/docs/1.6.1/api/scala/index.html#org.apache.spark.ml.feature.HashingTF) `Transformer`.

You may notice that the last two approaches can be applied not only to textual data.
**Problem** Create new features for our unbalanced data as follows:
  - bag-of-words with `minDF` is equal to 4
  - 15000 most important feature (according to trained random forest) from bag-of-words with `minDF` is equal to 1
  - hashing trick with `numFeatures` is equal to 15000
  
Train classifier and calculate the area under ROC curve for each of four samples (default bag-of-words and three new samples described above). What can you say about the quality of these approaches to reduce feature space dimension?

```scala
// Term frequency
val tfCountVec = new CountVectorizer()
  .setInputCol("filteredTokens")
  .setOutputCol("features")
  .setMinDF(4)

val tfPipeline = new Pipeline()
  .setStages(Array(regexTokenizer, 
                   remover, 
                   tfCountVec))

val tfModel = tfPipeline.fit(unbalancedData)
```


><pre>
> tfCountVec: org.apache.spark.ml.feature.CountVectorizer = cntVec_8a85399435dd
> tfPipeline: org.apache.spark.ml.Pipeline = pipeline_a9f66c8956ec
> tfModel: org.apache.spark.ml.PipelineModel = pipeline_a9f66c8956ec
></pre>




```scala
import org.apache.spark.ml.feature.VectorSlicer

// Most important features
val trainedRF = rf.fit(labelIndexer.transform(trainBagOfWords))
val mostImportant =  trainedRF.featureImportances
  .toArray
  .zipWithIndex
  .sortBy(- _._1)
  .take(15000)
  .map(_._2)

val rfCountVec = new CountVectorizer()
  .setInputCol("filteredTokens")
  .setOutputCol("allFeatures")

val mostImportantSelector = new VectorSlicer()
  .setInputCol(rfCountVec.getOutputCol)
  .setOutputCol("features")
  .setIndices(mostImportant)

val mostImpPipeline = new Pipeline()
  .setStages(Array(regexTokenizer, 
                   remover, 
                   rfCountVec,
                   mostImportantSelector
                  ))

val mostImpModel = mostImpPipeline.fit(unbalancedData)
```


><pre>
> import org.apache.spark.ml.feature.VectorSlicer
> trainedRF: org.apache.spark.ml.classification.RandomForestClassificationModel = RandomForestClassificationModel (uid=rfc_935390080402) with 200 trees
> mostImportant: Array[Int] = Array(43, 303, 23, 96, 17, 8, 34, 355, 19, 573, 110, 349, 49, 385, 10, 237, 1696, 94, 2128, 70, 4584, 7, 357, 174, 104, 93, 547, 28, 131, 196, 60, 1005, 1838, 483, 84, 132, 383, 1315, 459, 1220, 966, 3305, 14, 296, 116, 893, 71, 387, 4698, 1346, 679, 938, 40, 41, 1559, 718, 1713, 294, 2, 1121, 511, 211, 1869, 1139, 180, 5287, 1626, 729, 283, 1590, 817, 1817, 327, 1283, 1622, 455, 120, 269, 2614, 372, 64, 5025, 143, 7727, 399, 4300, 191, 631, 3529, 839, 11398, 556, 233, 38, 610, 7511, 3146, 1149, 2615, 691, 278, 299, 8671, 3652, 157, 3499, 1081, 3023, 1129, 5985, 89...
></pre>




```scala
import org.apache.spark.ml.feature.HashingTF

// Hashing trick
val hashingTF = new HashingTF()
  .setInputCol(regexTokenizer.getOutputCol)
  .setOutputCol("features")
  .setNumFeatures(15000)

val hashingTFPipeline = new Pipeline()
  .setStages(Array(regexTokenizer, 
                   hashingTF
                  ))
val hashingTFModel = hashingTFPipeline.fit(unbalancedData)
```


><pre>
> import org.apache.spark.ml.feature.HashingTF
> hashingTF: org.apache.spark.ml.feature.HashingTF = hashingTF_fc100dad0308
> hashingTFPipeline: org.apache.spark.ml.Pipeline = pipeline_00a717af6736
> hashingTFModel: org.apache.spark.ml.PipelineModel = pipeline_00a717af6736
></pre>




```scala
// trainBagOfWords and testBagOfWords are obtained previously

val trainTermFreqFeatures = tfModel.transform(trainingData).select("id", "sentiment", "features")
val testTermFreqFeatures = tfModel.transform(testData).select("id", "sentiment", "features")

val trainMostImpFeatures = mostImpModel.transform(trainingData).select("id", "sentiment", "features")
val testMostImpFeatures = mostImpModel.transform(testData).select("id", "sentiment", "features")

val trainHashingTFFeatures = hashingTFModel.transform(trainingData).select("id", "sentiment", "features")
val testHashingTFFeatures = hashingTFModel.transform(testData).select("id", "sentiment", "features")
```


><pre>
> trainTermFreqFeatures: org.apache.spark.sql.DataFrame = [id: string, sentiment: int, features: vector]
> testTermFreqFeatures: org.apache.spark.sql.DataFrame = [id: string, sentiment: int, features: vector]
> trainMostImpFeatures: org.apache.spark.sql.DataFrame = [id: string, sentiment: int, features: vector]
> testMostImpFeatures: org.apache.spark.sql.DataFrame = [id: string, sentiment: int, features: vector]
> trainHashingTFFeatures: org.apache.spark.sql.DataFrame = [id: string, sentiment: int, features: vector]
> testHashingTFFeatures: org.apache.spark.sql.DataFrame = [id: string, sentiment: int, features: vector]
></pre>




```scala
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
// CrossValidator is another option
// it produces more reliable results but it's more expensive to compute.

val lrParamGrid = new ParamGridBuilder()
  .addGrid(lr.regParam, Array(1e-3, 1e-2, 1e-1, 1e0))
  .build()

val lrValidator = new TrainValidationSplit()
  .setEstimator(lrPipeline)
  .setEvaluator(new BinaryClassificationEvaluator)
  .setEstimatorParamMaps(lrParamGrid)
  .setTrainRatio(0.7)
```


><pre>
> import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
> import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
> lrParamGrid: Array[org.apache.spark.ml.param.ParamMap] = 
> Array({
> 	logreg_e7ba5c026307-regParam: 0.001
> }, {
> 	logreg_e7ba5c026307-regParam: 0.01
> }, {
> 	logreg_e7ba5c026307-regParam: 0.1
> }, {
> 	logreg_e7ba5c026307-regParam: 1.0
> })
> lrValidator: org.apache.spark.ml.tuning.TrainValidationSplit = tvs_3bc192dcd700
></pre>




```scala
val bagOfWordsModel = lrValidator.fit(trainBagOfWords)
```


><pre>
> bagOfWordsModel: org.apache.spark.ml.tuning.TrainValidationSplitModel = tvs_3bc192dcd700
></pre>




```scala
val tfModel = lrValidator.fit(trainTermFreqFeatures)
```


><pre>
> tfModel: org.apache.spark.ml.tuning.TrainValidationSplitModel = tvs_3bc192dcd700
></pre>




```scala
val mostImpModel = lrValidator.fit(trainMostImpFeatures)
```


><pre>
> mostImpModel: org.apache.spark.ml.tuning.TrainValidationSplitModel = tvs_3bc192dcd700
></pre>




```scala
val hashingTFModel = lrValidator.fit(trainHashingTFFeatures)
```


><pre>
> hashingTFModel: org.apache.spark.ml.tuning.TrainValidationSplitModel = tvs_3bc192dcd700
></pre>




```scala
val predictions = bagOfWordsModel.transform(testBagOfWords)
  .select("rawPrediction", "label")
  .map(r => (r.getAs[DenseVector](0)(1), r.getAs[Double](1)))

val bagOfWordsMetrics = new BinaryClassificationMetrics(predictions)
```


><pre>
> predictions: org.apache.spark.rdd.RDD[(Double, Double)] = MapPartitionsRDD[1838] at map at <console>:114
> bagOfWordsMetrics: org.apache.spark.mllib.evaluation.BinaryClassificationMetrics = org.apache.spark.mllib.evaluation.BinaryClassificationMetrics@375b5f7b
></pre>




```scala
val predictions = tfModel.transform(testTermFreqFeatures)
  .select("rawPrediction", "label")
  .map(r => (r.getAs[DenseVector](0)(1), r.getAs[Double](1)))

val tfMetrics = new BinaryClassificationMetrics(predictions)
```


><pre>
> predictions: org.apache.spark.rdd.RDD[(Double, Double)] = MapPartitionsRDD[1859] at map at <console>:128
> tfMetrics: org.apache.spark.mllib.evaluation.BinaryClassificationMetrics = org.apache.spark.mllib.evaluation.BinaryClassificationMetrics@5f844ad7
></pre>




```scala
val predictions = mostImpModel.transform(testMostImpFeatures)
  .select("rawPrediction", "label")
  .map(r => (r.getAs[DenseVector](0)(1), r.getAs[Double](1)))

val mostImpMetrics = new BinaryClassificationMetrics(predictions)
```


><pre>
> predictions: org.apache.spark.rdd.RDD[(Double, Double)] = MapPartitionsRDD[1880] at map at <console>:144
> mostImpMetrics: org.apache.spark.mllib.evaluation.BinaryClassificationMetrics = org.apache.spark.mllib.evaluation.BinaryClassificationMetrics@42972bfd
></pre>




```scala
// prepare predictions for metric model

val predictions = hashingTFModel.transform(testHashingTFFeatures)
  .select("rawPrediction", "label")
  .map(r => (r.getAs[DenseVector](0)(1), r.getAs[Double](1)))

val hashingTFMetrics = new BinaryClassificationMetrics(predictions)
```


><pre>
> predictions: org.apache.spark.rdd.RDD[(Double, Double)] = MapPartitionsRDD[1901] at map at <console>:146
> hashingTFMetrics: org.apache.spark.mllib.evaluation.BinaryClassificationMetrics = org.apache.spark.mllib.evaluation.BinaryClassificationMetrics@48b4db44
></pre>




```scala
println("Area under roc curve")
println("Simple bag-of-words: " + bagOfWordsMetrics.areaUnderROC)
println("Term Frequency: " + tfMetrics.areaUnderROC)
println("Most important features: " + mostImpMetrics.areaUnderROC)
println("Hahing trick: " + hashingTFMetrics.areaUnderROC)
```


><pre>
> Area under roc curve
> Simple bag-of-words: 0.8981623234085643
> Term Frequency: 0.9084144762762402
> Most important features: 0.9013465976248115
> Hahing trick: 0.8982172004277599
></pre>



Thus, we managed to reduce the dimension of more than three times without much loss in quality. But even 15000 features is quite a lot. Let's say we want to reduce the dimension to 2000 features, however, you notice that three of the previous method gave a small quality degradation.

Also one of the ways of dimensionality reduction is [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) (principal component analysis). PCA uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components. Generaly this operation can efficiently perform dimentionality reduction but requires a lot of cpu and memory resources. So if you have sufficient resources, you can also try to solve the following problem:

**Problem** Try to reduce the dimension of up to 2000:
  - using one of above methods
  - using PCA (for usage example look [here](http://spark.apache.org/docs/1.6.1/ml-features.html#pca))

Which approach works best?
### What are some other ways of processing text data?

As you can see, the approach with a bag of words is very naive, because it does not allow to take into account the information about word frequency across all documents (reviews). In this case, it may be useful to use [tf-idf](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).

Another disadvantage of the bag-of-words is using of absolute frequencies of words. Some words can have very large frequencies and at the same time some other words can have very low frequencies. To "smooth out" the difference between them one can apply a log transformation $x \to log(x + 1)$ (we need to add `1` because `X` can be equal to `0` in bag-of-words encoding).

All of the above certainly is not comprehensive toolbox in text processing, but somehow is the basis for the further diving into this area.
