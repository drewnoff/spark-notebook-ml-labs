## Data Analysis Toolbox
In this lab we are going to get familiar with **Breeze** numerical processing library, Spark **DataFrames** (distributed collections of data organized into named columns) and **C3 Charts** library in a way of solving little challenges. At the beginning of each section are reference materials necessary for solving the problems.
### Breeze
* [Quick start tutorial](https://github.com/scalanlp/breeze/wiki/Quickstart)

```scala
import breeze.linalg._
import breeze.stats.{mean, stddev}
import breeze.stats.distributions._
```


><pre>
> import breeze.linalg._
> import breeze.stats.{mean, stddev}
> import breeze.stats.distributions._
></pre>



** Problem 1.** Implement a method that takes Matrix X and two sequences ii and jj of equal size as an input and produces breeze.linalg.DenseVector[Double] of elements [X[ii[0], jj[0]], X[ii[1], jj[1]], ..., X[ii[N-1], jj[N-1]]].

```scala
def constructVector(X: Matrix[Double], ii: Seq[Int], jj: Seq[Int]): DenseVector[Double] = ???
```


><pre>
> constructVector: (X: breeze.linalg.Matrix[Double], ii: Seq[Int], jj: Seq[Int])breeze.linalg.DenseVector[Double]
></pre>




```scala
// Solution for problem 1
def constructVector(X: Matrix[Double], ii: Seq[Int], jj: Seq[Int]): DenseVector[Double] =
  DenseVector(ii.zip(jj).map(ix => X(ix._1, ix._2)).toArray)

constructVector(DenseMatrix((1.0,2.0,3.0), 
                            (4.0,5.0,6.0), 
                            (7.0, 8.0, 9.0)), 
                List(0, 1, 2), List(0, 1, 2))
```


><pre>
> constructVector: (X: breeze.linalg.Matrix[Double], ii: Seq[Int], jj: Seq[Int])breeze.linalg.DenseVector[Double]
> res4: breeze.linalg.DenseVector[Double] = DenseVector(1.0, 5.0, 9.0)
></pre>

> DenseVector(1.0, 5.0, 9.0)

** Problem 2. ** Write a method to calculate the product of nonzero elements on the diagonal of a rectangular matrix. For example, for X = Matrix((1.0, 0.0, 1.0), (2.0, 0.0, 2.0), (3.0, 0.0, 3.0), (4.0, 4.0, 4.0)) the answer is Some(3). If there are no nonzero elements, the method should return None.

```scala
def nonzeroProduct(X: Matrix[Double]): Option[Double] = ???
```


><pre>
> nonzeroProduct: (X: breeze.linalg.Matrix[Double])Option[Double]
></pre>




```scala
// Solution for problem 2
def nonzeroProduct(X: Matrix[Double]): Option[Double] =
  (0 until min(X.rows, X.cols)).map(i => X(i, i)).filter(_ != 0) match {
  case Seq() => None
  case xs => Some(xs.reduce(_ * _))
}

nonzeroProduct(Matrix((1.0, 0.0, 1.0), (2.0, 0.0, 2.0), (3.0, 0.0, 3.0), (4.0, 4.0, 4.0)))
```


><pre>
> nonzeroProduct: (X: breeze.linalg.Matrix[Double])Option[Double]
> res7: Option[Double] = Some(3.0)
></pre>

> Some(3.0)

** Problem 3. ** Write a method to find the maximum element of the vector with the preceding zero element. For example, for Vector(6, 2, 0, 3, 0, 0, 5, 7, 0) the answer is Some(5). If there are no such an elements, the method should return None.

```scala
def maxAfterZeroElement(vec: Vector[Double]): Option[Double] = ???
```


><pre>
> maxAfterZeroElement: (vec: breeze.linalg.Vector[Double])Option[Double]
></pre>




```scala
def maxAfterZeroElement(vec: Vector[Double]): Option[Double] =
  vec.toArray.foldLeft((None, false): (Option[Double], Boolean))(
    (prev: (Option[Double], Boolean), el: Double) =>
    if (el == 0) {
      (prev._1, true)
    } else {
      prev match {
        case (p, false) => (p, false)
        case (None, true) => (Some(el), false)
        case (Some(m), true) => ({if (el > m) Some(el) else Some(m)}, false)
      }
    }
  )._1
```


><pre>
> maxAfterZeroElement: (vec: breeze.linalg.Vector[Double])Option[Double]
></pre>



** Problem 4. ** Write a method that takes Matrix X and some number Double v and returns closest matrix element to given number v. For example: for X = new DenseMatrix(2, 5, DenseVector.range(0, 10).mapValues(_.toDouble).toArray) and v = 3.6 the answer would be 4.0.

```scala
def closestValue(X: DenseMatrix[Double], v: Double): Double = ???
```


><pre>
> closestValue: (X: breeze.linalg.DenseMatrix[Double], v: Double)Double
></pre>




```scala
// Solution for problem 4
import scala.math.abs

def closestValue(X: DenseMatrix[Double], v: Double): Double =
  X(argmin(X.map(e => abs(e - v))))
```


><pre>
> import scala.math.abs
> closestValue: (X: breeze.linalg.DenseMatrix[Double], v: Double)Double
></pre>




```scala
// Another solution for problem 4
import breeze.numerics.abs

def closestValue(X: DenseMatrix[Double], v: Double): Double =
  X(argmin(abs(X - v)))
```


><pre>
> import breeze.numerics.abs
> closestValue: (X: breeze.linalg.DenseMatrix[Double], v: Double)Double
></pre>



** Problem 5. ** Write a method that takes Matrix X and scales each column of this matrix by subtracting mean value and dividing by standard deviation of the column. For testing one can generate random matrix. Avoid division by zero.

```scala
def scale(X: DenseMatrix[Double]): Unit = ???
```


><pre>
> scale: (X: breeze.linalg.DenseMatrix[Double])Unit
></pre>




```scala
// Solution for problem 5
def scale(X: DenseMatrix[Double]): Unit = {
  val mm = mean(X(::, *))    // using broadcasting
  val std = stddev(X(::, *)) // https://github.com/scalanlp/breeze/wiki/Quickstart#broadcasting
  (0 until X.cols).foreach{i =>
    if (std(0, i) == 0.0) {
      X(::, i) := 0.0
    } else {
      X(::, i) := (X(::, i) - mm(0, i)) :/ std(0, i)
    }
  }
}
```


><pre>
> scale: (X: breeze.linalg.DenseMatrix[Double])Unit
></pre>




```scala
// Another solution for problem 5
def scale(X: DenseMatrix[Double]): Unit =
  (0 until X.cols).map{i =>
    val col = X(::, i)
    val std = stddev(col)
    if (std != 0.0) {
      X(::, i) := (col - mean(col)) / std
    } else {
      X(::, i) := DenseVector.zeros[Double](col.size)
    }
  }
```


><pre>
> scale: (X: breeze.linalg.DenseMatrix[Double])Unit
></pre>




```scala
// Let's test our scale method on random data
val nd = new Gaussian(12, 20)
val m = DenseMatrix.rand(10, 3, nd)
println(m)
println("============")
scale(m)
println(m)
```


><pre>
> 15.590452840444563  26.751701453651677   -3.87442957211206    
> 20.327157147052404  4.872835405186789    -1.723076564770194   
> 8.623837647458954   -12.515032706820008  17.23652514034355    
> -22.6959606971933   -3.5252869052855402  -28.569802562830404  
> 5.084148521366598   6.537587281421278    1.27947368109675     
> 45.550604542120766  33.63584014298664    14.398835562651708   
> 28.39067989774948   21.884251067827837   26.21188242480804    
> 35.760270426060366  33.15913097645061    43.652905311745315   
> -6.957271573704126  30.631777233387844   4.858850308567796    
> 32.17744687777203   8.983683803901943    4.909365750891229    
> ============
> -0.02858428109928919  0.714489638531793    -0.6056134391326071   
> 0.1990918470152323    -0.6204508202172598  -0.4943741445319128   
> -0.36344410568028807  -1.6813727428933674  0.48596367654601474   
> -1.8688727663822855   -1.132862808340346   -1.882528878864535    
> -0.5335840727948753   -0.5188758744023809  -0.3391222773517981   
> 1.411491116397139     1.1345258158947291   0.33923620511713787   
> 0.5866760236928795    0.41750183136681246  0.9500494912429874    
> 0.9409054052767336    1.1054393745747901   1.8518666623146085    
> -1.1123712899791023   0.9512327188133928   -0.1540446402595702   
> 0.7686921235538567    -0.3696271333281644  -0.15143265508032536  
> nd: breeze.stats.distributions.Gaussian = Gaussian(12.0, 20.0)
> m: breeze.linalg.DenseMatrix[Double] = 
> -0.02858428109928919  0.714489638531793    -0.6056134391326071   
> 0.1990918470152323    -0.6204508202172598  -0.4943741445319128   
> -0.36344410568028807  -1.6813727428933674  0.48596367654601474   
> -1.8688727663822855   -1.132862808340346   -1.882528878864535    
> -0.5335840727948753   -0.5188758744023809  -0.3391222773517981   
> 1.411491116397139     1.1345258158947291   0.33923620511713787   
> 0.5866760236928795    0.41750183136681246  0.9500494912429874    
> 0.9409054052767336    1.1054393745747901   1.8518666623146085    
> -1.1123712899791023   0.9512327188133928   -0.1540446402595702   
> 0.7686921235538567    -0.3696271333281644  -0.15143265508032536  
></pre>



** Problem 6. ** Implement a method that for given matrix X finds:
* the determinant
* the trace
* max and min elements
* Frobenius Norm
* eigenvalues
* inverse matrix

For testing one can generate random matrix from normal distribution $N(10, 1)$.

```scala
def getStats(X: Matrix[Double]): Unit = ???
```


><pre>
> getStats: (X: breeze.linalg.Matrix[Double])Unit
></pre>




```scala
// Solution for problem 6
def getStats(X: DenseMatrix[Double]): String = {
  val dt = det(X)
  val tr = trace(X)
  val minE = min(X)
  val maxE = max(X)
  val frob = breeze.linalg.norm(X.toDenseVector)
  val ev = eig(X).eigenvalues
  val invM = inv(X)
  
  s"""Stats:
determinant: $dt
trace: $tr
min element: $minE
max element: $maxE
Frobenius Norm: $frob
eigenvalues: $ev
inverse matrix:\n$invM""".stripMargin 
}
```


><pre>
> getStats: (X: breeze.linalg.DenseMatrix[Double])String
></pre>




```scala
// Let's test our scale method on random data
val nd = new Gaussian(10, 1)
val X = DenseMatrix.rand(4, 4, nd)
```


><pre>
> nd: breeze.stats.distributions.Gaussian = Gaussian(10.0, 1.0)
> X: breeze.linalg.DenseMatrix[Double] = 
> 10.15867550081024   10.713391519035639  10.18898336794234   11.633517053992334  
> 9.077895190590993   10.687077605375258  9.75691251834008    10.289451974113568  
> 12.419948133142773  8.799359381094582   12.333412584337028  9.616047767507087   
> 9.018762639197664   11.122058811926983  9.603119538562519   10.441697550864596  
></pre>




```scala
println(getStats(X))
```


><pre>
> Stats:
> determinant: -14.64894396592202
> trace: 43.62086324138712
> min element: 8.799359381094582
> max element: 12.419948133142773
> Frobenius Norm: 41.681818838737364
> eigenvalues: DenseVector(41.461632636433905, 1.182643130384728, 1.182643130384728, -0.20605565581625584)
> inverse matrix:
> 0.37634342430946144  -4.699111409373191   0.45067158561397047   3.796260506021671    
> -0.3874775018168392  -1.7712409918032728  0.09247520887399419   2.0919567065524114   
> -0.6039672881460412  4.807753751137877    -0.20653804947039545  -3.8745431604779714  
> 0.6431296809327107   1.523754431265583    -0.297806479947489    -1.8480459447576396  
></pre>



### DataFrames
* https://databricks.com/blog/2015/02/17/introducing-dataframes-in-spark-for-large-scale-data-science.html
* http://spark.apache.org/docs/latest/sql-programming-guide.html

In this lab we will be using [data](https://www.kaggle.com/c/titanic/download/train.csv) from [Titanic dataset](https://www.kaggle.com/c/titanic/data).
To load data from csv file direct to Spark's Dataframe we will use [spark-csv](http://spark-packages.org/package/databricks/spark-csv) package.
To add spark-csv package to spark notebook one could add "com.databricks:spark-csv_2.10:1.4.0" (or "com.databricks:spark-csv_2.11:1.4.0" for Scala 2.11) dependency into customDeps conf section. Alternatively one could specify this dependency in `--packages` command line option while submiting spark application to a cluster (`spark-submit`) or launching spark shell (`spark-shell`). 

```scala
import org.apache.spark.sql.SQLContext
```


><pre>
> import org.apache.spark.sql.SQLContext
></pre>




```scala
val sqlContext = new SQLContext(sc)

val df = sqlContext.read
    .format("com.databricks.spark.csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load("notebooks/labs/DataAnalysisToolbox/titanic.csv")
```


><pre>
> sqlContext: org.apache.spark.sql.SQLContext = org.apache.spark.sql.SQLContext@31b5f894
> df: org.apache.spark.sql.DataFrame = [PassengerId: int, Survived: int, Pclass: int, Name: string, Sex: string, Age: double, SibSp: int, Parch: int, Ticket: string, Fare: double, Cabin: string, Embarked: string]
></pre>




```scala
// df.show()
df.limit(5)
```


><pre>
> res26: org.apache.spark.sql.DataFrame = [PassengerId: int, Survived: int, Pclass: int, Name: string, Sex: string, Age: double, SibSp: int, Parch: int, Ticket: string, Fare: double, Cabin: string, Embarked: string]
></pre>



**Problem 1.** Describe given dataset by answering following questions. How many women and men were on board? How many passengers were in each class? What is the average/minimum/maximum age of passengers? What can you say about the number of the surviving passengers?

```scala
// Solution for problem 1
import org.apache.spark.sql.functions.{min, max, mean}

df.groupBy("Sex").count().show()
df.groupBy("Pclass").count().show()
df.select(mean("Age").alias("Average Age"), min("Age"), max("Age")).show()

val totalPassengers = df.count()
val survived = df.groupBy("Survived").count()
survived.withColumn("%", (survived("count") / totalPassengers) * 100).show()
```


><pre>
> +------+-----+
> |   Sex|count|
> +------+-----+
> |female|  314|
> |  male|  577|
> +------+-----+
> 
> +------+-----+
> |Pclass|count|
> +------+-----+
> |     1|  216|
> |     2|  184|
> |     3|  491|
> +------+-----+
> 
> +-----------------+--------+--------+
> |      Average Age|min(Age)|max(Age)|
> +-----------------+--------+--------+
> |29.69911764705882|    0.42|    80.0|
> +-----------------+--------+--------+
> 
> +--------+-----+-----------------+
> |Survived|count|                %|
> +--------+-----+-----------------+
> |       0|  549|61.61616161616161|
> |       1|  342|38.38383838383838|
> +--------+-----+-----------------+
> 
> import org.apache.spark.sql.functions.{min, max, mean}
> totalPassengers: Long = 891
> survived: org.apache.spark.sql.DataFrame = [Survived: int, count: bigint]
></pre>



**Problem 2.** Is it true that women were more likely to survive than men? Who had more chances to survive: the passenger with a cheap ticket or the passenger with an expensive one? Is that true that youngest passengers had more chances to survive?

```scala
import org.apache.spark.sql.functions.{sum, count}
import org.apache.spark.sql.types.IntegerType
```


><pre>
> import org.apache.spark.sql.functions.{sum, count}
> import org.apache.spark.sql.types.IntegerType
></pre>




```scala
// Answer for q1
df.groupBy("Sex")
       .agg((sum("Survived") / count("Survived"))
       .alias("survived part"))
.show()
```


><pre>
> +------+-------------------+
> |   Sex|      survived part|
> +------+-------------------+
> |female| 0.7420382165605095|
> |  male|0.18890814558058924|
> +------+-------------------+
></pre>



Women were more likely to survive.

```scala
// Answer for q2
val survivedByFareRange = df.select(df("Survived"), 
                                  ((df("Fare") / (df("SibSp") + df("Parch") + 1) / 5).cast(IntegerType)
                                  ).alias("fareRange"))

survivedByFareRange.groupBy("fareRange")
                   .agg((sum("Survived") / count("Survived")).alias("Survived part"),
                      count("Survived").alias("passengers num"))
.sort("fareRange")
.show()
```


><pre>
> +---------+-------------------+--------------+
> |fareRange|      Survived part|passengers num|
> +---------+-------------------+--------------+
> |        0|0.26744186046511625|            86|
> |        1|0.27058823529411763|           425|
> |        2| 0.4122137404580153|           131|
> |        3| 0.5652173913043478|            23|
> |        4| 0.2222222222222222|             9|
> |        5| 0.5714285714285714|            70|
> |        6|             0.5625|            32|
> |        7|               0.56|            25|
> |        8|                0.6|            15|
> |        9|               0.75|             8|
> |       10| 0.4166666666666667|            12|
> |       11|                0.8|            10|
> |       13|                1.0|             3|
> |       14|               0.25|             4|
> |       15| 0.6666666666666666|             9|
> |       16|                1.0|             3|
> |       17|                1.0|             3|
> |       18|                1.0|             1|
> |       21|                1.0|             3|
> |       22|                1.0|             2|
> +---------+-------------------+--------------+
> only showing top 20 rows
> 
> survivedByFareRange: org.apache.spark.sql.DataFrame = [Survived: int, fareRange: int]
></pre>



We can see that passengers with cheapest tickets had lowest chances to survive. To obtain ticket cost per passenger we had to divide ticket fare by number of persons (one person itself + number of Siblings/Spouses aboard + number of parents/children aboard) included in fare.

```scala
// Answer for q3
val survivedByAgeDecade = df.select(df("Survived"), 
                                    ((df("Age") / 10).cast(IntegerType)).alias("decade"))
survivedByAgeDecade.filter(survivedByAgeDecade("decade").isNotNull).
                groupBy("decade")
                .agg((sum("Survived") / count("Survived")).alias("Survived part"),
                      count("Survived").alias("passengers num"))
.sort("decade")
.show()
```


><pre>
> +------+-------------------+--------------+
> |decade|      Survived part|passengers num|
> +------+-------------------+--------------+
> |     0| 0.6129032258064516|            62|
> |     1| 0.4019607843137255|           102|
> |     2|               0.35|           220|
> |     3|  0.437125748502994|           167|
> |     4|0.38202247191011235|            89|
> |     5| 0.4166666666666667|            48|
> |     6| 0.3157894736842105|            19|
> |     7|                0.0|             6|
> |     8|                1.0|             1|
> +------+-------------------+--------------+
> 
> survivedByAgeDecade: org.apache.spark.sql.DataFrame = [Survived: int, decade: int]
></pre>



Here we can see that youngest passengers had more chances to survive
**Problem 3.** Find all features with missing values. Suggest ways of handling features with missing values  and specify their advantages nad disadvantages. Apply these methods to a given data set.
**A.** Missing values can be replaced by the mean, the median or the most frequent value. The mean is not a robust tool since it is largely influenced by outliers and is better suited for normaly distributed features. The median is a more robust estimator for data with high magnitude variables and is generally used for skewed distributions. Fost frequent value is better suited for categorical features.

```scala
df.columns.filter(col => df.filter(df(col).isNull).count > 0)
```


><pre>
> res37: Array[String] = Array(Age)
></pre>




```scala
// using mean value
val meanAge = df.select(mean("Age")).first.getDouble(0)
df.select("Age").na.fill(meanAge).limit(10)
```


><pre>
> meanAge: Double = 29.69911764705882
> res39: org.apache.spark.sql.DataFrame = [Age: double]
></pre>




```scala
// using median value
import org.apache.spark.SparkContext._

def getMedian(rdd: RDD[Double]): Double = {
  val sorted = rdd.sortBy(identity).zipWithIndex().map {
    case (v, idx) => (idx, v)
  }

  val count = sorted.count()

  if (count % 2 == 0) {
    val l = count / 2 - 1
    val r = l + 1
    (sorted.lookup(l).head + sorted.lookup(r).head).toDouble / 2
  } else sorted.lookup(count / 2).head.toDouble
}
val ageRDD = df.filter(df("Age").isNotNull).select("Age").map(row => row.getDouble(0))
val medianAge = getMedian(ageRDD)

df.select("Age").na.fill(medianAge).limit(10)
```


><pre>
> import org.apache.spark.SparkContext._
> getMedian: (rdd: org.apache.spark.rdd.RDD[Double])Double
> ageRDD: org.apache.spark.rdd.RDD[Double] = MapPartitionsRDD[282] at map at <console>:91
> medianAge: Double = 28.0
> res41: org.apache.spark.sql.DataFrame = [Age: double]
></pre>



### C3 Charts
* http://c3js.org/examples.html
* also have a look at `viz/Simple & Flexible Custom C3 Charts` notebook supplied with spark-notebook distribution.

```scala
import notebook.front.widgets.CustomC3Chart
```


><pre>
> import notebook.front.widgets.CustomC3Chart
></pre>



** Problem 1. ** Plot funtion y(x) with blue color and it's confidence interval with green shaded area on the graph using data generated by following function.

```scala
import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions._
import math.{Pi=>pi}

val genData = () => {
  val x = linspace(0, 30, 100)
  val y = sin(x*pi/6.0) + DenseVector.rand(x.size, new Gaussian(0, 0.02))
  val error = DenseVector.rand(y.size, new Gaussian(0.1, 0.02))
  (x, y, error)
}
```


><pre>
> import breeze.linalg._
> import breeze.numerics._
> import breeze.stats.distributions._
> import math.{Pi=>pi}
> genData: () => (breeze.linalg.DenseVector[Double], breeze.linalg.DenseVector[Double], breeze.linalg.DenseVector[Double]) = <function0>
></pre>




```scala
// Incomplete solution (follow the issue https://github.com/c3js/c3/issues/402)

val (x, y, error) = genData()

case class Point(x: Double, y: Double, plusError: Double, minusError: Double)

val plotData = x.toArray.zip(y.toArray).zip(error.toArray).map(pp => Point(pp._1._1, 
                                                                           pp._1._2, 
                                                                           pp._1._2 + pp._2,
                                                                           pp._1._2 - pp._2))
CustomC3Chart(plotData,
              """{ data: { x: 'x', 
                          types: {y: 'line', plusError: 'line', minusError: 'line'},
                          colors: {y: 'blue',
                                   plusError: 'green',
                                   minusError: 'green'}
                         },
                    point: {
                      show: false
                    }
                  }""")
```


><pre>
> x: breeze.linalg.DenseVector[Double] = DenseVector(0.0, 0.30303030303030304, 0.6060606060606061, 0.9090909090909092, 1.2121212121212122, 1.5151515151515151, 1.8181818181818183, 2.121212121212121, 2.4242424242424243, 2.7272727272727275, 3.0303030303030303, 3.3333333333333335, 3.6363636363636367, 3.9393939393939394, 4.242424242424242, 4.545454545454546, 4.848484848484849, 5.151515151515151, 5.454545454545455, 5.757575757575758, 6.0606060606060606, 6.363636363636364, 6.666666666666667, 6.96969696969697, 7.272727272727273, 7.575757575757576, 7.878787878787879, 8.181818181818182, 8.484848484848484, 8.787878787878789, 9.090909090909092, 9.393939393939394, 9.696969696969697, 10.0, 10.303030303030303, 10.606060606060606, 10.90909090909091, 11.212121212121213, 11.515151515151516, 11.818181818181...
></pre>

<img src='https://github.com/drewnoff/spark-notebook-ml-labs/blob/public/labs/DataAnalysisToolbox/images/plotFunction.png?raw=true' alt='plot' height='252' width='978'></img>


** Problem 2. ** Plot histogram of ages for each passenger class (use data from Titanic dataset).

```scala
// Let's start with histogram of ages of all passengers.
val ageRdd = df.select("Age").rdd.map(r => r.getAs[Double](0))
val ageHist = ageRdd.histogram(10)

case class AgeHistPoint(ageBucket: Double, age: Long)

val ageHistData = ageHist._1.zip(ageHist._2).map(pp => AgeHistPoint(pp._1, pp._2))

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
> ageRdd: org.apache.spark.rdd.RDD[Double] = MapPartitionsRDD[312] at map at <console>:36
> ageHist: (Array[Double], Array[Long]) = (Array(0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0, 72.0, 80.0),Array(227, 33, 164, 181, 123, 74, 50, 26, 11, 2))
> defined class AgeHistPoint
> ageHistData: Array[AgeHistPoint] = Array(AgeHistPoint(0.0,227), AgeHistPoint(8.0,33), AgeHistPoint(16.0,164), AgeHistPoint(24.0,181), AgeHistPoint(32.0,123), AgeHistPoint(40.0,74), AgeHistPoint(48.0,50), AgeHistPoint(56.0,26), AgeHistPoint(64.0,11), AgeHistPoint(72.0,2))
> res47: notebook.front.widgets.CustomC3Chart[Array[AgeHistPoint]] = <CustomC3Chart widget>
></pre>

<img src='https://github.com/drewnoff/spark-notebook-ml-labs/blob/public/labs/DataAnalysisToolbox/images/ageHist.png?raw=true' alt='hist' height='252' width='978'></img>


```scala
// Now let's expand our solution.
val buckets = linspace(0, 100, 11).toArray
val p1AgesHist = df.filter(df("Pclass")===1)
                   .select("Age")
                   .rdd
                   .map(r => r.getAs[Double](0))
                   .histogram(buckets)
val p2AgesHist = df.filter(df("Pclass")===2)
                   .select("Age")
                   .rdd
                   .map(r => r.getAs[Double](0))
                   .histogram(buckets)
val p3AgesHist = df.filter(df("Pclass")===3)
                   .select("Age")
                   .rdd
                   .map(r => r.getAs[Double](0))
                   .histogram(buckets)
```


><pre>
> buckets: Array[Double] = Array(0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0)
> p1AgesHist: Array[Long] = Array(33, 18, 34, 50, 37, 27, 13, 3, 1, 0)
> p2AgesHist: Array[Long] = Array(28, 18, 53, 48, 18, 15, 3, 1, 0, 0)
> p3AgesHist: Array[Long] = Array(178, 66, 133, 69, 34, 6, 3, 2, 0, 0)
></pre>




```scala
case class AgeHistPoint(ageBucket: Double, c1: Long, c2: Long, c3: Long)

val ageHistData = (0 until buckets.length - 1).map(i => AgeHistPoint(buckets(i), p1AgesHist(i), p2AgesHist(i), p3AgesHist(i))).toArray
```


><pre>
> defined class AgeHistPoint
> ageHistData: Array[AgeHistPoint] = Array(AgeHistPoint(0.0,33,28,178), AgeHistPoint(10.0,18,18,66), AgeHistPoint(20.0,34,53,133), AgeHistPoint(30.0,50,48,69), AgeHistPoint(40.0,37,18,34), AgeHistPoint(50.0,27,15,6), AgeHistPoint(60.0,13,3,3), AgeHistPoint(70.0,3,1,2), AgeHistPoint(80.0,1,0,0), AgeHistPoint(90.0,0,0,0))
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
                    y: {label: 'Count'}
                   }
             }
             """)
```


><pre>
> res51: notebook.front.widgets.CustomC3Chart[Array[AgeHistPoint]] = <CustomC3Chart widget>
></pre>

<img src='https://github.com/drewnoff/spark-notebook-ml-labs/blob/public/labs/DataAnalysisToolbox/images/ageHistPerClass.png?raw=true' alt='ageHistPerClassStacked' height='252' width='978'></img>


```scala
// Using stacked bar chart
CustomC3Chart(ageHistData,
             chartOptions = """
             { data: { x: 'ageBucket', 
                       type: 'bar',
                       groups: [['c1', 'c2', 'c3']]},
               bar: {
                     width: {ratio: 0.9}
                    },
               axis: {
                    y: {label: 'Count'}
                   }
             }
             """)
```


><pre>
> res53: notebook.front.widgets.CustomC3Chart[Array[AgeHistPoint]] = <CustomC3Chart widget>
></pre>

<img src='https://github.com/drewnoff/spark-notebook-ml-labs/blob/public/labs/DataAnalysisToolbox/images/ageHistPerClassStacked.png?raw=true' alt='ageHistPerClassStacked' height='252' width='978'></img>
