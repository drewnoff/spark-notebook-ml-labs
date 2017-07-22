#  Titanic Survival Exploration

<div style="text-align:center">
  <img src="http://telegra.ph/file/f5adcd2e260285ed766bd.png" width="192" height="100" style="margin-right:70px">
  <img src="http://telegra.ph/file/72de05a5e0fc1e392e569.png" width="111" height="128">
</div>

## Spark quick review

Spark provides convenient programming abstraction and parallel runtime to hide distributed computations complexities.


<img src="http://telegra.ph/file/41a2ce855b179b4e9bd44.png" width="316" height="149">


In this first lab we will focus on DataFrames and SQL.
In second lab we will use Spark MLlib for building machine learning pipelines.

### Spark Cluster

<div style="text-align:left">
  <img src="http://telegra.ph/file/cf242107c6e3fc854ce04.png" width="567" height="492">
</div>

Main entry point for Spark functionality is a `SparkContex`. `SparkContext` tells Spark how to access a cluster.
`Spark Notebook` automatically creates `SparkContext`.

Examples of `master` parameter configuration for `SparkContext`:

| Master Parameter  |             Description                 |
| ----------------- |----------------------------------------:|
| local[K]          | run Spark locally with K worker threads |
| spark://HOST:PORT | connect to Spark Standalone cluster     |
| mesos://HOST:PORT | connect to Mesos cluster                |

```scala
sparkContext
```

## Spark SQL and DataFrames

* http://spark.apache.org/docs/latest/sql-programming-guide.html
* http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.sql.Dataset

A DataFrame is a distributed collection of data organized into named columns.
It is conceptually equivalent to a table in a relational database or a data frame in R/Python

The entry point to programming Spark with SQL and DataFrame API in Spark 2.0 is the new `SparkSession` class:

```scala
sparkSession
```

```scala
val spark = sparkSession

// This import is needed to use the $-notation
import spark.implicits._
```

With a SparkSession you can create DataFrames from an existing RDD, from files in HDFS or any other storage system, or from Scala collections.

```scala
Seq(("Alice", 20, "female"), ("Bob", 31, "male"), ("Eva", 16, "female")).toDF("name", "age", "gender").show()
```

```
+-----+---+------+
| name|age|gender|
+-----+---+------+
|Alice| 20|female|
|  Bob| 31|  male|
|  Eva| 16|female|
+-----+---+------+
```

```scala
case class Person(name: String, age: Int, gender: String)

val persons = Seq(Person("Alice", 20, "female"), Person("Bob", 31, "male"), Person("Eva", 16, "female")).toDF()
persons.show()
```

```
+-----+---+------+
| name|age|gender|
+-----+---+------+
|Alice| 20|female|
|  Bob| 31|  male|
|  Eva| 16|female|
+-----+---+------+

persons: org.apache.spark.sql.DataFrame = [name: string, age: int ... 1 more field]
```

```scala
persons.select("name", "age").show()
```

```
+-----+---+
| name|age|
+-----+---+
|Alice| 20|
|  Bob| 31|
|  Eva| 16|
+-----+---+
```

```scala
val young = persons.filter($"age" < 21)
young.show()
```

```
+-----+---+------+
| name|age|gender|
+-----+---+------+
|Alice| 20|female|
|  Eva| 16|female|
+-----+---+------+

young: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [name: string, age: int ... 1 more field]
```

```scala
young.select(young("name"), ($"age" + 1).alias("incremented age")).show()
```

```
+-----+---------------+
| name|incremented age|
+-----+---------------+
|Alice|             21|
|  Eva|             17|
+-----+---------------+
```

```scala
persons.groupBy("gender").count.show
```

```
+------+-----+
|gender|count|
+------+-----+
|female|    2|
|  male|    1|
+------+-----+
```

# Titanic Dataset

More on this dataset you can read [here](https://www.kaggle.com/c/titanic/data).

<div style="text-align:left">
  <img src="https://upload.wikimedia.org/wikipedia/commons/6/6e/Stöwer_Titanic.jpg" width="427" height="292">
</div>
<div style="font-size:x-small">
  By <span class="fn value"><a href="//commons.wikimedia.org/wiki/Willy_St%C3%B6wer" title="Willy Stöwer">Willy Stöwer</a>, died on 31st May 1931</span> - Magazine Die Gartenlaube, <a href="https://en.wikipedia.org/wiki/Die_Gartenlaube" class="extiw" title="en:Die Gartenlaube">en:Die Gartenlaube</a> and <a href="https://de.wikipedia.org/wiki/Die_Gartenlaube" class="extiw" title="de:Die Gartenlaube">de:Die Gartenlaube</a>, Public Domain, <a href="https://commons.wikimedia.org/w/index.php?curid=97646">Link</a>
</div>

Out of the box, DataFrame supports reading data from the most popular formats, including JSON files, CSV files, Parquet files, Hive tables.

```scala
val passengersDF = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv("notebooks/spark-notebook-ml-labs/labs/TitanicSurvivalExploration/data/titanic_train.csv")  
  
passengersDF.printSchema
```

```
root
 |-- PassengerId: integer (nullable = true)
 |-- Survived: integer (nullable = true)
 |-- Pclass: integer (nullable = true)
 |-- Name: string (nullable = true)
 |-- Sex: string (nullable = true)
 |-- Age: double (nullable = true)
 |-- SibSp: integer (nullable = true)
 |-- Parch: integer (nullable = true)
 |-- Ticket: string (nullable = true)
 |-- Fare: double (nullable = true)
 |-- Cabin: string (nullable = true)
 |-- Embarked: string (nullable = true)
```

Look at 5 records in passengers DataFrame:

```scala
passengersDF.show(5, truncate=false)
```

```
+-----------+--------+------+---------------------------------------------------+------+----+-----+-----+----------------+-------+-----+--------+
|PassengerId|Survived|Pclass|Name                                               |Sex   |Age |SibSp|Parch|Ticket          |Fare   |Cabin|Embarked|
+-----------+--------+------+---------------------------------------------------+------+----+-----+-----+----------------+-------+-----+--------+
|1          |0       |3     |Braund, Mr. Owen Harris                            |male  |22.0|1    |0    |A/5 21171       |7.25   |null |S       |
|2          |1       |1     |Cumings, Mrs. John Bradley (Florence Briggs Thayer)|female|38.0|1    |0    |PC 17599        |71.2833|C85  |C       |
|3          |1       |3     |Heikkinen, Miss. Laina                             |female|26.0|0    |0    |STON/O2. 3101282|7.925  |null |S       |
|4          |1       |1     |Futrelle, Mrs. Jacques Heath (Lily May Peel)       |female|35.0|1    |0    |113803          |53.1   |C123 |S       |
|5          |0       |3     |Allen, Mr. William Henry                           |male  |35.0|0    |0    |373450          |8.05   |null |S       |
+-----------+--------+------+---------------------------------------------------+------+----+-----+-----+----------------+-------+-----+--------+
only showing top 5 rows
```

The sql function on a SparkSession enables applications to run SQL queries programmatically and returns the result as a DataFrame.
To do this we need to register the DataFrame as a SQL temporary view

```scala
passengersDF.createOrReplaceTempView("passengers")

spark.sql("""
  SELECT Name, Age, Pclass, Survived FROM passengers
  WHERE Age < 30
""").show(3, truncate=false)
```

```
+------------------------------+----+------+--------+
|Name                          |Age |Pclass|Survived|
+------------------------------+----+------+--------+
|Braund, Mr. Owen Harris       |22.0|3     |0       |
|Heikkinen, Miss. Laina        |26.0|3     |1       |
|Palsson, Master. Gosta Leonard|2.0 |3     |0       |
+------------------------------+----+------+--------+
only showing top 3 rows
```

### Transformations and Actions

Spark operations on DataFrames are one of two types. 
* Transformations are lazily evaluated and create new Dataframes from existing ones. 
* Actions trigger computation and return results or write DataFrames to storage.

*Computations are only triggered when an action is invoked.*

Here are some examples.


|   Transformations   |    Actions   |
| :-----------------: |:------------:|
| select              |  count       |
| filter              |  show        |
| groupBy             |  save        |
| orderBy             |  **collect** |
| sample              |  take        |
| limit               |  reduce      |
| withColumn          ||
| join                ||

**Q-1. How many different classes of passengers were aboard the Titanic?**

```scala
val pclasses = passengersDF.select("Pclass").distinct

pclasses.count
```
```
res141: Long = 3
3
```

```scala
pclasses.show
```
```
+------+
|Pclass|
+------+
|     1|
|     3|
|     2|
+------+
```

```scala
spark.sql("""
  SELECT DISTINCT Pclass from passengers
""").count
```
```
res145: Long = 3
3
```

**Q-2. How many passengers were in each class?**

```scala
val numByClass = passengersDF.groupBy("Pclass").count
numByClass.show
```
```
+------+-----+
|Pclass|count|
+------+-----+
|     1|  216|
|     3|  491|
|     2|  184|
+------+-----+
```

```scala
spark.sql("""
 SELECT Pclass, count(PassengerID) as class_count FROM passengers
 GROUP BY Pclass
 ORDER BY class_count DESC
""").show
```
```
+------+-----------+
|Pclass|class_count|
+------+-----------+
|     3|        491|
|     1|        216|
|     2|        184|
+------+-----------+
```

```scala
CustomPlotlyChart(numByClass,
                  layout="{title: 'Passengers per class', xaxis: {title: 'Pclass'}}",
                  dataOptions="{type: 'bar'}",
                  dataSources="{x: 'Pclass', y: 'count'}")
```

<img src="http://telegra.ph/file/cd760d2837a43c2738614.png" width=800>
</img>

**Q-3. How many women and men were in each class?**
```scala
val grByGenderAndClass = passengersDF.groupBy("Pclass", "Sex").count
grByGenderAndClass.show()
```
```
+------+------+-----+
|Pclass|   Sex|count|
+------+------+-----+
|     2|female|   76|
|     3|  male|  347|
|     1|  male|  122|
|     3|female|  144|
|     1|female|   94|
|     2|  male|  108|
+------+------+-----+
```

```scala
CustomPlotlyChart(grByGenderAndClass,
                  layout="{title: 'Passengers per class', xaxis: {title: 'Pclass'}, barmode: 'group'}",
                  dataOptions="{type: 'bar', splitBy: 'Sex'}",
                  dataSources="{x: 'Pclass', y: 'count'}")
```

<img src="http://telegra.ph/file/ff0fd209a3804c90b9601.png" width=800>
</img>


### DataFrame Functions and UDF
