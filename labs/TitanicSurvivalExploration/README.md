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
