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

<img src="http://telegra.ph/file/cd760d2837a43c2738614.png" width=900>
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

<img src="http://telegra.ph/file/ff0fd209a3804c90b9601.png" width=900>
</img>


### DataFrame Functions and UDF

http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.sql.functions$

```scala
import org.apache.spark.sql.functions.{mean, min, max}

passengersDF.select(mean("Age").alias("Average Age"), min("Age"), max("Age")).show()
```
```
+-----------------+--------+--------+
|      Average Age|min(Age)|max(Age)|
+-----------------+--------+--------+
|29.69911764705882|    0.42|    80.0|
+-----------------+--------+--------+
```

```scala
import org.apache.spark.sql.functions.count

passengersDF.groupBy("Pclass")
            .agg(count("Pclass").alias("class_count"))
            .orderBy(-$"class_count")
            .show
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

For more specific tasks one can use User Defined Functions.

Let's say we want to get a column with full names of port of embarkation.

```scala
passengersDF.select("Embarked").distinct.show
```
```
+--------+
|Embarked|
+--------+
|       Q|
|    null|
|       C|
|       S|
+--------+
```

From dataset description we know that C = Cherbourg; Q = Queenstown; S = Southampton.

```scala
import org.apache.spark.sql.functions.udf

val embarkedFullName: (String) => String = (embarked: String) =>
  if (embarked == "Q")
    "Queenstown"
  else if (embarked == "C")
    "Cherbourg"
  else
    "Southampton"


val embarkedFullNameUDF = udf(embarkedFullName)
```

Also we want to get a column with more verbose survival status of passenger: `survived` and `died`.

```scala
val survivedStatus: (Integer) => String = (survived: Integer) =>
  if (survived == 1)
    "survived"
  else
    "died"

val survivedStatusUDF = udf(survivedStatus)

val pdf = passengersDF
        .withColumn("Embarkation", embarkedFullNameUDF($"Embarked"))
        .drop("Embarked")
        .withColumn("SurvivedStatus", survivedStatusUDF($"Survived"))
        .cache()
        
pdf.select("Name", "Embarkation", "SurvivedStatus").show(5, truncate=false)
```
```
+---------------------------------------------------+-----------+--------------+
|Name                                               |Embarkation|SurvivedStatus|
+---------------------------------------------------+-----------+--------------+
|Braund, Mr. Owen Harris                            |Southampton|died          |
|Cumings, Mrs. John Bradley (Florence Briggs Thayer)|Cherbourg  |survived      |
|Heikkinen, Miss. Laina                             |Southampton|survived      |
|Futrelle, Mrs. Jacques Heath (Lily May Peel)       |Southampton|survived      |
|Allen, Mr. William Henry                           |Southampton|died          |
+---------------------------------------------------+-----------+--------------+
only showing top 5 rows
```

**Q-5. Count the number and percentage of survivors and dead passengers.**

```scala
import org.apache.spark.sql.functions.count

val numPassengers = pdf.count()

val grBySurvived = pdf.groupBy("SurvivedStatus")
                      .agg(count("PassengerId").alias("count"), 
                           ((count("PassengerId") / numPassengers) * 100).alias("%"))
grBySurvived.show
```
```
+--------------+-----+-----------------+
|SurvivedStatus|count|                %|
+--------------+-----+-----------------+
|          died|  549|61.61616161616161|
|      survived|  342|38.38383838383838|
+--------------+-----+-----------------+
```

**Q-6.** 
- **Plot the distribution of dead and surviving passengers.**
- **Plot the distribution of survivors and dead passengers by class.**
- **Plot the distribution of survivors and dead passengers by gender.**
- **Plot the distribution of survivors and dead passengers by port of embarkation.**
- **Plot the % of survivors by port of embarkation.**
- **Plot the distribution of passenger classes by port of embarkation.**

```scala
// Distribution of dead and survived passengers

CustomPlotlyChart(grBySurvived,
                  layout="{title: 'Passengers by status', xaxis: {title: 'status'}, yaxis: {title: '%'}}",
                  dataOptions="{type: 'bar'}",
                  dataSources="{x: 'SurvivedStatus', y: '%'}")
```

<img src="http://telegra.ph/file/a03b6882b09456285e697.png" width=900>
</img>

```scala
// Distribution of the number of survivors and dead passengers by class.

CustomPlotlyChart(pdf.groupBy("SurvivedStatus", "Pclass").count,
                  layout="{title: 'Number of passengers by survival status per class', xaxis: {title: 'Pclass'}, barmode: 'group'}",
                  dataOptions="{type: 'bar', splitBy: 'SurvivedStatus'}",
                  dataSources="{x: 'Pclass', y: 'count'}")
```

<img src="http://telegra.ph/file/6e81137a13d88112dc7a3.png" width=900>
</img>

```scala
// Distribution of survivors and dead passengers by gender.

CustomPlotlyChart(pdf.groupBy("SurvivedStatus", "Sex").count,
                  layout="{title: 'Number of passengers by status by gender', xaxis: {title: 'Gender'}, barmode: 'group'}",
                  dataOptions="{type: 'bar', splitBy: 'SurvivedStatus'}",
                  dataSources="{x: 'Sex', y: 'count'}")
```

<img src="http://telegra.ph/file/3d717a5eb7e265dc858eb.png" width=900>
</img>

```scala
// Distribution of survivors and dead passengers by port of embarkation.

CustomPlotlyChart(pdf.groupBy("Embarkation", "SurvivedStatus").count,
                  layout="{barmode: 'group'}",
                  dataOptions="{type: 'bar', splitBy: 'SurvivedStatus'}",
                  dataSources="{x: 'Embarkation', y: 'count'}")
```

<img src="http://telegra.ph/file/75f81dd53d5bcb8f8db4e.png" width=900>
</img>

```scala
// % of survivors by port of embarkation.

CustomPlotlyChart(pdf.groupBy("Embarkation").agg((sum("Survived") / count("Survived") * 100).alias("SurvivalRate")),
                  layout="{title: '% of survival per embarkation'}",
                  dataOptions="{type: 'bar'}",
                  dataSources="{x: 'Embarkation', y: 'SurvivalRate'}")
```

<img src="http://telegra.ph/file/c430db6b285ae804f7d53.png" width=900>
</img>

```scala
// Distribution of passenger classes by port of embarkation.

CustomPlotlyChart(pdf.groupBy("Embarkation", "Pclass").count,
                  layout="{barmode: 'stack', title: 'Pclass distribution by Embarkation'}",
                  dataOptions="{type: 'bar', splitBy: 'Pclass'}",
                  dataSources="{x: 'Embarkation', y: 'count'}")
```
<img src="http://telegra.ph/file/9df9159d58880165e8a46.png" width=900>
</img>

How to get the % of survived passengers by port of embarkation in this case?

```scala
val byEmbark =  pdf.groupBy("Embarkation").agg(count("PassengerId").alias("totalCount"))
val byEmbarkByClass = pdf.groupBy("Embarkation", "Pclass").count

val embarkClassDistr = byEmbarkByClass.join(byEmbark, usingColumn="Embarkation")
                                      .select($"Embarkation",
                                              $"Pclass", 
                                              ($"count" / $"totalCount" * 100).alias("%"))

CustomPlotlyChart(embarkClassDistr,
                  layout="{barmode: 'stack', title: 'Pclass distribution by Embarkation', yaxis: {title: '%'}}",
                  dataOptions="{type: 'bar', splitBy: 'Pclass'}",
                  dataSources="{x: 'Embarkation', y: '%'}")
```

<img src="http://telegra.ph/file/53cf0b46923ad41549287.png" width=900>
</img>

### Histograms and Box Plots

**Q-7 Obtain age distributions by passengers survival status.**

```scala
CustomPlotlyChart(pdf, 
                  layout="{title: 'Age distribution by status', xaxis: {title: 'Age'}, barmode: 'overlay'}",
                  dataOptions="{type: 'histogram', opacity: 0.6, splitBy: 'SurvivedStatus'}",
                  dataSources="{x: 'Age'}")
```
<img src="http://telegra.ph/file/1b109fed97fb2e2eaa494.png" width=900>
</img>

```scala
CustomPlotlyChart(pdf, 
                  layout="{yaxis: {title: 'Age'}}",
                  dataOptions="{type: 'box', splitBy: 'SurvivedStatus'}",
                  dataSources="{y: 'Age'}")
```
<img src="http://telegra.ph/file/f8270cfa13e9245f998c0.png" width=900>
</img>

**Q-8. Plot box plots of age distributions by passengers classes.**

```scala
CustomPlotlyChart(pdf, 
                  layout="{yaxis: {title: 'Age'}}",
                  dataOptions="{type: 'box', splitBy: 'Pclass'}",
                  dataSources="{y: 'Age'}")
```

<img src="http://telegra.ph/file/8613f3df86862869837a4.png" width=900>
</img>

This scatter plots show the dependences of the chances of survival from the cabin class, age and gender:

```scala
val survByClassAndAge = List("male", "female").map{
  gender =>
    CustomPlotlyChart(pdf.filter($"Sex" === gender),
                  layout=s"""{
                    title: 'Survival by class and age, $gender.', 
                    yaxis: {title: 'class'}, 
                    xaxis: {title: 'age'}
                  }""",
                  dataOptions="""{
                    splitBy: 'SurvivedStatus',
                    byTrace: {
                      'survived': {
                        mode: 'markers',
                        marker: {
                          size: 20,
                          opacity: 0.3,
                          color: 'orange'
                        }
                      },
                      'died': {
                        mode: 'markers',
                        marker: {
                          size: 15,
                          opacity: 0.9,
                          color: 'rgba(55, 128, 191, 0.6)'
                        }
                      }
                    }
                  }""",
                  dataSources = "{x: 'Age', y: 'Pclass'}"
                     )
}

survByClassAndAge(0)
```

<img src="http://telegra.ph/file/e91733ba23293250a7500.png" width=900>
</img>

```scala
survByClassAndAge(1)
```

<img src="http://telegra.ph/file/8f52fcd59c70e68664a0c.png" width=900>
</img>

### More practice with UDF and Box Plots

The titles of passengers could be useful source of information. Let's explore that.

**Q-9. Plot box plots of age distributions by title.**

```scala
pdf.select("Name").show(3, truncate=false)
```
```
+---------------------------------------------------+
|Name                                               |
+---------------------------------------------------+
|Braund, Mr. Owen Harris                            |
|Cumings, Mrs. John Bradley (Florence Briggs Thayer)|
|Heikkinen, Miss. Laina                             |
+---------------------------------------------------+
only showing top 3 rows
```

```scala
val parseTitle: String => String = (name: String) =>
  name.split(", ")(1).split("\\.")(0)

val parseTitleUDF = udf(parseTitle)

CustomPlotlyChart(pdf.withColumn("Title", parseTitleUDF($"Name")), 
                  layout="{yaxis: {title: 'Age'}}",
                  dataOptions="{type: 'box', splitBy: 'Title'}",
                  dataSources="{y: 'Age'}")
```

<img src="http://telegra.ph/file/9036277a7a6e4b47390b4.png" width=900>
</img>

Often it is good practice to group the values of the categorical feature, especially when there are rare individual feature values such as `Don`, `Lady`, `Capt` in our case.

**Q-10. Write UDF to group all the titles into five groups according to the following table:**

|   Group       |    Title     |
| :------------:|:------------:|
| Aristocratic  | Capt, Col, Don, Dr, Jonkheer, Lady, Major, Rev, Sir, Countess |
| Mrs           | Mrs, Ms         |
| Miss          | Miss, Mlle, Mme |
| Mr            | Mr              |
| Master        | Master          |

** Create new column called 'TitleGroup' and plot box plots of age distributions by title group.**

```scala
val titleGroup: String => String = (title: String) => {
  val aristocratic = Set("Capt", "Col", "Don", "Dr", "Jonkheer", "Lady", "Major", "Rev", "Sir", "the Countess")
  val mrs = Set("Mrs", "Ms")
  val miss = Set("Miss", "Mlle", "Mme")
  if (aristocratic.contains(title))
    "Aristocratic"
  else if (mrs.contains(title))
    "Mrs"
  else if (miss.contains(title))
    "Miss"
  else
    title
}

// given column with passenger name obtain column with passenger title group.
val parseTitleGroupUDF = udf(parseTitle andThen titleGroup)
```

```scala
val withTitleDF = pdf.withColumn("TitleGroup", parseTitleGroupUDF($"Name"))

CustomPlotlyChart(withTitleDF, 
                  layout="{yaxis: {title: 'Age'}}",
                  dataOptions="{type: 'box', splitBy: 'TitleGroup'}",
                  dataSources="{y: 'Age'}")
```

<img src="http://telegra.ph/file/03cdce9bc6bcfdffb2a68.png" width=900>
</img>


**Q-11 Plot the distribution of the % of survivors by title group.**

```scala
val byTitleGr = withTitleDF
                   .groupBy("TitleGroup")
                   .agg((sum("Survived") / count("Survived") * 100).alias("%"))

CustomPlotlyChart(byTitleGr,
                  layout="{title: '% of survival by title group'}",
                  dataOptions="{type: 'bar'}",
                  dataSources="{x: 'TitleGroup', y: '%'}")
```

<img src="http://telegra.ph/file/71e50da227e09979866f9.png" width=900>
</img>

### Handling missing values

```scala
import org.apache.spark.sql.functions.isnull

100.0 * pdf.filter(isnull($"Age")).count / pdf.count
```
```
res209: Double = 19.865319865319865
19.865319865319865
```

```scala
100.0 * pdf.filter(isnull($"Cabin")).count / pdf.count
```
```
res237: Double = 77.10437710437711
77.10437710437711
```

```scala
val cabinStatus: (String) => String = (cabin: String) =>
  if (cabin == null)
    "noname"
  else
    "hasNumber"

val cabinStatusUDF = udf(cabinStatus)
```

```scala
val withCabinStatusDF = pdf.withColumn("CabinStatus", cabinStatusUDF($"Cabin"))
```

```scala
CustomPlotlyChart(withCabinStatusDF.groupBy("CabinStatus", "SurvivedStatus").count,
                  layout="{title: 'Number of passengers by survival status by cabin type', xaxis: {title: 'Cabin'}}",
                  dataOptions="{type: 'bar', splitBy: 'SurvivedStatus'}",
                  dataSources="{x: 'CabinStatus', y: 'count'}")
```

<img src="http://telegra.ph/file/80f0099117e7772825118.png" width=900>
</img>

### On your own

Explore family relationships variables (SibSp and Parch).
How does the number of siblings/spouses aboard affect the chances of survival?
How does the number of parents/children aboard affect the chances of survival?

Invent a new variable called `Family` to represent total number of relatives aboard and explore how does it affect hte chances of survival.
