# One-vs-Rest classifier (a.k.a. One-vs-All)

## Steps:  
### 1. Import libraries.
~~~
import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
~~~

### 2. Import a Spark Session.  
~~~
import org.apache.spark.sql.SparkSession
~~~

### 3. Create a Spark session.  
~~~
  def main(): Unit = {
    val spark = SparkSession.builder.appName("MulticlassClassificationEvaluator").getOrCreate()
~~~

### 4. load data file.
~~~
val inputData = spark.read.format("libsvm")
  .load("data/mllib/sample_multiclass_classification_data.txt")
~~~

### 5. generate the train/test split.
~~~
val Array(train, test) = inputData.randomSplit(Array(0.8, 0.2))
~~~
> `inputData.randomSplit`:

### 6. instantiate the base classifier
~~~
   val classifier = new LogisticRegression()
  .setMaxIter(10)
  .setTol(1E-6)
  .setFitIntercept(true)
~~~
> `.setMaxIter(10)`:  
> `.setTol(1E-6)`:  
> `.setFitIntercept(true)`:  
