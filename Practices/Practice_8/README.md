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

### 4. Load data file.
~~~
val inputData = spark.read.format("libsvm")
  .load("data/mllib/sample_multiclass_classification_data.txt")
~~~

### 5. Generate the train/test split.
~~~
val Array(train, test) = inputData.randomSplit(Array(0.8, 0.2))
~~~
> `inputData.randomSplit`:

### 6. Instantiate the base classifier
~~~
val classifier = new LogisticRegression()
.setMaxIter(10)
.setTol(1E-6)
.setFitIntercept(true)
~~~
> `.setMaxIter(10)`:  
> `.setTol(1E-6)`:  
> `.setFitIntercept(true)`:  
### 7. Instantiate the One Vs Rest Classifier.
~~~
val ovr = new OneVsRest().setClassifier(classifier)
~~~
### 8. Train the multiclass model.
~~~
val ovrModel = ovr.fit(train)
~~~

### 9. Score the model on test data.
~~~
val predictions = ovrModel.transform(test)
~~~

### 10. Obtain evaluator.
~~~
val evaluator = new MulticlassClassificationEvaluator()
.setMetricName("accuracy")
~~~
### 11. Compute the classification error on test data.
~~~
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${1 - accuracy}")
~~~
