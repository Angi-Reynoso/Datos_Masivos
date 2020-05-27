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
* Split the data using random split into 80% traning and 20% testing datasets

> `inputData.randomSplit`: randomly splits a RDD with the provided weights. 

### 6. Instantiate the base classifier

> First configure the regression object without having to have a base logistic model at hand so it can be fed into the classifier
~~~
val classifier = new LogisticRegression()
.setMaxIter(10)
.setTol(1E-6)
.setFitIntercept(true)
~~~

### 7. Instantiate the One Vs Rest Classifier.
> In this step, we feed the configured regression model into the classifier 
~~~
val ovr = new OneVsRest().setClassifier(classifier)
~~~
### 8. Train the multiclass model.
> We generate a trained model by invoking the fit method on our one vs rest object
~~~
val ovrModel = ovr.fit(train)
~~~

### 9. Score the model on test data.
> Now, we will use the trained model to generate predictions for the test data
~~~
val predictions = ovrModel.transform(test)
~~~

### 10. Obtain evaluator.
> We pass predictions to the Multi Class Classification Evaluator to generate an accurancy value
~~~
val evaluator = new MulticlassClassificationEvaluator()
.setMetricName("accuracy")
~~~
### 11. Compute the classification error on test data.
~~~
val accuracy = evaluator.evaluate(predictions)
~~~
### 12. Print result
~~~
println(s"Test Error = ${1 - accuracy}")
~~~
