
## Steps:  
### 1. Import libraries and package
~~~
package org.apache.spark.examples.ml
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
~~~

### 2. Import a Spark Session.
~~~
import org.apache.spark.sql.SparkSession
~~~

### 3. Load the data stored in LIBSVM format as a DataFrame.
~~~
val data = spark.read.format("libsvm")
  .load("data/mllib/sample_multiclass_classification_data.txt")
~~~
### 4. Split the data into train and test
~~~
val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)
~~~
### 5. specify layers for the neural network:
> input layer of size 4 (features), two intermediate of size 5 and 4 and output of size 3 (classes)
~~~
val layers = Array[Int](4, 5, 4, 3)
~~~
### 6. create the trainer and set its parameters
~~~
val trainer = new MultilayerPerceptronClassifier()
  .setLayers(layers)
  .setBlockSize(128)
  .setSeed(1234L)
  .setMaxIter(100)
~~~
### 7. train the model
~~~
val model = trainer.fit(train)
~~~

### 8. Compute accuracy on the test set
~~~
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator()
  .setMetricName("accuracy")
~~~
### 9. Print result
~~~
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
~~~
