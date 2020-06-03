# Decision Tree Classifier 

## Steps:  
### 1. Import libraries.
~~~
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
~~~
>  

### 2. Import a Spark Session.  
~~~
import org.apache.spark.sql.SparkSession
~~~

### 3. Create a Spark session.  
~~~
  def main(): Unit = {
    val spark = SparkSession.builder.appName("DecisionTreeClassificationExample").getOrCreate()
~~~
>  

### 4.  Load the data stored in LIBSVM format as a DataFrame.  
~~~
    val data = spark.read.format("libsvm").load("sample_libsvm_data.txt")
~~~
>  

### 5. Index labels, adding metadata to the label column.  
* Fit on whole dataset to include all labels in index.
~~~
    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
~~~
>  

### 6. Automatically identify categorical features, and index them.
* Set maxCategories so features with > 4 distinct values are treated as continuous.
~~~
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)
~~~
>  

### 7. Split the data into training and test sets.
* Split the data using random split into 70% 
* 30% held out for testing.  
~~~
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
~~~
>  

### 8. Train a DecisionTree model.  
~~~
    val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
~~~
>  

### 9. Convert indexed labels back to original labels.  
~~~
    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
~~~
>  

### 10. Chain indexers and tree in a Pipeline.  
~~~
    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
~~~
>  

### 11. Train model. 
* This also runs the indexers.  
~~~
    val model = pipeline.fit(trainingData)
~~~
>  

### 12. Make predictions.  
~~~
    val predictions = model.transform(testData)
~~~
>  

### 13. Select example rows to display.  
~~~
    predictions.select("predictedLabel", "label", "features").show(5)
~~~
> 

### 14. Select (prediction, true label) and compute test error.  
~~~
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${(1.0 - accuracy)}")
~~~
>  

### 15. Print the tree obtained from the model.
~~~
    val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
    println(s"Learned classification tree model:\n ${treeModel.toDebugString}")

    spark.stop()
  }

main()
~~~
>  

