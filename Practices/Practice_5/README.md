# Gradient-Boosted Tree (GBT) Classifier

## Steps:  
### 1. Import libraries.
~~~
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
~~~
>  

### 2. Import a Spark Session.  
~~~
import org.apache.spark.sql.SparkSession
~~~

### 3. Use the Error reporting code.
~~~
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
~~~
>  

### 4. Create a Spark session.  
~~~
  def main(): Unit = {
    val spark = SparkSession.builder.appName("GradientBoostedTreeClassifierExample").getOrCreate()
~~~
>  

### 5.  Load and parse the data file, converting it to a DataFrame.
* Print the schema.
~~~
    val data = spark.read.format("libsvm").load("sample_libsvm_data.txt")
    data.printSchema()
~~~
>  

### 6. Index labels, adding metadata to the label column.  
* Fit on whole dataset to include all labels in index.
~~~
    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
~~~
>  

### 7. Automatically identify categorical features, and index them.
* Set maxCategories so features with > 4 distinct values are treated as continuous.
~~~
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)
~~~
>  

### 8. Split the data into training and test sets.
* 30% held out for testing.  
~~~
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
~~~
>  

### 9. Train a GBT model.  
~~~
    val gbt = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(10).setFeatureSubsetStrategy("auto")
~~~
>  

### 10. Convert indexed labels back to original labels.  
~~~
    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
~~~
>  

### 11. Chain indexers and GBT in a Pipeline.  
~~~
    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))
~~~
>  

### 12. Train model. 
* This also runs the indexers.  
~~~
    val model = pipeline.fit(trainingData)
~~~
>  

### 13. Make predictions.  
~~~
    val predictions = model.transform(testData)
~~~
>  

### 14. Select example rows to display.  
~~~
    predictions.select("predictedLabel", "label", "features").show(5)
~~~
> 

### 15. Select (prediction, true label) and compute test error.  
~~~
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${1.0 - accuracy}")
~~~
>  

### 16. Print result of Trees using GBT (10).
~~~
    val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
    println(s"Learned classification GBT model:\n ${gbtModel.toDebugString}")

    spark.stop()
  }

main()
~~~
>  



