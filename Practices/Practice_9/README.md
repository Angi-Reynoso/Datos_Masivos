# Naive Bayes   
Naive Bayes classifiers are a set of probabilistic classifiers that aim to process, analyze, and categorize data. Introduced in the 1960's Bayes classifiers have been a popular tool for text categorization, which is the sorting of data based upon the textual content. An example of this is email filtering, where emails containing specific suspicious words may be flagged as spam.  
Naive Bayes is essentially a technique for assigning classifiers to a finite set. However, there is no single algorithm for training these classifiers, so Naive Bayes assumes that the value of a specific feature is independent from the value of any other feature, given the class variable. For example, a machine may be considered to be a car because it is large, has four wheels, and a rectangular shape. The Naive Bayes classifier would consider each of these features to contribute independently to the likelihood that the object is a car, regardless of any correlations between number of wheels, size, or shape.  


## Steps:  
### 1. Import libraries.
~~~
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
~~~

### 2. Import a Spark Session.  
~~~
import org.apache.spark.sql.SparkSession
~~~

### 3. Create a Spark session.  
~~~
  def main(): Unit = {
    val spark = SparkSession.builder.appName("NaiveBayesExample").getOrCreate()
~~~  
> The application is also named: "NaiveBayesExample".  

### 4. Load the data stored in LIBSVM format as a DataFrame.  
~~~
    val data = spark.read.format("libsvm").load("sample_libsvm_data.txt")
~~~  

### 5. Split the data into training and test sets.  
~~~
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed = 1234L)
~~~
> Split the data using random split into 70% traning and 30% testing datasets.    

### 6. Train a NaiveBayes model.  
~~~
    val model = new NaiveBayes().fit(trainingData)
~~~
> Se entrena el modelo y se ajusta a los datos de entrenamiento.   

### 7. Select example rows to display.  
~~~
    val predictions = model.transform(testData)
        predictions.show()
~~~  
> Se calculan las predicciones al transformar el modelo a los datos de prueba, y se despliegan los resultados.  

### 8. Select (prediction, true label) and compute test error.  
~~~
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test set accuracy = $accuracy")

    spark.stop()
  }
main()
~~~  
> Se seleccionan las columnas "label" y "prediction" para calcular el nivel de exactitud del modelo.  
> Se calcula el nivel de exactitud en base a las predicciones hechas por el modelo y se utiliza la funcion `evaluate`.  
> Se imprime el resultado, y en este caso se obtuvo un nivel de exactitud del 100%, lo cual nos indica que el modelo es completamente funcional y que las predicciones fueron realizadas de manera correcta.  
