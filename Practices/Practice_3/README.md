# Decision Tree Classifier  
Decision Trees are a non-parametric supervised learning method used for both classification and regression tasks. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.  

## Steps:  
### 1. Import libraries.
~~~
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
~~~
>  Importamos las librerias para utiliza Pipeline, IndexToString, StringIndexer, VectorIndexer, y las que corresponden al clasificador de árboles de decisión: DecisionTreeClassificationModel y DecisionTreeClassifier.  

### 2. Import a Spark Session.  
~~~
import org.apache.spark.sql.SparkSession
~~~

### 3. Create a Spark session.  
~~~
  def main(): Unit = {
    val spark = SparkSession.builder.appName("DecisionTreeClassificationExample").getOrCreate()
~~~
>  Se crea una nueva sesión de Spark, y se le asigna a la aplicación el nombre: "DecisionTreeClassificationExample".  

### 4.  Load the data stored in LIBSVM format as a DataFrame.  
~~~
    val data = spark.read.format("libsvm").load("sample_libsvm_data.txt")
~~~

### 5. Index labels, adding metadata to the label column.  
* Fit on whole dataset to include all labels in index.
~~~
    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
~~~
> Se indexan las etiquetas y se añaden metadatos para que los valores tipo texto pasen a valores numericos.  
> Se incluyen las etiquetas en el indice mediante un ajuste en todo el dataset (data).  

### 6. Automatically identify categorical features, and index them.
* Set maxCategories so features with > 4 distinct values are treated as continuous.
~~~
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)
~~~

### 7. Split the data into training and test sets.
* Split the data using random split into 70% for traingin and 30% held out for testing.  
~~~
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
~~~

### 8. Train a DecisionTree model.  
~~~
    val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
~~~
>  Se entrena el modelo usando DecisionTreeClassifier() y, señalando las columnas para el label y los features.  

### 9. Convert indexed labels back to original labels.  
~~~
    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
~~~
>  Se toman las columnas 'labelIndexer' y 'labels', para regresar los valores orginales de las etiquetas indexadas dentro de las columnas 'prediction' y 'predictedLabel'.  

### 10. Chain indexers and tree in a Pipeline.  
~~~
    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
~~~

### 11. Train model. 
~~~
    val model = pipeline.fit(trainingData)
~~~
>  Se ajusta el modelo a los datos de entrenamiento.  

### 12. Make predictions.  
~~~
    val predictions = model.transform(testData)
~~~
>  Se calculan las predcciones del modelo, transformando los datos de entrenamiento por los de prueba.  

### 13. Select example rows to display.  
~~~
    predictions.select("predictedLabel", "label", "features").show(5)
~~~
> Se seleccionan las columnas 'predictedLabel', 'label' y 'features' para desplegar en consola las primeras 5 filas de las predicciones calculadas por el modelo.  

### 14. Select (prediction, true label) and compute test error.  
~~~
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${(1.0 - accuracy)}")
~~~
> Se utiliza `evaluate` para calcular el nivel de exactitud de las predicciones realizadas por el modelo.  
> Esta misma cifra se utiliza para obtener el porcentaje de error del modelo, mediante la resta: 1.0 menos el nivel de exactitud.  

### 15. Print the tree obtained from the model.
~~~
    val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
    println(s"Learned classification tree model:\n ${treeModel.toDebugString}")

    spark.stop()
  }

main()
~~~
>  Se muestra completo el árbol generado por el modelo; en el aparecen las decisiones utilizadas y los resultados de predicción obtenidos para cada rama. También se menciona el nivel de profundidad y la cantidad de nodos resultantes.  
> En este caso se obtuvo un `Test Error = 0.02564102564102566` lo cual nos dejaría aproximadamente con un 98% de exactitud, significando que el modelo funciona bastante bien.  
