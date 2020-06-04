# Random Forest Classifier
En Random Forest se ejecutan varios algoritmos de árbol de decisiones en lugar de uno solo. Para clasificar un nuevo objeto basado en atributos, cada árbol de decisión da una clasificación y finalmente la decisión con más “votos” es la predicción del algoritmo. Para regresión, se toma el promedio de las salidas (predicciones) de todos los árboles.  

## Steps:  
### 1. Import libraries.
~~~
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
~~~

### 2. Import a Spark Session.  
~~~
import org.apache.spark.sql.SparkSession
~~~

### 3. Create a Spark session.  
~~~
  def main(): Unit = {
    val spark = SparkSession.builder.appName("RandomForestClassifierExample").getOrCreate()
~~~

### 4.  Load and parse the data file, converting it to a DataFrame.
~~~
    val data = spark.read.format("libsvm").load("sample_libsvm_data.txt")
~~~

### 5. Index labels, adding metadata to the label column.  
* Fit on whole dataset to include all labels in index.
~~~
    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
~~~
>  Se crea la variable `labelIndexer` para guardar la indexación de las etiquetas a las cuales les agregamos metadatos para pasarlas de tipo texto a tipo numerico, y posteriormente ajustarlas a todo el dataset (data).  

### 6. Automatically identify categorical features, and index them.
* Set maxCategories so features with > 4 distinct values are treated as continuous.
~~~
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)
~~~
>  Se indexan las caracteristicas dento de la columna features, y se establece un maximo de 4 categorias a partir del cual los valores seran tratados como continuos.  

### 7. Split the data into training and test sets.  
~~~
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
~~~
>  Se dividen los datos en 70% para entrenamiento y 30% para prueba.  

### 8. Train a RandomForest model.  
~~~
    val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)
~~~
>  Se usa la función `RandomForestClassifier()` para entrenar al modelo, mediante las columnas indexadas (label y features), y se indica el número total de árboles a generar (en este caso 10, que es el valor por defecto).  

### 9. Convert indexed labels back to original labels.  
~~~
    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
~~~
>  Se utiliza la función `IndexToString()` para convertir de nuevo las etiquetas indexadas a los valores tipo texto originales, y se guardan dentro de las columnas 'prediction' y 'predictedLabel'.  

### 10. Chain indexers and forest in a Pipeline.  
~~~
    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))
~~~
>  Se une todo lo hecho anteriormente dentro de una pipeline: los indexadores, el modelo (rf) y el resultado de la función `IndexToString` (labelConverter).  

### 11. Train model. 
~~~
    val model = pipeline.fit(trainingData)
~~~
>  Se entrena el modelo usando los datos de entrenamiento.  

### 12. Make predictions.  
~~~
    val predictions = model.transform(testData)
~~~
>  Se calculan las predicciones del modelo y se transforman los datos de entrenamiento a los de prueba.  

### 13. Select example rows to display.  
~~~
    predictions.select("predictedLabel", "label", "features").show(5)
~~~
> Se seleccionan las columnas "predictedLabel", "label" y "features" para desplegar en consola las primeras 5 filas de las predicciones hechas por el modelo.  

### 14. Select (prediction, true label) and compute test error.  
~~~
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${(1.0 - accuracy)}")
~~~
> Se calcula el nivel de exactitud del modelo en base a las predicciones obtenidas (función `evaluate`).  
> Se calcula el porcentaje de error mediante la resta: 1.0 menos el nivel de exactitud.  

### 15. Print the trees obtained from the model (10).
~~~
    val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
    println(s"Learned classification forest model:\n ${rfModel.toDebugString}")

    spark.stop()
  }

main()
~~~
>  Se imprimen todos los árboles generados por el modelo, en este caso 10, junto con sus condiciones y predicciones resultantes para cada una de sus ramas.  
> En esta ocasión se obtuvo un aproximado del 3% de error, lo cual nos deja con un 97% de nivel de exactitud para las predicciones realizadas por el modelo. Esto significa que el nivel de desempeño del modelo es muy bueno y por lo tanto los resultados con confiables y en gran medida exactos.  

