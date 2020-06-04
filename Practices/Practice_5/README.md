# Gradient-Boosted Tree (GBT) Classifier
Es una técnica de aprendizaje automático utilizada para el análisis de regresión y para problemas de clasificación estadística, que produce un modelo predictivo en forma de un conjunto de modelos de predicción débiles, normalmente árboles de decisión. GBT construye árboles de uno en uno, donde cada árbol nuevo ayuda a corregir los errores cometidos por un árbol previamente entrenado.  

## Steps:  
### 1. Import libraries.
~~~
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
~~~

### 2. Import a Spark Session.  
~~~
import org.apache.spark.sql.SparkSession
~~~

### 3. Use the Error reporting code.
~~~
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
~~~
>  Sirve para reducir los errores desplegados en consola durante la ejecución del código.  

### 4. Create a Spark session.  
~~~
  def main(): Unit = {
    val spark = SparkSession.builder.appName("GradientBoostedTreeClassifierExample").getOrCreate()
~~~
>  Se crea una sesión de Spark, y se asigna "GradientBoostedTreeClassifierExample" como nombre de la aplicación.  

### 5.  Load and parse the data file, converting it to a DataFrame.
* Print the schema.
~~~
    val data = spark.read.format("libsvm").load("sample_libsvm_data.txt")
    data.printSchema()
~~~
>  Se carga el dataset desde el archivo "sample_libsvm_data.txt", y se imprime el esquema del mismo.  

### 6. Index labels, adding metadata to the label column.  
* Fit on whole dataset to include all labels in index.
~~~
    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
~~~
>  Se indexan las etiquetas en la columna "label", se añaden metadatos para cambiar los valores de tipo string a tipo numerico y se ajusta al dataset (data).  

### 7. Automatically identify categorical features, and index them.
* Set maxCategories so features with > 4 distinct values are treated as continuous.
~~~
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)
~~~
> Se indexa la columna features, y se estable un limite de 4 categorias, para que apartir de ello los datos sean tratados como continuos. También se ajustan los cambios al dataset (data).  

### 8. Split the data into training and test sets.
~~~
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
~~~
> Se dividen los datos en 70% entrenamiento y 30% prueba.  

### 9. Train a GBT model.  
~~~
    val gbt = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(10).setFeatureSubsetStrategy("auto")
~~~
> Se entrena el modelo con la función `GBTClassifier()`.  
> Se usan las columnas indexadas, y se indica un maximo de 10 iteraciones para la ejecución del modelo.  

### 10. Convert indexed labels back to original labels.  
~~~
    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
~~~
> Se convierten nuevamente las etiquetas indexadas en sus valores originales, con la función `IndexToString()`.  

### 11. Chain indexers and GBT in a Pipeline.  
~~~
    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))
~~~
> Se unen los indexadores, el modelo (gbt) y la variable labelConverter dentro del pipeline.  

### 12. Train model. 
~~~
    val model = pipeline.fit(trainingData)
~~~
> Se entrena el modelo y se ajusta a los datos de entrenamiento.  

### 13. Make predictions.  
~~~
    val predictions = model.transform(testData)
~~~
> Se hacen las predicciones del modelo y se ajusta a los datos de prueba.  

### 14. Select example rows to display.  
~~~
    predictions.select("predictedLabel", "label", "features").show(5)
~~~
> Se seleccionan las columnas "predictedLabel", "label" y "features" para desplegar las primeras 5 filas de las predicciones realizadas por el modelo.  

### 15. Select (prediction, true label) and compute test error.  
~~~
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${1.0 - accuracy}")
~~~
> Se calcula el nivel de exactitud del modelo con la función `evaluate`, y usando el resultado se calcula el porcentaje de error con la resta: 1.0 - nivel de exactitud.  

### 16. Print result of Trees using GBT (10).
~~~
    val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
    println(s"Learned classification GBT model:\n ${gbtModel.toDebugString}")

    spark.stop()
  }

main()
~~~
> Se imprimen todos los árboles generados por el modelo, en este caso 10 en total, incluyendo las condiciones y los resultados de las predicciones para cada una de sus ramas.  
> El modelo arrojo un aproximado del 3% de error, lo cual nos deja con un nivel de exactitud del 97% aproximadamente; esto significa que el modelo trabaja muy bien y las predicciones que realiza son bastante confiables. 

