# ÍNDICE  
* [Algorithm 1: SVM](https://github.com/Angi-Reynoso/Datos_Masivos/blob/Unidad_4/Unit_4/README.md#)  
* [Algorithm 2: Decision Tree](https://github.com/Angi-Reynoso/Datos_Masivos/blob/Unidad_4/Unit_4/README.md#)  
* [Algorithm 3: Logistic Regression](https://github.com/Angi-Reynoso/Datos_Masivos/blob/Unidad_4/Unit_4/README.md#)  
* [Algorithm 4: Multilayer Perceptron](https://github.com/Angi-Reynoso/Datos_Masivos/blob/Unidad_4/Unit_4/README.md#)  
* [Documentation](https://github.com/Angi-Reynoso/Datos_Masivos/blob/Unidad_4/Unit_4/#)  

---

<br>

## Support Vector Machine (SVM)  
**1. Iniciar nueva sesión de spark.**  
~~~
import org.apache.spark.sql.SparkSession
val spar = SparkSession.builder().getOrCreate()
~~~

**2. Código para reducir errores durante la ejecución.**  
~~~
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
~~~

**3. Cargar el archivo con el dataset.**  
~~~
val dataset = spark.read.option("header","true").option("inferSchema","true").csv("bank-additional-full.csv")
~~~

**4. Importar las librerías para trabajar con SVM.**  
~~~
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
~~~

**5. Transformación de los datos.**  
~~~
val coly = when($"y".contains("yes"), 1.0).otherwise(0.0)
val df = dataset.withColumn("y", coly)
~~~
> Se buscan los valores iguales a "yes" de la columna "y", y se sustituyen por 1, o en el caso contrario por 0.  
> Una vez cambiados los valores, estos se vuelven a insertar en la columna "y".  

**6. Transformación para los datos categoricos (etiquetas a clasificar).**  
~~~
val data = df.select(df("y").as("label"), $"age",$"duration",$"campaign",$"pdays",$"previous",$"emp_var_rate",$"cons_price_idx",$"cons_conf_idx",$"euribor3m",$"nr_employed")
~~~
> Se seleccionan las columnas del dataframe y se renombra la columna "y" como "label".  

**7. Importar las librerias VectorAssembler y Vectors.**  
~~~
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
~~~

**8. Crear nuevo objeto VectorAssembler.**  
~~~
val assembler = new VectorAssembler().setInputCols(Array("age","duration","campaign","pdays","previous","emp_var_rate","cons_price_idx","cons_conf_idx","euribor3m","nr_employed")).setOutputCol("features")
val features = assembler.transform(data)
~~~
> Se crea un nuevo objeto VectorAssembler llamado assembler para guardar el resto de las columnas como features.
> Se crea la variable features para guardar el dataframe con los cambios anteriores.  

**9. Importar la libreria StringIndexer.**  
~~~
import org.apache.spark.ml.feature.StringIndexer
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(features)
~~~
> Indexamos la columna label y añadimos metadata, esto es para cambiar los valores tipo texto de las etiquetas por valores numericos.  
> También se hace un ajuste a todo el dataset para incluir todas las etiquetas en el indice.  

**10. Importar la librería VectorIndexer.**  
~~~
import org.apache.spark.ml.feature.VectorIndexer
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(features)
~~~
> Se indexa la columna "features" y se establece el número máximo de categorias a tomar como 4.  
> También se hace un ajuste a todo el dataset para incluir todas las etiquetas en el indice.  

**11. Dividir datos.**  
~~~
val splits = features.randomSplit(Array(0.7, 0.3), seed = 1234L)
val train = splits(0)
val test = splits(1)
~~~
> Se dividen los datos en datos de entrenamiento (0.7) y de prueba (0.3) con randomSplit.  
> Se crean las variables train y test para guardar los datos divididos.  

**12. Tiempo de ejecución.**  
~~~
val t1 = System.nanoTime
~~~
> System.nanoTime 

**13. Crear modelo.**  
~~~
val lsvc = new LinearSVC().setMaxIter(100).setRegParam(0.1)
~~~
> Se crea el modelo (lsvc) y se establecen sus parámetros.  
> * setMaxIter = número máximo de iteraciones a realizar por el modelo.  
> * setRegParam = .  

**14. Ajustar el modelo.**  
~~~
val lsvcModel = lsvc.fit(test)
~~~
> Se ajusta el modelo para que trabaje con los datos de prueba y, posteriormente se puedan obtener los coeficientes y la intercepción.  

**15. Importar la librería para utilizar Pipeline.**  
~~~
import org.apache.spark.ml.Pipeline
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, lsvc))
val model = pipeline.fit(train)
~~~
> Se unen los indexadores y el modelo en una Pipeline.  
> Se entrena el modelo, y se ajusta la Pipeline para trabajar con los datos de entrenamiento.  

**16. Imprimir los resultados del modelo.**  
~~~
val predictions = model.transform(test)
val predictionAndLabels = predictions.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator()
.setLabelCol("indexedLabel")
.setPredictionCol("prediction")
.setMetricName("accuracy")
~~~
> Se utiliza transform para cambiar los datos a utilizar, de train a test.  
> Se seleccionan las columnas de prediccion y "label", y se guardan dentro de la variable "predictionAndLabels".  
> Se crea un evaluador con la función MulticlassClassificationEvaluator(), en donde se seleccionan las columnas "indexedLabel" y "prediction", y se calcula el nivel de exactitud (accuracy) del modelo.  

~~~
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))
println("Accuracy = " + accuracy)
~~~
> Se utiliza evaluate para valorar las predicciones hechas por el modelo.  
> Se imprime en consola el porcentaje de error como resultado de la resta: 1.0 - accuracy; así como el valor obtenido para el accuracy.  

~~~
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")
~~~
> Se imprimen los coeficientes y la intercepción obtenidas para el modelo.  

**17. Imprimir tiempo total de ejecución.**  
~~~
val duration = (System.nanoTime - t1) / 1e9d
~~~
> 1e9d

#### Resultados: 
* Porcentaje de error promedio = 0.10981912144702843 ≈ 11%  
* Nivel de exactitud promedio = 0.8901808785529716 ≈ 89%  
* Tiempo de ejecución promedio = 367.97762408  


---

<br>

## Decision Tree 

#### Resultados: 
* Porcentaje de error promedio = 0.08599806201550386 ≈ 9%  
* Nivel de exactitud promedio = 0.9140019379844961 ≈ 91%  
* Tiempo de ejecución promedio = 7.9979738504  

---

<br>

## Logistic Regression  

#### Resultados: 
* Porcentaje de error promedio = 0.0900355297157 ≈ 9%  
* Nivel de exactitud promedio = 0.9099644702842378 ≈ 91%  
* Tiempo de ejecución promedio = 9.6360462944  

---

<br>

## Multilayer Perceptron  

#### Resultados: 
* Porcentaje de error promedio = 0.10981912144702843 ≈ 11%  
* Nivel de exactitud promedio = 0.8901808785529716 ≈ 89%  
* Tiempo de ejecución promedio = 25.0155136536  
