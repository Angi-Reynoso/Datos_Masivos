# Index
* [Algorithm 1: SVM](https://github.com/Angi-Reynoso/Datos_Masivos/blob/Unidad_4/Unit_4/README.md#support-vector-machine-svm)  
* [Algorithm 2: Decision Tree](https://github.com/Angi-Reynoso/Datos_Masivos/blob/Unidad_4/Unit_4/README.md#decision-tree)  
* [Algorithm 3: Logistic Regression](https://github.com/Angi-Reynoso/Datos_Masivos/blob/Unidad_4/Unit_4/README.md#logistic-regression)  
* [Algorithm 4: Multilayer Perceptron](https://github.com/Angi-Reynoso/Datos_Masivos/blob/Unidad_4/Unit_4/README.md#multilayer-perceptron)  
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
> Se utilizan la opciones "header" e "inferSchema" para obtener el título y tipo de dato de cada columna en el dataset.  

**4. Importar las librerías para trabajar con SVM.**  
~~~
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
~~~
> La primera librería sí corresponde y es necesaria para la implementación del modelo SVM, no obstante, la segunda librería se utiliza para evaluar el desempeño de los modelos de clasificación, ya sea que se trate de SVM o algún otro algoritmo.  

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
> Indexamos la columna label y añadimos metadata, esto es para cambiar los valores tipo texto de las etiquetas por valores numéricos.  
> También se hace un ajuste a todo el dataset para incluir todas las etiquetas en el índice.  

**10. Importar la librería VectorIndexer.**  
~~~
import org.apache.spark.ml.feature.VectorIndexer
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(features)
~~~
> Se indexa la columna "features" y se establece el número máximo de categorias a tomar como 4.  
> También se hace un ajuste a todo el dataset para incluir todas las etiquetas en el índice.  

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
> System.nanoTime se utiliza para medir la diferencia de tiempo transcurrido.  
> Esta línea se coloca antes de que comience el código del cual se desea tomar el tiempo de ejecución.  

**13. Crear modelo.**  
~~~
val lsvc = new LinearSVC().setMaxIter(100).setRegParam(0.1)
~~~
> Se crea el modelo (lsvc) y se establecen sus parámetros.  
> * setMaxIter = número máximo de iteraciones a realizar por el modelo.  
> * setRegParam = establecer el parámetro de regularización. El valor predeterminado es 0.0.  

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
> Esta línea se coloca al final del código del cual se desea tomar el tiempo de ejecución.  
> El tiempo se obtiene como resulado de la resta: tiempo actual en nanosegundos (System.nanoTime) menos el tiempo al iniciar el código (en este caso la variable denominada t1).  
> Como el resultado se encuentra en nanosegundos se utiliza una división entre 1e9d para obtener el tiempo en segundos.  
> * `1e9d` = operación 10^9, y la "d" es para indicar que el resultado sea de tipo double.  

#### Resultados: 
* Porcentaje de error promedio = 0.10981912144702843 ≈ 11%  
* Nivel de exactitud promedio = 0.8901808785529716 ≈ 89%  
* Tiempo de ejecución promedio = 367.97762408  


---

<br>

## Decision Tree 
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
> Se utilizan la opciones "header" e "inferSchema" para obtener el título y tipo de dato de cada columna en el dataset.  

**4. Importar las librerías para trabajar con Decision Tree.**  
~~~
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

~~~
> Las primeras dos librerías sí corresponden y son necesarias para la implementación del modelo de Decision Tree, no obstante, la tercera librería se utiliza para evaluar el desempeño de los modelos de clasificación, ya sea que se trate de Decision Tree o algún otro algoritmo.  

**5. Transformación para los datos categoricos (etiquetas a clasificar).**  
~~~
val data = df.select(df("y").as("label"), $"age",$"duration",$"campaign",$"pdays",$"previous",$"emp_var_rate",$"cons_price_idx",$"cons_conf_idx",$"euribor3m",$"nr_employed")
~~~
> Se seleccionan las columnas del dataframe y se renombra la columna "y" como "label".  

**6. Importar las librerias VectorAssembler y Vectors.**  
~~~
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
~~~

**7. Crear nuevo objeto VectorAssembler.**  
~~~
val assembler = new VectorAssembler().setInputCols(Array("age","duration","campaign","pdays","previous","emp_var_rate","cons_price_idx","cons_conf_idx","euribor3m","nr_employed")).setOutputCol("features")
val df = assembler.transform(data)
~~~
> Se crea un nuevo objeto VectorAssembler llamado assembler para guardar el resto de las columnas como features.
> Se crea la variable df para guardar el dataframe con los cambios anteriores.  

**8. Importar la libreria StringIndexer.**  
~~~
import org.apache.spark.ml.feature.StringIndexer
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(features)
~~~
> Indexamos la columna label y añadimos metadata, esto es para cambiar los valores tipo texto de las etiquetas por valores numéricos.  
> También se hace un ajuste a todo el dataset para incluir todas las etiquetas en el índice.  

**9. Importar la librería VectorIndexer.**  
~~~
import org.apache.spark.ml.feature.VectorIndexer
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(features)
~~~
> Se indexa la columna "features" y se establece el número máximo de categorias a tomar como 4.  
> También se hace un ajuste a todo el dataset para incluir todas las etiquetas en el índice.  

**10. Dividir datos.**  
~~~
val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3), seed = 1234L)
~~~
> Se dividen los datos en datos de entrenamiento (0.7) y de prueba (0.3) con randomSplit.  
> Los datos divididos se guardan como trainingData y testData.  

**11. Tiempo de ejecución.**  
~~~
val t1 = System.nanoTime
~~~
> System.nanoTime se utiliza para medir la diferencia de tiempo transcurrido.  
> Esta línea se coloca antes de que comience el código del cual se desea tomar el tiempo de ejecución.  

**12. Crear modelo.**  
~~~
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
~~~
> Se crea el modelo (lsvc) y se establecen sus parámetros.  
> * setLabelCol = columna label indexada indexedLabel.  
> * setFeaturesCol = columna features indexada indexedFeatures.  

**13. Importar la librería IndexToString.**  
~~~
import org.apache.spark.ml.feature.IndexToString
val labelConverter = new IndexToString()
.setInputCol("prediction")
.setOutputCol("predictedLabel")
.setLabels(labelIndexer.labels) 
~~~
> Se convierten los valores de las etiquetas indexadas en los de las etiquetas originales, y se les asigna el nombre de "predictedLabel".  

**14. Importar la librería para utilizar Pipeline.**  
~~~
import org.apache.spark.ml.Pipeline
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
val model = pipeline.fit(train)
~~~
> Se unen los indexadores y el modelo en una Pipeline.  
> Se entrena el modelo, y se ajusta la Pipeline para trabajar con los datos de entrenamiento.  

**15. Imprimir los resultados del modelo.**  
~~~
val predictions = model.transform(testData)
predictions.select("predictedLabel", "label", "features").show(5)
val evaluator = new MulticlassClassificationEvaluator()
.setLabelCol("indexedLabel")
.setPredictionCol("prediction")
.setMetricName("accuracy")
~~~
> Se utiliza transform para cambiar los datos a utilizar, de train a test.  
> Se seleccionan 5 filas de ejemplo para mostrar en consola.  
> Se crea un evaluador con la función MulticlassClassificationEvaluator(), en donde se seleccionan las columnas "indexedLabel" y "prediction", y se calcula el nivel de exactitud (accuracy) del modelo.  

~~~
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))
~~~
> Se utiliza evaluate para valorar las predicciones hechas por el modelo.  
> Se imprime en consola el porcentaje de error como resultado de la resta: 1.0 - accuracy.  

~~~
val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
~~~
> Se imprime el árbol generado por el modelo.  

**16. Imprimir tiempo total de ejecución.**  
~~~
val duration = (System.nanoTime - t1) / 1e9d
~~~
> Esta línea se coloca al final del código del cual se desea tomar el tiempo de ejecución.  
> El tiempo se obtiene como resulado de la resta: tiempo actual en nanosegundos (System.nanoTime) menos el tiempo al iniciar el código (en este caso la variable denominada t1).  
> Como el resultado se encuentra en nanosegundos se utiliza una división entre 1e9d para obtener el tiempo en segundos.  
> * `1e9d` = operación 10^9, y la "d" es para indicar que el resultado sea de tipo double.  

#### Resultados: 
* Porcentaje de error promedio = 0.08599806201550386 ≈ 9%  
* Nivel de exactitud promedio = 0.9140019379844961 ≈ 91%  
* Tiempo de ejecución promedio = 7.9979738504  


---

<br>

## Logistic Regression  
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
> Se utilizan la opciones "header" e "inferSchema" para obtener el título y tipo de dato de cada columna en el dataset.  

**4. Importar las librerías para trabajar con SVM.**  
~~~
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
~~~
> La primera librería sí corresponde y es necesaria para la implementación del modelo SVM, no obstante, la segunda librería se utiliza para evaluar el desempeño de los modelos de clasificación, ya sea que se trate de SVM o algún otro algoritmo.  

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
> Indexamos la columna label y añadimos metadata, esto es para cambiar los valores tipo texto de las etiquetas por valores numéricos.  
> También se hace un ajuste a todo el dataset para incluir todas las etiquetas en el índice.  

**10. Importar la librería VectorIndexer.**  
~~~
import org.apache.spark.ml.feature.VectorIndexer
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(features)
~~~
> Se indexa la columna "features" y se establece el número máximo de categorias a tomar como 4.  
> También se hace un ajuste a todo el dataset para incluir todas las etiquetas en el índice.  

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
> System.nanoTime se utiliza para medir la diferencia de tiempo transcurrido.  
> Esta línea se coloca antes de que comience el código del cual se desea tomar el tiempo de ejecución.  

**13. Crear modelo.**  
~~~
val lsvc = new LinearSVC().setMaxIter(100).setRegParam(0.1)
~~~
> Se crea el modelo (lsvc) y se establecen sus parámetros.  
> * setMaxIter = número máximo de iteraciones a realizar por el modelo.  
> * setRegParam = establecer el parámetro de regularización. El valor predeterminado es 0.0.  

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
> Esta línea se coloca al final del código del cual se desea tomar el tiempo de ejecución.  
> El tiempo se obtiene como resulado de la resta: tiempo actual en nanosegundos (System.nanoTime) menos el tiempo al iniciar el código (en este caso la variable denominada t1).  
> Como el resultado se encuentra en nanosegundos se utiliza una división entre 1e9d para obtener el tiempo en segundos.  
> * `1e9d` = operación 10^9, y la "d" es para indicar que el resultado sea de tipo double.  

#### Resultados: 
* Porcentaje de error promedio = 0.0900355297157 ≈ 9%  
* Nivel de exactitud promedio = 0.9099644702842378 ≈ 91%  
* Tiempo de ejecución promedio = 9.6360462944  


---

<br>

## Multilayer Perceptron  
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
> Se utilizan la opciones "header" e "inferSchema" para obtener el título y tipo de dato de cada columna en el dataset.  

**4. Importar las librerías para trabajar con SVM.**  
~~~
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
~~~
> La primera librería sí corresponde y es necesaria para la implementación del modelo SVM, no obstante, la segunda librería se utiliza para evaluar el desempeño de los modelos de clasificación, ya sea que se trate de SVM o algún otro algoritmo.  

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
> Indexamos la columna label y añadimos metadata, esto es para cambiar los valores tipo texto de las etiquetas por valores numéricos.  
> También se hace un ajuste a todo el dataset para incluir todas las etiquetas en el índice.  

**10. Importar la librería VectorIndexer.**  
~~~
import org.apache.spark.ml.feature.VectorIndexer
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(features)
~~~
> Se indexa la columna "features" y se establece el número máximo de categorias a tomar como 4.  
> También se hace un ajuste a todo el dataset para incluir todas las etiquetas en el índice.  

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
> System.nanoTime se utiliza para medir la diferencia de tiempo transcurrido.  
> Esta línea se coloca antes de que comience el código del cual se desea tomar el tiempo de ejecución.  

**13. Crear modelo.**  
~~~
val lsvc = new LinearSVC().setMaxIter(100).setRegParam(0.1)
~~~
> Se crea el modelo (lsvc) y se establecen sus parámetros.  
> * setMaxIter = número máximo de iteraciones a realizar por el modelo.  
> * setRegParam = establecer el parámetro de regularización. El valor predeterminado es 0.0.  

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
> Esta línea se coloca al final del código del cual se desea tomar el tiempo de ejecución.  
> El tiempo se obtiene como resulado de la resta: tiempo actual en nanosegundos (System.nanoTime) menos el tiempo al iniciar el código (en este caso la variable denominada t1).  
> Como el resultado se encuentra en nanosegundos se utiliza una división entre 1e9d para obtener el tiempo en segundos.  
> * `1e9d` = operación 10^9, y la "d" es para indicar que el resultado sea de tipo double.  

#### Resultados: 
* Porcentaje de error promedio = 0.10981912144702843 ≈ 11%  
* Nivel de exactitud promedio = 0.8901808785529716 ≈ 89%  
* Tiempo de ejecución promedio = 25.0155136536  
