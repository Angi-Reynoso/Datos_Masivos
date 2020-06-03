## Instructions  

Develop the following instructions in Spark with the Scala programming language,
using only documentation from Spark's Machine Learning Mllib library and
Google

**1. Data**
* From the Iris.csv dataset found at https://github.com/jcromerohdz/iris.
* Elaborate the cleanliness of data necessary to be processed by the following algorithm.
* (Important, this cleaning must be by means of Scala script in Spark).

**a. Using Spark's Mllib library the Machine Learning Algorithm called Multilayer Perceptron**  
* Iniciamos una nueva sesión de spark:  
~~~
import org.apache.spark.sql.SparkSession
val spar = SparkSession.builder().getOrCreate()
~~~  
* Utilizamos el siguiente código para reducir los errores durante la ejecución del resto.  
~~~
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
~~~  
* Cargamos el archivo con el dataset: "iris.csv"  
~~~
val df = spark.read.option("header", "true").option("inferSchema","true")csv("iris.csv")
df.na.drop().show()
~~~  
> `df.na.drop().show` sirve para eliminar todos los datos que tenga NA o que sean nulos.  

* Importamos las librerías para poder trabajar con Multilayer Perceptron.  
~~~
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
~~~  

**2. What are the column names?**  
~~~
df.columns
~~~  
> `.columns` sirve para desplegar los nombres de las columnas del dataframe.  

**3. How is the scheme?**  
~~~
df.printSchema()
~~~  
> `.printSchema()` sirve para desplegar las columnas del dataframe junto con su tipo de datos.  

**4. Print the first 5 columns.**  
~~~
df.show(5)

df.head(5)
for(row <- df.head(5)){
    println(row)
}
~~~  
> `df.show(5)` imprime las primeras 5 filas del dataframe en forma de tabla.    
> `df.head(5)` imprime las primeras 5 filas del dataframe en forma de arreglos.  
> El ciclo for se utiliza para desplegar lo mismo que en el codigo anterior, pero con salto de linea en lugar de forma lineal (todo junto).   

**5. Use the describe() method to learn more about the data in the DataFrame.**  
~~~
df.describe().show()
~~~  
> `describe().show()` imprime un resumen de los datos del dataframe (numero total de datos, promedio, desviacion estandar, valores minimo y maximo).  

**6. Make the corresponding transformation for the categorical data which will be our labels to classify.**  
* Se seleccionan las columnas del dataframe y se renombra la de "species" como "label".  
~~~
val data = df.select(df("species").as("label"), $"sepal_length", $"sepal_width", $"petal_length", $"petal_width")
~~~ 
* Importamos las librerias VectorAssembler y Vectors. 
~~~
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
~~~  
* Se crea un nuevo objeto VectorAssembler llamado assembler para guardar el resto de las columnas como features.  
* Se crea la variable features para guardar el dataframe con los cambios anteriores.  
~~~
val assembler = new VectorAssembler().setInputCols(Array("sepal_length", "sepal_width","petal_length","petal_width")).setOutputCol("features")
val features = assembler.transform(data)
~~~  
* Importamos la libreria StringIndexer.  
~~~
import org.apache.spark.ml.feature.StringIndexer
~~~  
* Indexamos la columna label y añadimos metadata, esto es para cambiar los valores tipo texto de las etiquetas por valores numericos.  
* Se ajusta a todo el dataset para incluir todas las etiquetas en el indice.  
~~~
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(features)
~~~  
* Importamos la librería VectorIndexer.  
~~~
import org.apache.spark.ml.feature.VectorIndexer
~~~  
* Se indexa la columna "features" y se establece el numero maximo de categorias a tomar como 4.
* Se ajusta a todo el dataset para incluir todas las etiquetas en el indice.  
~~~
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(features)
~~~  
* Se dividen los datos en datos de entrenamiento (0.6) y de prueba (0.4) con `randomSplit`.  
* Se crean las variables train y test para guardar los datos divididos.  
~~~
val splits = features.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)
~~~  

**7. Build the classification models and explain your architecture.**  
* Specify layers for the neural network: input layer of size 4 (features), two intermediate of size 5 and 4, and output of size 3 (classes).  
~~~
val layers = Array[Int](4, 5, 4, 3)
~~~  
* Create the trainer and set its parameters:  
  * `setLayers` se hace referencia a la variable creada en el paso anterior `layers`.  
  * `setLabelCol` se hace referencia a la columna label indexada `indexedLabel`.  
  * `setFeaturesCol` se hace referencia a la columna features indexada `indexedFeatures`.  
  * `setBlockSize` se establece el tamaño del bloque por defecto en kilobytes (128).  
  * `setSeed` se utiliza para dar aleatoriedad a los datos (1234L).  
  * `setMaxIter` se estable el número máximo de iteraciones a realizar por el modelo (100 es el valor predeterminado).  
~~~
val trainer = new MultilayerPerceptronClassifier()
.setLayers(layers)
.setLabelCol("indexedLabel")
.setFeaturesCol("indexedFeatures")
.setBlockSize(128)
.setSeed(1234L)
.setMaxIter(100)
~~~  
* Importamos la librería IndexToString.  
~~~
import org.apache.spark.ml.feature.IndexToString
~~~  
* Se convierten los valores de las etiquetas indexadas en los de las etiquetas originales, y se les asigna el nombre de "predictedLabel".  
~~~
val labelConverter = new IndexToString()
.setInputCol("prediction")
.setOutputCol("predictedLabel")
.setLabels(labelIndexer.labels)
~~~  
* Importamos la librería para utilizar Pipeline.  
~~~
import org.apache.spark.ml.Pipeline
~~~  
* Se unen los indexadores y el MultilayerPerceptronClassifier (el modelo) en una Pipeline.  
~~~
val pipeline = new Pipeline()
.setStages(Array(labelIndexer, featureIndexer, trainer, labelConverter))
~~~  
* Se entrena el modelo, y se ajusta la Pipeline para trabajar con los datos de entrenamiento.  
~~~
val model = pipeline.fit(train)
~~~  

**8. Print the model results.**  
* Se calcula el nivel de exactitud utilizando los datos de prueba.  
  * Se utiliza `transform` para cambiar el dataframe de train a test.  
~~~
val predictions = model.transform(test)
val predictionAndLabels = predictions.select("prediction", "label")
~~~  
* Se seleccionan las columnas "indexedLabel" y "prediction", y se calcula el porcentaje de error del modelo en base al nivel de exactitud (accuracy).  
* Se utiliza `evaluate` para valorar las predicciones hechas por el modelo.  
* Se imprime en consola el porcentaje de error como el resultado de la resta: `1.0 - accuracy`.  
~~~
val evaluator = new MulticlassClassificationEvaluator()
.setLabelCol("indexedLabel")
.setPredictionCol("prediction")
.setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))
~~~  
* El resultado obtenido fue el siguiente:  
~~~
---------------------------------------
scala> println("Test Error = " + (1.0 - accuracy))
Test Error = 0.039215686274509776
~~~  
> Se obtuvo un porcentaje de error aproximado al 3%, lo cual significa que el modelo posee un nivel de exactitud del 97% (aproximadamente), lo cual es muy bueno.  
> Para corroborar los resultados se desplego en consola la variable `predictionAndLabels`, donde se obtiene una tabla con las 60 predicciones calculadas por el modelo (prediction and label), correspondientes al test set.  
> De los datos desplegados se localizaron unicamente 2 errores, lo cual con respecto al total de 60 predicciones, nos arroja el 3% obtenido en el porcentaje de error.  


