# Multilayer Perceptron Classifier
Los MLP son redes dirigidas hacia adelante con una o más capas de nodos entre los nodos de entrada y los nodos de salida (Ocultas). Cada neurona es tipo perceptrón. Cada capa está completamente conectada a la siguiente capa en la red.  
* Perceptrón: Neurona artificial o unidad básica de inferencia en forma de discriminador lineal, a partir de lo cual se desarrolla un algoritmo capaz de generar un criterio para seleccionar un sub-grupo a partir de un grupo de componentes más grande.  

Las capas pueden clasificarse en tres tipos:  
* Capa de entrada: Constituida por aquellas neuronas que introducen los patrones de entrada en la red. En estas neuronas no se produce procesamiento.  
* Capas ocultas: Formada por aquellas neuronas cuyas entradas provienen de capas anteriores y cuyas salidas pasan a neuronas de capas posteriores.  
* Capa de salida: Neuronas cuyos valores de salida se corresponden con las salidas de toda la red.  


## Steps:  
### 1. Import libraries and packages.  
~~~
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
~~~  

### 2. Import a Spark Session.  
~~~
import org.apache.spark.sql.SparkSession

def main(): Unit = {
    val spark = SparkSession.builder.appName("MultilayerPerceptronClassifierExample").getOrCreate()
~~~  
> Se crea una nueva sesión spark, y se nombra a la aplicación como "MultilayerPerceptronClassifierExample".  

### 3. Load the data stored in LIBSVM format as a DataFrame.
~~~
    val data = spark.read.format("libsvm").load("sample_multiclass_classification_data.txt")
~~~  

### 4. Split the data into train and test.
~~~
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val train = splits(0)
    val test = splits(1)
~~~  
> Se divide el dataframe en 60% para entrenamiento y 40% para prueba.  

### 5. Specify layers for the neural network.  
* Input layer of size 4 (features).  
* Two intermediate of size 5 and 4.  
* Output of size 3 (classes).  
~~~
    val layers = Array[Int](4, 5, 4, 3)
~~~  
> Se crea un arreglo de tipo numerico para guardar los valores de cada capa.  

### 6. Create the trainer and set its parameters.  
~~~
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)
~~~  
> `.setLayers(layers)` se usa para indicar las capas, guardadas en la variable layers.  
> `.setBlockSize(128)` se usa para indicar el tamaño del bloque en kilobytes (por defecto 128).  
> `.setSeed(1234L)` se usa para dar aletoriedad a los datos.  
> `.setMaxIter(100)` se usa para indicar el numero maximo de iteraciones (por defecto 100).  

### 7. Train the model.  
~~~
    val model = trainer.fit(train)
~~~  
> Se entrena el modelo y se ajusta a los datos de entrenamiento.  

### 8. Compute accuracy on the test set.
~~~
    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
~~~  
> Se ajusta el modelo a los datos de prueba usando la función `transform`.  
> Se seleccionan las columnas "prediction" y "label", y se calcula el nivel de exactitud del modelo.  

### 9. Print result.  
~~~
    println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
 spark.stop()
  }

main()
~~~
> Se imprime el nivel de exactitud obtenido para el modelo.  
> En este caso se obtuvo un aproximado del 90%, por lo cual se puede decir que el modelo funciona bastante bien.  

