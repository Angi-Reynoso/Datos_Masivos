# Multilayer Perceptron Classifier

MLPs are forward-directed networks with one or more node layers between the input nodes and the output nodes (Hidden). Each neuron is a perceptron type. Each layer is fully connected to the next layer in the network.
* Perceptron: Artificial neuron or basic unit of inference in the form of a linear discriminator, from which an algorithm is developed capable of generating a criterion to select a sub-group from a larger group of components.

Layers can be classified into three types:
* Input layer: Made up of those neurons that introduce input patterns into the network. No processing occurs in these neurons.
* Hidden layers: Formed by those neurons whose inputs come from previous layers and whose outputs go to neurons from later layers.
* Output layer: Neurons whose output values correspond to the outputs of the entire network.

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
> A new spark session is created, and the application is named "MultilayerPerceptronClassifierExample  

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

> The dataframe is divided into 60% for training and 40% for testing.

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

