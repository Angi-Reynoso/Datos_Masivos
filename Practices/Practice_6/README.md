# Multilayer Perceptron Classifier

MLPs are forward-directed networks with one or more node layers between the input nodes and the output nodes (Hidden). Each neuron is a perceptron type. Each layer is fully connected to the next layer in the network.
* Perceptron: Artificial neuron or basic unit of inference in the form of a linear discriminator, from which an algorithm is developed capable of generating a criterion to select a sub-group from a larger group of components.

Layers can be classified into three types:
* Input layer: Made up of those neurons that introduce input patterns into the network. No processing occurs in these neurons.
* Hidden layers: Formed by those neurons whose inputs come from previous layers and whose outputs go to neurons from later layers.
* Output layer: Neurons whose output values correspond to the outputs of the entire network.

## Steps:  
### 1. Import libraries.  
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

> An array of numerical type is created to save the values ​​of each layer.  

### 6. Create the trainer and set its parameters.  
~~~
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)
~~~  
> `.setLayers (layers)` is used to indicate the layers, stored in the layers variable.  
> `.setBlockSize (128)` is used to indicate the size of the block in kilobytes (default 128).  
> `.setSeed (1234L)` is used to randomize the data.  
> `.setMaxIter (100)` is used to indicate the maximum number of iterations (default 100).  

### 7. Train the model.  
~~~
    val model = trainer.fit(train)
~~~  
>The model is trained and adjusted to the training data. 

### 8. Compute accuracy on the test set.
~~~
    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
~~~  
> Fit the model to the test data using the `transform` function.
> The columns "prediction" and "label" are selected, and the level of accuracy of the model is calculated.

### 9. Print result.  
~~~
    println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
 spark.stop()
  }

main()
~~~
> The level of accuracy obtained for the model is printed.
> In this case, approximately 90% was obtained, so it can be said that the model works quite well.  

