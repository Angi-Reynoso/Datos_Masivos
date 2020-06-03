## Instructions  

Develop the following instructions in Spark with the Scala programming language,
using only documentation from Spark's Machine Learning Mllib library and
Google

**1. Data**
* From the Iris.csv dataset found at https://github.com/jcromerohdz/iris.
* Elaborate the cleanliness of data necessary to be processed by the following algorithm.
* (Important, this cleaning must be by means of Scala script in Spark).

**a. Using Spark's Mllib library the Machine Learning Algorithm called Multilayer Perceptron**  
~~~
//Iniciar sesion
import org.apache.spark.sql.SparkSession
val spar = SparkSession.builder().getOrCreate()

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR) //reducir errores

val df = spark.read.option("header", "true").option("inferSchema","true")csv("iris.csv")
df.na.drop().show()

import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
~~~  

**2. What are the column names?**  
~~~
df.columns
~~~  

**3. How is the scheme?**  
~~~
df.printSchema()
~~~  

**4. Print the first 5 columns.**  
~~~
df.show(5)

df.head(5)
for(row <- df.head(5)){
    println(row)
}
~~~  

**5. Use the describe() method to learn more about the data in the DataFrame.**  
~~~
df.describe().show()
~~~  

**6. Make the corresponding transformation for the categorical data which will be our labels to classify.**  
~~~
val data = df.select(df("species").as("label"), $"sepal_length", $"sepal_width", $"petal_length", $"petal_width")

// Importe VectorAssembler y Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

// Cree un nuevo objeto VectorAssembler llamado assembler para los features
val assembler = new VectorAssembler().setInputCols(Array("sepal_length", "sepal_width","petal_length","petal_width")).setOutputCol("features")
val features = assembler.transform(data)
features.show()

//Importar StringIndexer
// Index labels, adding metadata to the label column.
// Fit on whole dataset to include all labels in index.
import org.apache.spark.ml.feature.StringIndexer
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(features)

//Importar VectorIndexer
// Automatically identify categorical features, and index them.
// Set maxCategories so features with > 4 distinct values are treated as continuous.
import org.apache.spark.ml.feature.VectorIndexer
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(features)

// Split the data into train and test
val splits = features.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)
~~~  

**7. Build the classification models and explain your architecture.**  
~~~
// specify layers for the neural network: input layer of size 4 (features), two intermediate of size 5 and 4
// and output of size 3 (classes)
val layers = Array[Int](4, 5, 4, 3)

// create the trainer and set its parameters
val trainer = new MultilayerPerceptronClassifier()
.setLayers(layers)
.setLabelCol("indexedLabel")
.setFeaturesCol("indexedFeatures")
.setBlockSize(128)
.setSeed(1234L)
.setMaxIter(100)

//Import IndexToString
import org.apache.spark.ml.feature.IndexToString

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString()
.setInputCol("prediction")
.setOutputCol("predictedLabel")
.setLabels(labelIndexer.labels)

//Import pipeline
import org.apache.spark.ml.Pipeline

// Chain indexers and MultilayerPerceptronClassifier in a Pipeline.
val pipeline = new Pipeline()
.setStages(Array(labelIndexer, featureIndexer, trainer, labelConverter))

// train the model
val model = pipeline.fit(train)
~~~  

**8. Print the model results.**  
~~~
// Compute accuracy on the test set
val predictions = model.transform(test)
val predictionAndLabels = predictions.select("prediction", "label")

// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator()
.setLabelCol("indexedLabel")
.setPredictionCol("prediction")
.setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

---------------------------------------
scala> println("Test Error = " + (1.0 - accuracy))
Test Error = 0.039215686274509776
~~~  
> Poner que comprobamos imprimiendo la tabla (2 errores de 60)
