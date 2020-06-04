## Instructions  
Develop the following instructions in Spark with the Scala programming language, using only documentation from the Mllib library of Spark Machine Learning and Google.  

**1. Data**
* From the Iris.csv dataset found at https://github.com/jcromerohdz/iris.
* Elaborate the cleaning of the necessary data to be processed by the following algorithm.
* (Important, this cleaning must be done using the Scala script in Spark).

**a. Using Spark's Mllib library, the Machine Learning Algorithm called Multilayer Perceptron**  
*  We start a new session of spark;  
~~~
import org.apache.spark.sql.SparkSession
val spar = SparkSession.builder().getOrCreate()
~~~  
* We use the following code to reduce errors while executing the res
~~~
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
~~~  
* We load the file with the dataset: "iris.csv"  
~~~
val df = spark.read.option("header", "true").option("inferSchema","true")csv("iris.csv")
df.na.drop().show()
~~~  
> `df.na.drop().show` It's used to delete all the data that has NA or that is null.
* We import the libraries to be able to work with Multilayer Perceptron.
~~~
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
~~~  

**2. What are the column names? **
~~~
df.columns
~~~  
> `.columns` it´s used to display the names of the columns of the dataframe.

**3. How is the scheme? ** 
~~~
df.printSchema()
~~~  
> `.printSchema()` it's used to display the columns of the dataframe together with their data type.  

**4. Print the first 5 columns. **
~~~
df.show(5)

df.head(5)
for(row <- df.head(5)){
    println(row)
}
~~~  
> `df.show (5)` prints the first 5 rows of the dataframe in table form.
> `df.head (5)` prints the first 5 rows of the dataframe as arrays.
> The for loop is used to display the same as in the previous code, but with a line break instead of linear (all together).

**5. Use the describe () method to get more information about the data in the DataFrame. **
~~~
df.describe().show()
~~~  
> `describe (). show ()` prints a summary of the data in the dataframe (total number of data, average, standard deviation, minimum and maximum values).

** 6. Perform the corresponding transformation for the categorical data that will be our labels to classify. **
* The columns of the dataframe are selected and the one of "species" is renamed as "label".
~~~
val data = df.select(df("species").as("label"), $"sepal_length", $"sepal_width", $"petal_length", $"petal_width")
~~~ 
* Import the VectorAssembler and Vectors libraries.
~~~
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
~~~  
* A new VectorAssembler object called assembler is created to save the rest of the columns as features.
* The variable features is created to save the dataframe with the previous changes.
~~~
val assembler = new VectorAssembler().setInputCols(Array("sepal_length", "sepal_width","petal_length","petal_width")).setOutputCol("features")
val features = assembler.transform(data)
~~~  
* The StringIndexer library was imported
~~~
import org.apache.spark.ml.feature.StringIndexer
~~~  
* The label column was indexed and we added metadata, this is to change the text-type values ​​of the labels by numerical values.
* Fits the entire dataset to include all the tags in the index.
~~~
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(features)
~~~  
* The VectorIndexer library was imported.  
~~~
import org.apache.spark.ml.feature.VectorIndexer
~~~  
* The "features" column is indexed and the maximum number of categories to take as 4 is established.
* Fits the entire dataset to include all the tags in the index.
~~~
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(features)
~~~  
* Divide data into training (0.6) and test (0.4) data with `randomSplit`.
* The train and test variables are created to save the divided data.
~~~
val splits = features.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)
~~~  

** 7. Create the classification models and explain their architecture. **
* Layers are specified for the neural network: input layer of size 4 (characteristics), two intermediate layers of size 5 and 4, and output of size 3 (classes).
~~~
val layers = Array[Int](4, 5, 4, 3)
~~~  
* The traainer is created and its parameters are established:
  * `setLayers` refers to the variable created in the previous step` layers`.
  * `setLabelCol` is referenced by the indexed label column` indexedLabel`.
  * `setFeaturesCol` refers to the indexed features column` indexedFeatures`.
  * `setBlockSize` sets the default block size in kilobytes (128).
  * `setSeed` is used to randomize the data (1234L).
  * `setMaxIter` sets the maximum number of iterations to be performed by the model (100 is the default).  
~~~
val trainer = new MultilayerPerceptronClassifier()
.setLayers(layers)
.setLabelCol("indexedLabel")
.setFeaturesCol("indexedFeatures")
.setBlockSize(128)
.setSeed(1234L)
.setMaxIter(100)
~~~  
* We import the IndexToString library.  
~~~
import org.apache.spark.ml.feature.IndexToString
~~~  
* The values of the indexed tags are converted to those of the original tags, and are named "predictedLabel".
~~~
val labelConverter = new IndexToString()
.setInputCol("prediction")
.setOutputCol("predictedLabel")
.setLabels(labelIndexer.labels)
~~~  
* The library was imported to use Pipeline.  
~~~
import org.apache.spark.ml.Pipeline
~~~  
* The indexers and the MultilayerPerceptronClassifier (the model) are joined in a Pipeline. 
~~~
val pipeline = new Pipeline()
.setStages(Array(labelIndexer, featureIndexer, trainer, labelConverter))
~~~  
* The model is trained, and the Pipeline is adjusted to work with the training data.
~~~
val model = pipeline.fit(train)
~~~  

**8. Print the model results.**  
* The level of accuracy is calculated using the test data.
  * Use `transform` to change the dataframe from train to test. 
~~~
val predictions = model.transform(test)
val predictionAndLabels = predictions.select("prediction", "label")
~~~  
* The columns "indexedLabel" and "prediction" are selected, and the percentage error of the model is calculated based on the level of accuracy.
* `Evaluate` is used to evaluate the predictions made by the model.
* The error percentage is printed in the console as the result of the subtraction: `1.0 - accuracy`.  
~~~
val evaluator = new MulticlassClassificationEvaluator()
.setLabelCol("indexedLabel")
.setPredictionCol("prediction")
.setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))
~~~  
The following result was the one obtained:  
~~~
---------------------------------------
scala> println("Test Error = " + (1.0 - accuracy))
Test Error = 0.039215686274509776
~~~  
> An error rate of approximately 3% was obtained, which means that the model has an accuracy level of 97% (approximately), which is very good.
> To corroborate the results, the variable `predictionAndLabels` is displayed in the console, where a table is obtained with the 60 predictions calculated by the model (prediction and label), corresponding to the test set.
> From the displayed data, only 2 errors were found, which with respect to the total of 60 predictions, gives us the 3% obtained in the error percentage.


