# Index
* [Algorithm 1: SVM](https://github.com/Angi-Reynoso/Datos_Masivos/blob/Unidad_4/Unit_4/README.md#support-vector-machine-svm)  
* [Algorithm 2: Decision Tree](https://github.com/Angi-Reynoso/Datos_Masivos/blob/Unidad_4/Unit_4/README.md#decision-tree)  
* [Algorithm 3: Logistic Regression](https://github.com/Angi-Reynoso/Datos_Masivos/blob/Unidad_4/Unit_4/README.md#logistic-regression)  
* [Algorithm 4: Multilayer Perceptron](https://github.com/Angi-Reynoso/Datos_Masivos/blob/Unidad_4/Unit_4/README.md#multilayer-perceptron)  
* [Documentation](https://github.com/Angi-Reynoso/Datos_Masivos/blob/Unidad_4/Unit_4/#)  

---

<br>

## Support Vector Machine (SVM)  
**1. Start new spark session.**
~~~
import org.apache.spark.sql.SparkSession
val spar = SparkSession.builder().getOrCreate()
~~~

**2. Code to reduce errors during execution.**
~~~
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
~~~

**3. Upload the file with the dataset.**
~~~
val dataset = spark.read.option("header","true").option("inferSchema","true").csv("bank-additional-full.csv")
~~~
> The "header" and "inferSchema" options are used to obtain the title and data type of each column in the dataset.

**4. Import libraries to work with SVM.** 
~~~
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
~~~
> The first library does correspond and is necessary for the implementation of the SVM model, however, the second library is used to evaluate the precision of the classification models, whether it is SVM or some other algorithm.

**5. Data transformation.** 
~~~
val coly = when($"y".contains("yes"), 1.0).otherwise(0.0)
val df = dataset.withColumn("y", coly)
~~~
> The values equal to "yes" of the column "y" are searched, and are replaced by 1, or otherwise by 0.
> Once the values have been changed, they are inserted again in the "y" column.

**6. Transformation for categorical data (labels to classify)**
~~~
val data = df.select(df("y").as("label"), $"age",$"duration",$"campaign",$"pdays",$"previous",$"emp_var_rate",$"cons_price_idx",$"cons_conf_idx",$"euribor3m",$"nr_employed")
~~~
> The columns of the dataframe are selected and the column "y" is renamed as "label".

**7. Import the VectorAssembler and Vectors libraries.**
~~~
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
~~~

**8. Create new VectorAssembler object.**
~~~
val assembler = new VectorAssembler().setInputCols(Array("age","duration","campaign","pdays","previous","emp_var_rate","cons_price_idx","cons_conf_idx","euribor3m","nr_employed")).setOutputCol("features")
val features = assembler.transform(data)
~~~
> A new VectorAssembler object called assembler is created to save the rest of the columns as features.
> The variable features is created to save the dataframe with the previous changes.

**9. Import the StringIndexer library.** 
~~~
import org.apache.spark.ml.feature.StringIndexer
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(features)
~~~
> The label column was indexed and the metadata was added, this is to change the text-type values ​​of the labels by numerical values.
> An adjustment was also made to the entire dataset to include all the labels in the index.  

**10. Import the VectorIndexer library.**
~~~
import org.apache.spark.ml.feature.VectorIndexer
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(features)
~~~
> The "features" column is indexed and the maximum number of categories to take as 4 is established.
> An adjustment is also made to the entire dataset to include all the labels in the index.

**11. Split data.** 
~~~
val splits = features.randomSplit(Array(0.7, 0.3), seed = 1234L)
val train = splits(0)
val test = splits(1)
~~~
> The data is divided into training data (0.7) and test data (0.3) with randomSplit.
> The train and test variables are created to save the divided data.

**12. Execution time.**  
~~~
val t1 = System.nanoTime
~~~
> System.nanoTime se utiliza para medir la diferencia de tiempo transcurrido.  
> This line is placed before the code from which you want to take the runtime begins.  

**13. Create model.**
~~~
val lsvc = new LinearSVC().setMaxIter(100).setRegParam(0.1)
~~~
> The model (lsvc) is created and its parameters are established.
> * setMaxIter = maximum number of iterations to be made by the model.
> * setRegParam = set the regularization parameter. The default value is 0.0.

**14. Fit the model.**
~~~
val lsvcModel = lsvc.fit(test)
~~~
> The model is adjusted to work with the test data and, subsequently, the coefficients and interception can be obtained.

**15. Import the library to use Pipeline.** 
~~~
import org.apache.spark.ml.Pipeline
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, lsvc))
val model = pipeline.fit(train)
~~~
> The indexers and the model are joined in a Pipeline.
> The model is trained, and the Pipeline is adjusted to work with the training data.

**16. Print the model results.**  
~~~
val predictions = model.transform(test)
val predictionAndLabels = predictions.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator()
.setLabelCol("indexedLabel")
.setPredictionCol("prediction")
.setMetricName("accuracy")
~~~
> Transform is used to change the data to be used, from train to test.
> The prediction and "label" columns are selected, and are stored inside the "predictionAndLabels" variable.
> An evaluator is created with the MulticlassClassificationEvaluator () function, where the columns "indexedLabel" and "prediction" are selected, and the level of accuracy of the model is calculated. 

~~~
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))
println("Accuracy = " + accuracy)
~~~
> Evaluate is used to evaluate the predictions made by the model.
> The error percentage is printed in the console as a result of the subtraction: 1.0 - accuracy; as well as the value obtained for the accuracy.

~~~
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")
~~~
> The coefficients and interception obtained for the model are printed.

**17. Print total execution time.** 
~~~
val duration = (System.nanoTime - t1) / 1e9d
~~~
> This line is placed at the end of the code from which you want to take the runtime.
> The time is obtained as a result of the subtraction: current time in nanoseconds (System.nanoTime) minus the time when starting the code (in this case the variable named t1).
> Since the result is in nanoseconds, a division between 1e9d is used to obtain the time in seconds.
> * `1e9d` = operation 10 ^ 9, and the" d "is to indicate that the result is of type double.

#### Results:
* Average error percentage = 0.10981912144702843 ≈ 11%
* Average accuracy level = 0.8901808785529716 ≈ 89%
* Average execution time = 367.97762408

---

<br>

## Decision Tree 
**1. Start a new session of spark.**
~~~
import org.apache.spark.sql.SparkSession
val spar = SparkSession.builder().getOrCreate()
~~~

**2. Code to reduce errors during execution.**
~~~
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
~~~

**3. Upload the file with the dataset.**
~~~
val dataset = spark.read.option("header","true").option("inferSchema","true").csv("bank-additional-full.csv")
~~~
> The "header" and "inferSchema" options are used to obtain the title and data type of each column in the dataset.

**4. Import libraries to work with Decision Tree.**  
~~~
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

~~~
> The first two libraries do correspond and are necessary for the implementation of the Decision Tree model, however, the third library is used to evaluate the precision of the classification models, whether it is Decision Tree or some other algorithm.

**5. Transformation for categorical data (labels to classify)**
~~~
val data = dataset.select(dataset("y").as("label"), $"age",$"duration",$"campaign",$"pdays",$"previous",$"emp_var_rate",$"cons_price_idx",$"cons_conf_idx",$"euribor3m",$"nr_employed")
~~~
> The columns of the dataframe are selected and the column "y" is renamed as "label".

**6. Import the VectorAssembler and Vectors libraries.**
~~~
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
~~~

**7. Create new VectorAssembler object.**  
~~~
val assembler = new VectorAssembler().setInputCols(Array("age","duration","campaign","pdays","previous","emp_var_rate","cons_price_idx","cons_conf_idx","euribor3m","nr_employed")).setOutputCol("features")
val df = assembler.transform(data)
~~~
> A new VectorAssembler object called assembler is created to save the rest of the columns as features.
> The df variable is created to save the dataframe with the previous changes.

**8. Import the StringIndexer library.**
~~~
import org.apache.spark.ml.feature.StringIndexer
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(df)
~~~
> The label column is indexed and we add metadata, this is to change the text-type values of the labels by numerical values.
> An adjustment is also made to the entire dataset to include all the labels in the index.

**9. Import the VectorIndexer library.**  
~~~
import org.apache.spark.ml.feature.VectorIndexer
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(df)
~~~
> The "features" column was indexed and the maximum number of categories to take as 4 is established.
> An adjustment is also made to the entire dataset to include all the labels in the index.

**10. Split Data.**  
~~~
val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3), seed = 1234L)
~~~
> The data is divided into training data (0.7) and test data (0.3) with randomSplit.
> The divided data is saved as trainingData and testData.

**11. Execution time.**  
~~~
val t1 = System.nanoTime
~~~
> System.nanoTime is used to measure the difference in elapsed time.
> This line is placed before the code from which you want to take the runtime begins.

**12. Create model.**  
~~~
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
~~~
> The model (dt) is created and its parameters are established.
> * setLabelCol = indexed label column indexedLabel.
> * setFeaturesCol = indexed features column indexedFeatures.

**13. Import the IndexToString library.**  
~~~
import org.apache.spark.ml.feature.IndexToString
val labelConverter = new IndexToString()
.setInputCol("prediction")
.setOutputCol("predictedLabel")
.setLabels(labelIndexer.labels) 
~~~
> The values of the indexed labels are converted into those of the original labels, and are named "predictedLabel  

**14. Import the library to use Pipeline.**
~~~
import org.apache.spark.ml.Pipeline
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
val model = pipeline.fit(train)
~~~
> The indexers and the model are joined in a Pipeline.
> The model is trained, and the Pipeline is adjusted to work with the training data.

**15. Print the model results.**
~~~
val predictions = model.transform(testData)
predictions.select("predictedLabel", "label", "features").show(5)
val evaluator = new MulticlassClassificationEvaluator()
.setLabelCol("indexedLabel")
.setPredictionCol("prediction")
.setMetricName("accuracy")
~~~
> Transform is used to change the data to be used, from train to test.
> 5 sample rows are selected to display in console.
> An evaluator is created with the MulticlassClassificationEvaluator () function, where the columns "indexedLabel" and "prediction" are selected, and the level of accuracy of the model is calculated.

~~~
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))
~~~
> Evaluate is used to evaluate the predictions made by the model.
> The error percentage is printed in the console as a result of the subtraction: 1.0 - accuracy.
~~~
val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
~~~
> The tree generated by the model is printed.

**16. Print total execution time.**
~~~
val duration = (System.nanoTime - t1) / 1e9d
~~~
> This line is placed at the end of the code from which you want to take the runtime.
> The time is obtained as a result of the subtraction: current time in nanoseconds (System.nanoTime) minus the time when starting the code (in this case the variable named t1).
> Since the result is in nanoseconds, a division between 1e9d is used to obtain the time in seconds.
> * `1e9d` = operation 10 ^ 9, and the" d "is to indicate that the result is of type double.

#### Results:
* Average error percentage = 0.08599806201550386 ≈ 9%
* Average accuracy level = 0.9140019379844961 ≈ 91%
* Average run time = 7.9979738504
---

<br>

## Logistic Regression  
**1. Start a new session of spark.**
~~~
import org.apache.spark.sql.SparkSession
val spar = SparkSession.builder().getOrCreate()
~~~

**2. Code to reduce errors during execution.**
~~~
import org.apache.log4j._
Logger.getLogger ("org"). SetLevel (Level.ERROR)
~~~

**3. Upload the file with the dataset.**
~~~
val dataset = spark.read.option ("header", "true"). option ("inferSchema", "true"). format ("csv"). load ("bank-additional-full.csv")
~~~
> The "header" and "inferSchema" options are used to obtain the title and data type of each column in the dataset.

**4. Import the libraries to work with Logistic Regression.** 
~~~
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.evaluation.MulticlassMetrics 
~~~
> The first library corresponds to and is necessary for the implementation of the Logistic Regression model.
> The second library is somewhat old, however, it is the one used to evaluate the precision of this type of algorithm.

**5. Data transformation.**
~~~
val coly = when($"y".contains("yes"), 1.0).otherwise(0.0)
val df = dataset.withColumn("y", coly)
~~~
> The values equal to "yes" of the column "y" are searched, and are replaced by 1, or otherwise by 0.
> Once the values have been changed, they are inserted again in the "y" column.

**6. Transformation for categorical data (labels to classify)**  
~~~
val data = df.select(df("y").as("label"), $"age",$"duration",$"campaign",$"pdays",$"previous",$"emp_var_rate",$"cons_price_idx",$"cons_conf_idx",$"euribor3m",$"nr_employ")
~~~
> The columns of the dataframe are selected and the column "y" is renamed as "label".

**7. Import the VectorAssembler and Vectors libraries.**
~~~
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
~~~

**8. Create new VectorAssembler object.**  
~~~
val assembler = (new VectorAssembler()
.setInputCols(Array("age","duration","campaign","pdays","previous","emp_var_rate","cons_price_idx","cons_conf_idx","euribor3m","nr_employed"))
.setOutputCol("features"))
~~~
> A new VectorAssembler object called assembler is created to save the rest of the columns as features.

**9. Split data.**
~~~
val Array(training, test) = data.randomSplit(Array(0.7, 0.3), seed = 1234L)
~~~
> The data is divided into training data (0.7) and test data (0.3) with randomSplit.
> The divided data is saved as training and test.

**10. Execution time.**
~~~
val t1 = System.nanoTime
~~~
> System.nanoTime is used to measure the difference in elapsed time.
> This line is placed before the code from which you want to take the runtime begins.

**11. Create model.**
~~~
val lr = new LogisticRegression (). setMaxIter (100)
~~~
> The model (lr) is created and its parameters are established.
> * setMaxIter = maximum number of iterations to be made by the model.

**12. Import the library to use Pipeline.**
~~~
import org.apache.spark.ml.Pipeline
val pipeline = new Pipeline().setStages(Array(assembler,lr))
val model = pipeline.fit(training)
~~~
> The assembler and the model are joined in a Pipeline.
> The model is trained, and the Pipeline is adjusted to work with the training data.

**13. Print the model results.**
~~~
val results = model.transform(test)
val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
val metrics = new MulticlassMetrics(predictionAndLabels)
~~~
> Transform is used to change the data to use, from train to test.
> Convert test results to RDD using .as and .rdd.
> * .rdd = spark data structure to see the results.

> A MulticlassMetrics () object is initialized.
> * metrics = is used to perform various tests to check the accuracy of the model.

~~~
println("Confusion matrix:")
println(metrics.confusionMatrix)
~~~
> The confusion matrix is printed for the model predictions.

~~~
metrics.accuracy
~~~
> The level of accuracy of the model is printed.

**14. Print total execution time.**
~~~
val duration = (System.nanoTime - t1) / 1e9d
~~~
> This line is placed at the end of the code from which you want to take the runtime.
> The time is obtained as a result of the subtraction: current time in nanoseconds (System.nanoTime) minus the time when starting the code (in this case the variable named t1).
> Since the result is in nanoseconds, a division between 1e9d is used to obtain the time in seconds.
> * `1e9d` = operation 10 ^ 9, and the" d "is to indicate that the result is of type double.

#### Results:
* Average error percentage = 0.0900355297157 ≈ 9%
* Average accuracy level = 0.9099644702842378 ≈ 91%
* Average run time = 9.6360462944


---

<br>

## Multilayer Perceptron  

**1. Start a new session of spark.**
~~~
import org.apache.spark.sql.SparkSession
val spar = SparkSession.builder (). getOrCreate ()
~~~

**2. Code to reduce errors during execution.**
~~~
import org.apache.log4j._
Logger.getLogger ("org"). SetLevel (Level.ERROR)
~~~

**3. Upload the file with the dataset.**
~~~
val df = spark.read.option ("header", "true"). option ("inferSchema", "true") csv ("bank-additional-full.csv")
~~~
> The "header" and "inferSchema" options are used to obtain the title and data type of each column in the dataset.

**4. Import the libraries to work with Multilayer Perceptron.**
~~~
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
~~~
> The first library does correspond and is necessary for the implementation of the Multilayer Perceptron model, however, the second library is used to evaluate the accuracy of the classification models, whether it is Multilayer Perceptron or some other algorithm.

**5. Transformation for categorical data (labels to classify)**
~~~
val data = df.select(df("y").as("label"), $"age",$"duration",$"campaign",$"pdays",$"previous",$"emp_var_rate",$"cons_price_idx",$"cons_conf_idx",$"euribor3m",$"nr_employed")
~~~
> The columns of the dataframe are selected and the column "y" will be renamed as "label".  

**6. Importar las librerias VectorAssembler y Vectors.**  
~~~
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
~~~
**7. Create new VectorAssembler object.**
~~~
val assembler = new VectorAssembler (). setInputCols (Array ("age", "duration", "campaign", "pdays", "previous", "emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed ")). setOutputCol (" features ")
val features = assembler.transform (data)
~~~
> A new VectorAssembler object called assembler is created to save the rest of the columns as features.
> The variable features is created to save the dataframe with the previous changes.

**8. Import the StringIndexer library.**
~~~
import org.apache.spark.ml.feature.StringIndexer
val labelIndexer = new StringIndexer (). setInputCol ("label"). setOutputCol ("indexedLabel"). fit (features)
~~~
> We index the label column and add metadata, this is to change the text-type values ​​of the labels by numerical values.
> An adjustment is also made to the entire dataset to include all the labels in the index.

**9. Import the VectorIndexer library.**
~~~
import org.apache.spark.ml.feature.VectorIndexer
val featureIndexer = new VectorIndexer (). setInputCol ("features"). setOutputCol ("indexedFeatures"). setMaxCategories (4) .fit (features)
~~~
> The "features" column is indexed and the maximum number of categories to be taken is set to 4.
> An adjustment is also made to the entire dataset to include all the labels in the index.

**10. Split data.**
~~~
val splits = features.randomSplit (Array (0.7, 0.3), seed = 1234L)
val train = splits (0)
val test = splits (1)
~~~
> The data is divided into training data (0.7) and test data (0.3) with randomSplit.
> The train and test variables are created to save the divided data.

**11. Execution time.**
~~~
val t1 = System.nanoTime
~~~
> System.nanoTime is used to measure the difference in elapsed time.
> This line is placed before the code from which you want to take the runtime begins.

**12. Specify layers for the neural network.**
~~~
val layers = Array[Int](10, 11, 3, 2)
~~~
> The layers for the neural network of the model are specified:
> * Size 10 input layer (features).
> * Two intermediate layers of size 11 and 3.
> * Size 2 output layer (classes).

**13. Create model.**
~~~
val trainer = new MultilayerPerceptronClassifier ()
.setLayers (layers)
.setLabelCol ("indexedLabel")
.setFeaturesCol ("indexedFeatures")
.setBlockSize (128)
.setSeed (1234L)
.setMaxIter (100)
~~~
> The coach is created and its parameters are established.
> * setLayers = variable created in the previous step layers.
> * setLabelCol = indexed label column indexedLabel.
> * setFeaturesCol = indexed features column indexedFeatures.
> * setBlockSize = default block size in kilobytes (128).
> * setSeed = randomness in the data (1234L).
> * setMaxIter = maximum number of iterations to be made by the model (100 is the default value).

**14. Import IndexToString library.**
~~~
import org.apache.spark.ml.feature.IndexToString
val labelConverter = new IndexToString ()
.setInputCol ("prediction")
.setOutputCol ("predictedLabel")
.setLabels (labelIndexer.labels)
~~~
> The values of the indexed tags are converted into those of the original tags, and are named "predictedLabel".

**15. Import the library to use Pipeline.**
~~~
import org.apache.spark.ml.Pipeline
val pipeline = new Pipeline ()
.setStages (Array (labelIndexer, featureIndexer, trainer, labelConverter))
val model = pipeline.fit (train)
~~~
> The indexers and the model (trainer) are joined in a Pipeline.
> The model is trained, and the Pipeline is adjusted to work with the training data.

**16. Print the model results.**
~~~
val predictions = model.transform (test)
val predictionAndLabels = predictions.select ("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator ()
.setLabelCol ("indexedLabel")
.setPredictionCol ("prediction")
.setMetricName ("accuracy")
~~~
> Transform is used to change the data to be used, from train to test.
> The prediction and "label" columns are selected, and are stored inside the "predictionAndLabels" variable.
> An evaluator is created with the MulticlassClassificationEvaluator () function, where the columns "indexedLabel" and "prediction" are selected, and the level of accuracy of the model is calculated.

~~~
val accuracy = evaluator.evaluate (predictions)
println ("Test Error =" + (1.0 - accuracy))
~~~
> Evaluate is used to evaluate the predictions made by the model.
> The error percentage is printed in the console as a result of the subtraction: 1.0 - accuracy.

**17. Print total execution time.**
~~~
val duration = (System.nanoTime - t1) / 1e9d
~~~
> This line is placed at the end of the code from which you want to take the runtime.
> The time is obtained as a result of the subtraction: current time in nanoseconds (System.nanoTime) minus the time when starting the code (in this case the variable named t1).
> Since the result is in nanoseconds, a division between 1e9d is used to obtain the time in seconds.
> * `1e9d` = operation 10 ^ 9, and the" d "is to indicate that the result is of type double.

#### Results:
* Average error percentage = 0.10981912144702843 ≈ 11%
* Average accuracy level = 0.8901808785529716 ≈ 89%
* Average execution time = 25.0155136536
