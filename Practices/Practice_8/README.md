# One-vs-Rest classifier (a.k.a. One-vs-All)  
OneVsRest is an example of a machine learning reduction for performing multiclass classification given a base classifier that can perform binary classification efficiently. It is also known as “One-vs-All.”  
OneVsRest is implemented as an Estimator. For the base classifier, it takes instances of Classifier and creates a binary classification problem for each of the k classes. The classifier for class i is trained to predict whether the label is i or not, distinguishing class i from all other classes.  
Predictions are done by evaluating each binary classifier and the index of the most confident classifier is output as label.  


## Steps:  
### 1. Import libraries.
~~~
import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
~~~

### 2. Import a Spark Session.  
~~~
import org.apache.spark.sql.SparkSession
~~~

### 3. Create a Spark session.  
~~~
  def main(): Unit = {
    val spark = SparkSession.builder.appName("MulticlassClassificationEvaluator").getOrCreate()
~~~  
> Tambien se asigna el nombre a la aplicacion: "MulticlassClassificationEvaluator".  

### 4. Load data file.
~~~
val inputData = spark.read.format("libsvm").load("sample_multiclass_classification_data.txt")
~~~

### 5. Generate the train/test split.
~~~
val Array(train, test) = inputData.randomSplit(Array(0.8, 0.2))
~~~
> Split the data using random split into 80% traning and 20% testing datasets.  
> `inputData.randomSplit`: randomly splits a RDD with the provided weights.  

### 6. Instantiate the base classifier.
~~~
val classifier = new LogisticRegression()
.setMaxIter(10)
.setTol(1E-6)
.setFitIntercept(true)
~~~
> Configure the regression object without having to have a base logistic model at hand so it can be fed into the classifier.  
> Se indica el maximo de iteraciones en 10.  
> .setTol es un parámetro para la tolerancia de convergencia para algoritmos iterativos.  
> .setFitIntercept es un valor de tipo Booleano; el cual contesta a la pregunta ¿debería el modelo ajustarse a un término de intercepción?.  

### 7. Instantiate the One Vs Rest Classifier.
~~~
val ovr = new OneVsRest().setClassifier(classifier)
~~~  
> In this step, we feed the configured regression model into the classifier.  

### 8. Train the multiclass model.
~~~
val ovrModel = ovr.fit(train)
~~~  
> We generate a trained model by invoking the fit method on our one vs rest object.  

### 9. Score the model on test data.
~~~
val predictions = ovrModel.transform(test)
~~~  
> Now, we will use the trained model to generate predictions for the test data.  

### 10. Obtain evaluator.
~~~
val evaluator = new MulticlassClassificationEvaluator()
.setMetricName("accuracy")
~~~  
> We pass predictions to the MultiClassClassificationEvaluator to generate an accurancy value.  

### 11. Compute the classification error on test data.
~~~
val accuracy = evaluator.evaluate(predictions)
~~~  
> Se evaluan las predicciones hechas por el modelo con la funcion `evaluate` para obtener el nivel de exactitud.  

### 12. Print result.
~~~
println(s"Test Error = ${1 - accuracy}")
~~~
> Se calcula el porcentaje de error con base al resultado obtenido para el nivel de exactitud.  
> En este caso obtuvimos un 6% (aprox.), lo cual indica que el nivel de exactitud del modelo es bastante alto y por lo tanto muy bueno.  
