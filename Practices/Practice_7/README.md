# Linear Support Vector Machine

## Steps:  
### 1. Import libraries and package
~~~
package org.apache.spark.examples.ml
import org.apache.spark.ml.classification.LinearSVC
~~~
### 2. Import a Spark Session.
~~~
import org.apache.spark.sql.SparkSession
~~~

### 3.Load the data from the file and add it to a variable to train it.
~~~
val spark = SparkSession.builder.appName("LinearSVCExample").getOrCreate()
~~~
### 4. Load the data stored in LIBSVM format as a DataFrame.
~~~
val training = spark.read.format("libsvm").load("/usr/local/spark-2.3.4-bin-hadoop2.6/data/mllib/sample_libsvm_data.txt")
~~~
### 5. Create an object of type LinearSVC. 
> Set the number of iterations to 10 with the setMaxIter method and Set the regularization parameter
~~~
val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)
~~~
### 6. Fit the model
~~~
val lsvcModel = lsvc.fit(training)
~~~
### 7. Print result
~~~
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")
~~~
