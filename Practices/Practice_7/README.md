# Linear Support Vector Machine
SVM or Support Vector Machine is a linear model for classification and regression problems. It can solve linear and non-linear problems and work well for many practical problems. The idea of SVM is simple: The algorithm creates a line or a hyperplane which separates the data into classes.  

## Steps:  
### 1. Import library.
~~~
import org.apache.spark.ml.classification.LinearSVC
~~~  
> The library is loaded to use SVM: `LinearSVC`.
### 2. Import a Spark Session.
~~~
import org.apache.spark.sql.SparkSession

def main(): Unit = {
    val spark = SparkSession.builder.appName("LinearSVCExample").getOrCreate()
~~~  
> A new spark session is created, and the application name "LinearSVCExample" is assigned.  

### 3. Load the data stored in LIBSVM format as a DataFrame.
~~~
val training = spark.read.format("libsvm").load("sample_libsvm_data.txt")
~~~

### 4. Create an object of type LinearSVC. 
~~~
val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)
~~~  
> Set the number of iterations to 10 with the setMaxIter method and Set the regularization parameter to 0.1.  

### 5. Fit the model.
~~~
val lsvcModel = lsvc.fit(training)
~~~  
> The model conforms to the training data.  

### 6. Print result.
~~~
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")
~~~  
> The coefficients and interception obtained with the model are printed.  
> Los coeficientes corresponden a los datos clasificados dentro del modelo, donde los valores positivos (mayores a 0) corresponden a una categoria, los valores negativos (menores a 0) corresponden a otra, y los valores igual a 0 son aquellos que no han podido ser clasificados de manera correcta.  
> La intercepcion es un coeficiente que se utiliza dentro de la formula del SVM para hacer los calculos necesarios para clasificar a los datos provenientes del dataset.  
