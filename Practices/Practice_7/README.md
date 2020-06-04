# Linear Support Vector Machine
SVM or Support Vector Machine is a linear model for classification and regression problems. It can solve linear and non-linear problems and work well for many practical problems. The idea of SVM is simple: The algorithm creates a line or a hyperplane which separates the data into classes.  

## Steps:  
### 1. Import library.
~~~
import org.apache.spark.ml.classification.LinearSVC
~~~  
> Se carga la libreria para usar SVM: `LinearSVC`.  

### 2. Import a Spark Session.
~~~
import org.apache.spark.sql.SparkSession

def main(): Unit = {
    val spark = SparkSession.builder.appName("LinearSVCExample").getOrCreate()
~~~  
> Se crea una nueva sesión de spark, y se asigna el nombre de la aplicación "LinearSVCExample".  

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
> Se ajusta el modelo a los datos de entrenamiento.  

### 6. Print result.
~~~
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")
~~~  
> Se imprimen los coeficientes y la intercepción obtenidas con el modelo.  
