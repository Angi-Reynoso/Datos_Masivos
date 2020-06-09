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
> The coefficients correspond to the data classified within the model, where the positive values (greater than 0) correspond to one category, the negative values (less than 0) correspond to another, and the values equal to 0 are those that could not be correctly classified.  
> Interception is a coefficient that is used within the SVM formula to make the calculations necessary to classify the data coming from the dataset.  
