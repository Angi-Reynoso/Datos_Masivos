## Instructions
Develop the following instructions in Spark with the Scala programming language.

## Objective
The objective of this practical exam is to try to group customers from the specific regions of a wholesale distributor. This is based on the sales of some product categories.
The source of the data is located in the repository: https://github.com/jcromerohdz/BigData/blob/master/Spark_clustering/Wholesale%20customers%20data.csv

## Steps
**1. Import Session**
Import a simple Spark session.
~~~
import org.apache.spark.sql.SparkSession
~~~  

**2. Minimize errors**
Use the lines of code to minimize errors.
~~~
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
~~~  

**3. Create Session**
Create an instance of the Spark session.
~~~
val spark = SparkSession.builder().getOrCreate()
~~~  

**4. Import Kmeans**   
Import the K Means library for the grouping algorithm.
~~~
import org.apache.spark.ml.clustering.KMeans
~~~  

**5. Upload the Dataset**  
Load the Wholesale Customers Data dataset. 
~~~
val dataset = spark.read.option("header","true").option("inferSchema","true").csv("Wholesale customers data.csv")
~~~  
> We use `option (" header "," true ")` to preserve the column names of the original dataset.
> We use `.option (" inferSchema "," true ")` to get the data type of each column based on the original dataset.

**6. Select columns**
Select the following columns: Fresh, Milk, Grocery, Frozen, Detergents_Paper and Delicassen, and call this set feature_data.  
~~~
val feature_data = dataset.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen")
~~~  
> We select the columns mentioned above with the name of the dataset ('dataset') and the function `select`.
> To indicate that it is a column it is necessary to put the '$' sign first, followed by the column name in double quotes.

**7. Import Vector Assembler and Vectors.**
~~~
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
~~~  

**8. Create Vector Assembler object**  
Create a new Vector Assembler object for the feature columns (features column) as an input set, remembering that there are no labels (label column).
~~~
val assembler = new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")).setOutputCol("features")
~~~  
> We use the `VectorAssembler ()` function to create the object. Next we use `setInputCols` to specify the input columns, and` setOutputCol` to indicate the name of the output column.
> In this case the names of the columns are declared in the form of an array, so it is not necessary to use the '$' sign as in one of the previous steps.

**9. Data transformation**  
Use the assembler object to transform feature_data.
~~~
val training_data = assembler.transform(feature_data).select("features")
~~~  
> The `transform` function was used to transform the selected data (features), and the result is saved inside the training_data variable.

**10. Create Model**  
Create a Kmeans model with K = 3.
The model fits the data (training_data).
~~~
val kmeans = new KMeans().setK(3).setSeed(1L)
val model = kmeans.fit(training_data)
~~~  
> The `KMeans ()` function is used to create the model. Next, `setK ()` is used to set the number of 'K' or groups to create by the model, and `setSeed ()` is used to randomize the data to be selected.
> Once the model is created, it fits the data (training_data) using the `fit` function. 

**11. Evaluate Groups**  
Evaluate the groups using Within Set Sum of Squared Errors (WSSSE) and print the centroids.
<br>
_Within Set Sum of Squared Errors_ equals:
* The sum of the squared distances of each observation to the centroid of your group.
* Measure to indicate how well centroids represent members of your group.

In each iteration that the model performs, an attempt is made to reduce the value obtained from this sum.
<br>
A _Centroid_:
* It's the average of the points that make up a group.
* This is the name of the representative of each group.
~~~
val WSSSE = model.computeCost(training_data)
println(s"Within Set Sum of Squared Errors = $WSSSE")

println("Cluster Centers: ")
model.clusterCenters.foreach(println)
~~~  
> The `computeCost` function is used to calculate the WSSSE, indicating the data to use (training_data), and the result is printed.
> The `clusterCenters` function is used to obtain the centroids of each group created by the model, and` foreach` is used to print the results of each group as separate arrays; as can be seen in the results below.

Results:
~~~
Within Set Sum of Squared Errors = 8.095172370767671E10
Cluster Centers: 
[7993.574780058651,4196.803519061584,5837.4926686217,2546.624633431085,2016.2873900293255,1151.4193548387098]
[9928.18918918919,21513.081081081084,30993.486486486487,2960.4324324324325,13996.594594594595,3772.3243243243246]
[35273.854838709674,5213.919354838709,5826.096774193548,6027.6612903225805,1006.9193548387096,2237.6290322580644]
~~~  

### Conclusion
As a result of the model, a total of 6 iterations were obtained; This can be verified by counting the number of centroids obtained for each group or "K", which are represented in the form of arrays (enclosed in '[]').
