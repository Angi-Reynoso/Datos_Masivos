
## LINEAR REGRESSION EXERCISE

## Instructions: 

1. Import LinearRegression
~~~
import org.apache.spark.ml.regression.LinearRegression
~~~

2. Optional: Use the following code to configure errors 
~~~
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
~~~
>`log4j._` It is an additional library that allows our application to display information messages about what is happening in it. 

>`getLogger`It is a method of a Logger class used find or create a logger. 

3. Start a simple Spark Session
> The entry point into all functionality in Spark is the `SparkSession` class. To create a basic SparkSession, just use `SparkSession.builder()`
~~~
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()
~~~

4. Use Spark for the Clean-Ecommerce csv file.
~~~
val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("Clean-Ecommerce.csv")
~~~
> First declare a variable (df) to which the file to be loaded will be assigned.  
`spark.read` is used to load a CSV file into Spark.  
`.option ("header", "true")` is used to load file headers.  
`.option ("inferSchema", "true")` helps Spark automatically infer data types from the file.  
 And finally in `csv("")` we put the name of the file and its extension, in this case _"Clean-Ecommerce.csv"_.  


5. Print the schema on the DataFrame.
~~~
data.printSchema
~~~
> `printSchema()` is used to display only existing columns and information about them

6. Print an example row from the DataFrane.
~~~
data.head(1)
val colnames = data.columns
val firstrow = data.head(1)(0)
println("\n")
println("Example data row")
for(ind <- Range(1, colnames.length)){
    println(colnames(ind))
    println(firstrow(ind))
    println("\n")
}
~~~
> `head()` is used to display the first N elements of the DataFrame.  

## Set up the DataFrame for Machine Learning

7. Transform the data frame so that it takes the form of ("label", "features")
Import VectorAssembler and Vectors
~~~
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
~~~

8. Rename the Yearly Amount Spent column as "label"  
8.1 Also from the data take only the numerical column  
8.2 Leave all this as a new DataFrame called df  

9. Use the VectorAssembler object to convert the input columns of the df  
9.1 to a single output column of an array named "features"  
9.2 Configure the input columns from where we are supposed to read the values.  
9.3 Call this a new assambler.  

10. Use the assembler to transform our DataFrame to two columns: label and features

11. Create an object for line regression model.

12. Fit the model for the data and call this model lrModel


14. Print the coefficients and intercept for the linear regression

15. Summarize the model on the training set print the output of some metrics!
Use our model's .summary method to create an object called trainingSummary

16. Show the residuals values, the RMSE, the MSE, and also the R ^ 2.


