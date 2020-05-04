// Import LinearRegression
import org.apache.spark.ml.regression.LinearRegression

// Optional: use the following code to configure errors
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

//Start a simple Spark Session
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()

//Use Spark for the Clean-Ecommerce csv file.
val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("Clean-Ecommerce.csv")

// Print the schema on the DataFrame.
data.printSchema

//Print an example row from the DataFrame.
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

//Transform the data frame so that it takes the form of
// ("label", "features")
// Import VectorAssembler and Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

// Rename the Yearly Amount Spent column as "label"
// Also from the data take only the numerical column
// set all this as a new DataFrame called df
val df = data.select(data("Yearly Amount Spent").as("label"),$"Avg Session Length",$"Time on App",$"Time on Website",$"Length of Membership")


// Use the VectorAssembler object to convert the input columns of the df
// to a single output column of an array named "features"
// Set the input columns from where we are supposed to read the values.
// Call this a new assambler.
val assembler = new VectorAssembler().setInputCols(Array("Avg Session Length","Time on App","Time on Website","Length of Membership")).setOutputCol("features")

// Use the assembler to transform our DataFrame to two columns: label and features
val output = assembler.transform(df).select($"label",$"features")

// Create an object for line regression model.
val p1 = new LinearRegression()

// Fit the model for the data and call this model lrModel
val p1Model = p1.fit(output) //ajustar el output


// Print the coefficients and intercept for the linear regression
println(s"Coefficients: ${p1Model.coefficients} Intercept: ${p1Model.intercept}")

// Summarize the model on the training set print the output of some metrics!
// Use our model's .summary method to create an object
// called trainingSummary
val trainingSummary = p1Model.summary

// Show the residuals values, the RMSE, the MSE, and also the R ^ 2.

trainingSummary.residuals.show()
trainingSummary.predictions.show()
trainingSummary.r2 
trainingSummary.rootMeanSquaredError

