// Import LinearRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.LinearRegression

// Optional: use the following code to configure errors
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

//Start a simple Spark Session

val spark = SparkSession.builder().getOrCreate()

//Use Spark for the Clean-Ecommerce csv file.
val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("Clean-Ecommerce.csv")

// Print the schema on the DataFrame.
data.printSchema

//Print an example row from the DataFrane.
data.head(1)

//Transform the data frame so that it takes the form of
// ("label", "features")

// Import VectorAssembler and Vectors


// Rename the Yearly Amount Spent column as "label"
// Also from the data take only the numerical column
// Leave all this as a new DataFrame called df

// Have the assembler object convert the input values to a vector


// Use the VectorAssembler object to convert the input columns of the df
// to a single output column of an array named "features"
// Set the input columns from where we are supposed to read the values.
// Call this a new assambler.

// Use the assembler to transform our DataFrame to two columns: label and features


// Create an object for line regression model.
val p1 = new LinearRegression()

// Fit the model for the data and call this model lrModel
val p1Model = p1.fit(output) //ajustar el output

// Print the coefficients and intercept for the linear regression

val trainingSummary = p1Model.summary

// Summarize the model on the training set print the output of some metrics!
// Use our model's .summary method to create an object
// called trainingSummary

// Show the residuals values, the RMSE, the MSE, and also the R ^ 2.


