# Logistic Regression Project  

## Description: 
In this project we will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.  
This data set contains the following features:  
* _'Daily Time Spent on Site'_: consumer time on site in minutes.  
* _'Age'_: cutomer age in years.  
* _'Area Income'_: Avg. Income of geographical area of consumer.  
* _'Daily Internet Usage'_: Avg. minutes a day consumer is on the internet.  
* _'Ad Topic Line'_: Headline of the advertisement.  
* _'City'_: City of consumer.  
* _'Male'_: Whether or not consumer was male.  
* _'Country'_: Country of consumer.  
* _'Timestamp'_: Time at which consumer clicked on Ad or closed window.  
* _'Clicked on Ad'_: 0 or 1 indicated clicking on Ad.  

## Steps:  
### 1. Take the data
**1.1 Import a SparkSession with the Logistic Regression library**  
~~~
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
~~~  
> `LogisticRegression`:  

**1.2 Optional: Use the Error reporting code**  
~~~
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
~~~  
> `log4j._`:  
> `Logger.getLogger("org").setLevel(Level.ERROR)`:  

**1.3 Create a Spark session**  
~~~
val spark = SparkSession.builder().getOrCreate()
~~~  

**1.4 Use Spark to read the csv Advertising file**  
~~~
val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("advertising.csv")
~~~  

**1.5 Print the Schema of the DataFrame**  
~~~
data.printSchema()
~~~  

### 2. Display the data  
**2.1 Print an example line**  
~~~
data.head(1)

val colnames = data.columns
val firstrow = data.head(1)(0)
println("\n")
println("Example data row")
for(ind <- Range(0, colnames.length)){
    println(colnames(ind))
    println(firstrow(ind))
    println("\n")
}
~~~  
> `head()`:  

### 3. Prepare the DataFrame for Machine Learning  
**3.1 Create a new column called "Hour" of the Timestamp containing "Hour of the click"**  
~~~
val timedata = data.withColumn("Hour",hour(data("Timestamp")))
~~~  
> `withColumn()`:  

**3.2 Rename the column "Clicked on Ad" to "label"**  
**3.3 Take the following columns as features "Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage", "Hour", "Male"**  
~~~
val logregdata = timedata.select(data("Clicked on Ad").as("label"), $"Daily Time Spent on Site", $"Age", $"Area Income", $"Daily Internet Usage", $"Hour", $"Male")
~~~  
> `.select()`:  

**3.4 Import VectorAssembler and Vectors**  
~~~
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
~~~  
> `VectorAssembler`:  
> `Vectors`:  

**3.5 Create a new VectorAssembler object called assembler for the features**  
~~~
val assembler = (new VectorAssembler()
                  .setInputCols(Array("Daily Time Spent on Site", "Age","Area Income","Daily Internet Usage","Hour","Male"))
                  .setOutputCol("features"))
~~~  
> `.setInputCols`:  
> `.setOutputCol`:  

**3.6 Use randomSplit to create train and test data divided by 70/30**  
~~~
val Array(training, test) = logregdata.randomSplit(Array(0.7, 0.3), seed = 12345)
~~~  
> `.randomSplit()`:  
>   * `seed`:   

### 4. Set up a Pipeline  
**4.1 Import Pipeline**  
~~~
import org.apache.spark.ml.Pipeline
~~~  
> `Pipeline`:  

**4.2 Create a new LogisticRegression object called lr**  
~~~
val lr = new LogisticRegression()
~~~  

**4.3 Create a new pipeline with the elements: assembler, lr**  
~~~
val pipeline = new Pipeline().setStages(Array(assembler, lr))
~~~  
> `.setStages()`:  

**4.4 Adjust (fit) the pipeline for the training set**  
~~~
val model = pipeline.fit(training)
~~~  
> `.fit()`:  

**4.5 Take the Results in the Test set with transform**  
~~~
val results = model.transform(test)
~~~  
> `.transform()`:  

### 5. Model evaluation  
**5.1 For Metrics and Evaluation import MulticlassMetrics**  
~~~
import org.apache.spark.mllib.evaluation.MulticlassMetrics
~~~  
> `MulticlassMetrics`:  

**5.2 Convert test results to RDD using .as and .rdd**  
~~~
val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
~~~  
> `.as()`:  
> `.rdd`:  

**5.3 Initialize a MulticlassMetrics object**  
~~~
val metrics = new MulticlassMetrics(predictionAndLabels)
~~~  

**5.4 Print the Confusion Matrix**  
~~~
println("Confusion matrix:")
println(metrics.confusionMatrix)
~~~  
> `metrics`:  
> `confusionMatrix`:  

**5.5 Print the model Accuracy value**  
~~~
metrics.accuracy
~~~  
> `accuracy`:  

