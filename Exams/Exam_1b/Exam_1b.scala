//1. Start spark session
import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder().getOrCreate()

//2. Upload file and make spark infer data types
val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016.csv")

//3. Column names
df.columns

//4. See scheme
df.printSchema()

//5. First 5 rows
df.head(5)

for(row <- df.head(5)){
    println(row)
}

// Another option: limit number of rows to print
df.limit(5).show() 

//6. Learn about the DataFrame
df.describe().show()

//7. Create new column "HV Ratio"
val df2 = df.withColumn("HV Ratio", df("High")/df("Volume"))
df2.show()

//8. What day had the highest peak in the Close column?
df.orderBy($"Close".desc).show(1)
df.select(max("Close")).show()

//9. Meaning of the Close column
df.describe("Close").show()
//The Close column refers to closing prices, that is, the last level at which an asset was 
//traded before the market closed on a given day.
//La columna Close se refiere a los precios de cierre, es decir, el último nivel en el que 
//se negoció un activo antes de que el mercado cerrara en un día determinado.

//10. Maximum and minimum of the Volume column
df.select(max("Volume")).show()
df.select(min("Volume")).show()

//11.
//11a. How many days was the Close column less than 600?
val res2 = df.filter($"Close"<600)
res2.select(count("Close")).show()

//11b. What percentage of the time was the High column greater than 500?
(df.filter($"High">500).count()*1.0 / df.count()) * 100

//df.filter($"High">500).count() = registros mayores a 500
//df.count() = total de registros

//11c. What is Pearson's correlation between the High column and the Volume column?
df.select(corr("High", "Volume")).show() 

//11d. What is the maximum of the High column per year?
val df2 = df.withColumn("Year", year(df("Date")))
val dfmax = df2.groupBy("Year").max()
val res3 = dfmax.select($"Year", $"max(High)")
res3.orderBy($"Year".asc).show()

//11e. What is the average of the Close column for each calendar month?
val df3 = df.withColumn("Month", month(df("Date")))
val dfavg = df3.groupBy("Month").mean()
val res4 = dfavg.select($"Month", $"avg(Close)")
res4.orderBy($"Month".asc).show()
