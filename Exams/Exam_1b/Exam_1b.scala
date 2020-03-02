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
