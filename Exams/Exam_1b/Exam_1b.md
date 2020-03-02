# Exam: Unit 1 - Part 2 (DataFrames)  

### Instructions:
Answer the following questions with Spark DataFrames using the "CSV" Netflix_2011_2016.csv.
> A **DataSet** is a collection of distributed data that already has a structure, unlike RDDs,
which are unstructured data sets.
A **DataFrame** is a DataSet that is organized in columns at the same time, that is, we will have the data
structured and each column with its corresponding name, which will be much easier for us
query, modify or transform that data set.  

### Questions and answers:
1. Start a simple Spark session.  
~~~~
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()
~~~~
> The first line serves to import SparkSQL.  
In the following line we use `SparkSession`, which is the main object or the base from which all the 
functionality of Apache Spark hangs, and as the name implies, it will also help us to create a new 
session in Spark.  

2. Load the Netflix Stock CSV file, have Spark infer the data types.
~~~~
val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016.csv")
~~~~
> First we declare a variable (df) to which the file to be loaded will be assigned.  
`spark.read` is used to load a CSV file into Spark.  
`.option ("header", "true")` is used to load file headers.  
`.option ("inferSchema", "true")` helps Spark automatically infer data types from the file.  
And finally in `csv("")` we put the name of the file and its extension, in this case _"Netflix_2011_2016.csv"_.  

3. What are the names of the columns?  
~~~~
df.columns
~~~~
> `.columns` is used to display only the names of the DataFrame columns.  

4. How is the scheme?  
~~~~
df.printSchema()
~~~~
> `.printSchema()` is used to display only existing columns and information about them.  

5. Print the first 5 rows.  
~~~~
df.head(5)

for(row <- df.head(5)){
    println(row)
}

// Another option: limit number of rows to print
df.limit(5).show() 
~~~~
> `.head()` is used to display the first N elements of the DataFrame.  
_To print the rows separately you can use a for, where the counter (row) is equal to only the 
first N elements (.head), and prints them one by one._  
There is also `.limit()`, which allows us to delimit up to how many rows we want to print, 
following the order of the data in the DataFrame.  
`.show()` is used to display the results or print the DataFrame. Also within the parentheses 
you can specify the number of elements to display.  

6. Use describe () to learn about the DataFrame.  
~~~~
df.describe().show()
~~~~
> `.describe()` is used to print a statistical summary of the DataFrame.

7. Create a new dataframe with a new column called "HV Ratio" which is the ratio between the price of the 
"High" column versus the "Volume" column of shares traded for one day.  
~~~~
val df2 = df.withColumn("HV Ratio", df("High")/df("Volume"))
df2.show()
~~~~
> `.withColumn()` is used to add a column in a DataFrame. Within the parentheses, the name of the column 
and the values it will take will be specified, either directly or as a result of an operation.  

8. What day had the highest peak in the "Close" column?  
~~~~
df.orderBy($"Close".desc).show(1)
df.select(max("Close")).show()
~~~~
> `.orderBy()` is used to sort the data according to the specified column and order.  
In Scala it is necessary to place a '$' sign before the column name between " ".  
It can be sorted ascending (`.asc`) or descending (`.desc`).  
_In this case, the data is being sorted according to the "Close" column, in descending order, 
and only the first row will be displayed._  

9. What is the meaning of the "Close" column?  
~~~~
~~~~
> 

10. What is the maximum and minimum of the "Volume" column?  
~~~~
~~~~
> 

11. With Syntaxis Scala / Spark $ answer the following:  
  a. How many days was the "Close" column less than $ 600? 
~~~~
~~~~
> 
  b. What percentage of the time was the "High" column greater than $ 500?  
~~~~
~~~~
> 
  c. What is Pearson's correlation between the "High" column and the "Volume" column?  
~~~~
~~~~
> 
  d. What is the maximum of the "High" column per year?  
~~~~
~~~~
> 
  e. What is the average of the "Close" column for each calendar month?  
~~~~
~~~~
> 
