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
> `columns` is used to display only the names of the DataFrame columns.  

4. How is the scheme?  
~~~~
df.printSchema()
~~~~
> `printSchema()` is used to display only existing columns and information about them.  

5. Print the first 5 rows.  
~~~~
df.head(5)

for(row <- df.head(5)){
    println(row)
}

// Another option: limit number of rows to print
df.limit(5).show() 
~~~~
> `head()` is used to display the first N elements of the DataFrame.  
_To print the rows separately you can use a for, where the counter (row) is equal to only the 
first N elements (head), and prints them one by one._  
There is also `limit()`, which allows us to delimit up to how many rows we want to print, 
following the order of the data in the DataFrame.  
`show()` is used to display the results or print the DataFrame. Also within the parentheses 
you can specify the number of elements to display.  

6. Use describe () to learn about the DataFrame.  
~~~~
df.describe().show()
~~~~
> `describe()` is used to print a statistical summary of the DataFrame.

7. Create a new dataframe with a new column called "HV Ratio" which is the ratio between the price of the 
"High" column versus the "Volume" column of shares traded for one day.  
~~~~
val df2 = df.withColumn("HV Ratio", df("High")/df("Volume"))
df2.show()
~~~~
> The `withColumn()` function is used to rename, change the value, convert the datatype of an existing 
DataFrame column and also can be used to create a new column.  
Within the parentheses, the name of the column and the values it will take will be specified, either 
directly or as a result of an operation.  

8. What day had the highest peak in the "Close" column?  
~~~~
df.orderBy($"Close".desc).show(1)
df.select(max("Close")).show()
~~~~
> `orderBy()` is used to sort the data according to the specified column and order.  
In Scala it is necessary to place a '$' sign before the column name between " ".  
It can be sorted ascending (`asc`) or descending (`desc`).  
_In this case, the data is being sorted according to the "Close" column, in descending order, 
and only the first row will be displayed._  
Note: The second line is only to verify the result obtained, it will be explained in greater detail
in the following exercises.  

9. What is the meaning of the "Close" column?  
~~~~
//The Close column refers to closing prices, that is, the last level at which an asset was 
//traded before the market closed on a given day.
~~~~

10. What is the maximum and minimum of the "Volume" column?  
~~~~
df.select(max("Volume")).show()
df.select(min("Volume")).show()
~~~~
> `select()` allows queries in Spark Scala similar to SQL.  
`max()` is used to obtain the maximum value within a column.  
`min()` is used to obtain the minimum value within a column.  

11. With Syntaxis Scala / Spark $ answer the following:  

a. How many days was the "Close" column less than $ 600? 
~~~~
val res2 = df.filter($"Close"<600)
res2.select(count("Close")).show()
~~~~
> The `filter()` method is used to select all the elements of the DataFrame that satisfy a given condition.  
`count()` returns the total number of elements within a group (column).  
_In this case, the data in the "Close" column whose values are less than 600 are filtered and stored in the 
"res2" variable, then the variable is selected and the total number of filtered elements is counted, to later 
show the result._  

b. What percentage of the time was the "High" column greater than $ 500?  
~~~~
(df.filter($"High">500).count()*1.0 / df.count()) * 100
~~~~
> _The data in the "High" column whose values are greater than 500 are filtered and counted using the `count()` 
method. The result is multiplied by 1.0 to convert it to a Double type, divided by the total elements in the 
DataFrame (`df.count()`), and then multiplied by 100 to obtain the percentage._  

c. What is Pearson's correlation between the "High" column and the "Volume" column?  
~~~~
df.select(corr("High", "Volume")).show() 
~~~~
> `corr()` returns the Pearson Correlation Coefficient for two columns.  
A **Pearson correlation** is a number between -1 and 1 that indicates the extent to which two variables are 
linearly related.  

d. What is the maximum of the "High" column per year?  
~~~~
val df2 = df.withColumn("Year", year(df("Date")))
val dfmax = df2.groupBy("Year").max()
val res3 = dfmax.select($"Year", $"max(High)")
res3.orderBy($"Year".asc).show()
~~~~
> `groupBy()` allows you to group the data according to a given column.  
`year()` extracts the year as an integer from a given date/timestamp/string.  
_A new "Year" column is named, which will get its values from the `year()` function applied to the data 
in the "Date" column.  
The data is grouped according to the "Year" column and the maximum is calculated.  
The "Year" column and the new "max(High)" column are selected, where the maximum values calculated for 
the "High" column data will be displayed.  
Finally, the data obtained are sorted according to the "Year" column in ascending order, and displayed._  

e. What is the average of the "Close" column for each calendar month?  
~~~~
val df3 = df.withColumn("Month", month(df("Date")))
val dfavg = df3.groupBy("Month").mean()
val res4 = dfavg.select($"Month", $"avg(Close)")
res4.orderBy($"Month".asc).show()
~~~~
> `month()` extracts the month as an integer from a given date/timestamp/string.  
`mean()` returns the average of the values in a group. `avg()` is also used.  
_A new "Month" column is named, which will get its values from the `month()` function applied to the data 
in the "Date" column.  
The data is grouped according to the "Month" column and the average is calculated.  
The "Month" column and the new "avg(Close)" column are selected, where the average values calculated for 
the data in the "Close" column will be displayed.  
Finally, the data obtained are sorted according to the "Month" column in ascending order, and displayed._
