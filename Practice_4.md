# Functions in scala  

1.  `def avg(columnName: String): Column`  
Returns the average of the values in a group. 

2.  `def collect_list(columnName: String): Column`  
Returns a list of objects with duplicates. 

3.  `def collect_set(columnName: String): Column `  
Returns a set of objects with duplicate elements eliminated. 

4. `def first(columnName: String): Column `   
The function by default returns the first values it sees. It will return the first non-null value it sees when ignoreNulls is set to true. If all values are null, then null is returned.   

5.  `def last(columnName: String): Column `  
The function by default returns the last values it sees. It will return the last non-null value it sees when ignoreNulls is set to true. If all values are null, then null is returned.   

6.  `def max(columnName: String): Column`  
Returns the maximum value of the column in a group.   

7.  `def min(columnName: String): Column`  
Returns the minimum value of the column in a group.   

8.  `def skewness(columnName: String): Column`  
Returns the skewness of the values in a group.   

9.  `def sum(columnName: String): Column`  
Returns the sum of all values in the given column.   

10. `def kurtosis(columnName: String): Column`  
Returns the kurtosis of the values in a group.  

11. `def countDistinct(columnName: String, columnNames: String*): Column`  
Aggregate function: returns the number of distinct items in a group.  

12. `def mean(columnName: String): Column`  
Returns the average of the values in a group.  

13. `def var_pop(columnName: String): Column`  
Returns the population variance of the values in a group.  

14. `def concat(exprs: Column*): Column`  
Concatenates multiple input columns together into a single column.  

15. `def reverse(e: Column): Column`  
Returns a reversed string or an array with reverse order of elements.  

16. `def current_date(): Column`  
Returns the current date as a date column.  

17. `def dayofmonth(e: Column): Column`  
Extracts the day of the month as an integer from a given date/timestamp/string.  

18. `def dayofweek(e: Column): Column`  
Extracts the day of the week as an integer from a given date/timestamp/string. Ranges from 1 for a Sunday through to 7 for a Saturday.  

19. `def month(e: Column): Column`  
Extracts the month as an integer from a given date/timestamp/string.  

20. `def year(e: Column): Column`  
Extracts the year as an integer from a given date/timestamp/string.  
