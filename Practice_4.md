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

11. `def grouping(columnName: String): Column`  
Indicates whether a specified column in a GROUP BY list is aggregated or not, returns 1 for aggregated or 0 for not 
aggregated in the result set.  

12. `def mean(columnName: String): Column`  
Returns the average of the values in a group.  

13. `def array_distinct(e: Column): Column`  
Removes duplicate values from the array.  

14. `def concat(exprs: Column*): Column`  
Concatenates multiple input columns together into a single column.  

15. `def explode(e: Column): Column`  
Creates a new row for each element in the given array or map column.  

16. `def reverse(e: Column): Column`  
Returns a reversed string or an array with reverse order of elements.  

17. `def size(e: Column): Column`  
Returns length of array or map.  

18. `def current_date(): Column`  
Returns the current date as a date column.  

19. `def greatest(exprs: Column*): Column`  
Returns the greatest value of the list of values, skipping null values. This function takes at least 2 parameters. 
It will return null iff all parameters are null.  

20. `def least(exprs: Column*): Column`  
Returns the least value of the list of values, skipping null values. This function takes at least 2 parameters. 
It will return null iff all parameters are null.  
