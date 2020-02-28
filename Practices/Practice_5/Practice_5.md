# Aggregate Functions  

1. `def approx_count_distinct(columnName: String, rsd: Double): Column`  
Returns the approximate number of distinct items in a group.  
_rsd_ = maximum estimation error allowed (default = 0.05)  

2. `def avg(columnName: String): Column`  
Returns the average of the values in a group.  

3. `def collect_list(columnName: String): Column`  
Returns a list of objects with duplicates.  

4. `def covar_pop(columnName1: String, columnName2: String): Column`   
Returns the population covariance for two columns.   
> **Covariance** is a value that indicates the degree of joint variation of 
two random variables with respect to their means.  

5. `def first(columnName: String, ignoreNulls: Boolean): Column`   
Returns the first value of a column in a group.  
 -> The function by default returns the first values it sees.   
 -> It will return the first non-null value it sees when ignoreNulls is set to true.   
 -> If all values are null, then null is returned.   

6. `def kurtosis(columnName: String): Column`  
Returns the kurtosis of the values in a group.   
> **Kurtosis** refers to the sharpness of the peak of a frequency-distribution curve.  

7. `def last(columnName: String, ignoreNulls: Boolean): Column`  
Returns the last value of the column in a group.  
 -> The function by default returns the last values it sees.   
 -> It will return the last non-null value it sees when ignoreNulls is set to true.  
 -> If all values are null, then null is returned.   

8. `def skewness(columnName: String): Column`   
Returns the skewness of the values in a group.   
> **Skewness** refers to distortion or asymmetry in a symmetrical bell curve, or normal 
distribution, in a set of data. If the curve is shifted to the left or to the right, 
it is said to be skewed.  
Skewness can be quantified as a representation of the extent to which a given distribution
varies from a normal distribution. A normal distribution has a skew of zero, while a 
lognormal distribution, for example, would exhibit some degree of right-skew.  

9. `def corr(columnName1: String, columnName2: String): Column`  
Returns the Pearson Correlation Coefficient for two columns. 
> **Pearson's correlation coefficient** is the test statistics that measures the statistical 
relationship, or association, between two continuous variables. It is known as the best 
method of measuring the association between variables of interest because it is based on 
the method of covariance.
A Pearson correlation is a number between -1 and 1 that indicates the extent to which two 
variables are linearly related.*/

10. `def count(columnName: String): TypedColumn[Any, Long]`  
Returns the number of items in a group. 

11. `def covar_samp(columnName1: String, columnName2: String): Column`  
Returns the sample covariance for two columns. 

12. `def stddev_pop(columnName: String): Column`  
Returns the population standard deviation of the expression in a group.
> **The standard deviation** is the most common measure of dispersion, which indicates 
how scattered the data is with respect to the average. The greater the standard 
deviation, the greater the dispersion of the data.*/

13. `def stddev_samp(columnName: String): Column`  
Returns the sample standard deviation of the expression in a group. 
stddev is an alias for stddev_samp.

14. `def var_pop(columnName: String): Column`  
Returns the population variance of the values in a group.

15. `def var_samp(columnName: String): Column`  
Returns the unbiased variance of the values in a group. 
ariance is an alias for var_samp.

> **Variance** is a measure of dispersion that represents the variability of a 
series of data with respect to its mean.
The variance of a sample or a set of values is the sum of the squared 
deviations from the average, all this divided by the total number of 
observations minus 1.  
