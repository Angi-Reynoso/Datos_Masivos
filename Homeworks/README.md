# Index #  
* [Homework 1: Pearson’s Correlation Coefficient](#homework-1-pearsons-correlation-coefficient)  
* [Homework 2: Variance Function](#homework-2-variance-function)  

<br>

---

<br>

# Homework 1: Pearson’s Correlation Coefficient #

Pearson's correlation coefficient is a test that measures the statistical relationship between two continuous variables. If the association between the elements is not linear, then the coefficient is not specifically identified.
The correlation coefficient can take a range of values from +1 to -1. A value of 0 indicates that there is no association between the two variables. A value greater than 0 indicates a positive association. That is, a measure that increases the value of one variable, so does the value of the other. A value less than 0 indicates a negative association; that is, a measure that increases the value of one variable, the value of the other reduces.

The Pearson correlation coefficient aims to indicate how associated two variables are associated with each other so:

* **Correlation less than zero:** If the correlation is less than zero, it means that it is negative, that is, that the variables are inversely related.  

* **When the value of one variable is high**, the value of the other variable is low. The closer to -1, the clearer the extreme covariation. If the coefficient is equal to -1, we refer to a perfect negative correlation.

* **Correlation greater than zero:** If the correlation is equal to +1 it means that it is perfect positive. In this case it means that the correlation is positive, that is, that the variables are directly correlated.
When the value of one variable is high, the value of the other is also high, the same happens when they are low. If it is close to +1, the coefficient will be covariation.

* **Correlation equal to zero:** when the correlation is equal to zero it means that it is not possible to determine some sense of covariation. However, it does not mean that there is no non-linear relationship between the variables.
When the variables are independent, it means that they are correlated, but this means that the result is true.

> APA

<br>

---

<br>

# Homework 2: Variance Function #  

There are two ways to use this function in Scala:
~~~~
def var_samp (columnName: String): Column  
def variance (columnName: String): Column  
~~~~  
Both are used to calculate and return the unbiased variance of the values in a group.

### What is the variance? ###  
Variance is a measure of dispersion that represents the variability of a series of data with respect to its mean.  

### How is the Variance calculated? ###  
* The variance of a sample or a set of values is equal to the sum of the squared deviations from the average, all this divided by the total number of observations minus 1.  
* The variance of a population is formally calculated as the sum of squared residues divided by the total observations. Incidentally, we understand as residual the difference between the value of a variable at a time and the average value of the entire variable.  
* The variance is always greater than or equal to zero. When the residuals are squared it is mathematically impossible for the variance to come out negative. And that way it can't be less than zero.  

### Variance as a measure of dispersion ###  
The variance, together with the standard deviation, are measures of data dispersion or observations. The dispersion of these data indicates the variety that they present, that is, if all the values in a set of data are equal, then there is no dispersion, but instead, if not all are equal then there is dispersion.  
This dispersion can be large or small, depending on how close the values are to the average.  

### Applications ###
Variance as a measure of dispersion has multiple applications in various areas, some of its utilities are:  
* It represents a decision-making aid.  
  * For example on an investment (also interpreted as the risk in an investment), if the variance or distribution in the probability of the returns on an investment is high, it may indicate an unfavorable investment.  
* To describe, analyze and understand the behavior of a variable over time.  
* It allows comparisons between different groups of data.  

> López, J. (2019). Varianza - Definición, qué es y concepto | Economipedia. Economipedia. Retrieved February 25, 2020, as of https://economipedia.com/definiciones/varianza.html  
Riquelme, M. (2019). Varianza en Estadística (Uso, definición y formula) - Web y Empresas. Web y Empresas. Retrieved February 25, 2020, from https://www.webyempresas.com/varianza/  
