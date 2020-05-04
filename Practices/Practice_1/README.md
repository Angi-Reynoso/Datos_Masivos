
## LINEAR REGRESSION EXERCISE

## Instructions: 

1. Import LinearRegression

2. Optional: Use the following code to configure errors

3. Start a simple Spark Session

4. Use Spark for the Clean-Ecommerce csv file.

5. Print the schema on the DataFrame.

6. Print an example row from the DataFrane.

 ## Set up the DataFrame for Machine Learning

7. Transform the data frame so that it takes the form of ("label", "features")

8. Import VectorAssembler and Vectors

9. Rename the Yearly Amount Spent column as "label"  
9.1 Also from the data take only the numerical column  
9.2 Leave all this as a new DataFrame called df  

10. Use the VectorAssembler object to convert the input columns of the df  
10.1 to a single output column of an array named "features"  
10.2 Configure the input columns from where we are supposed to read the values.  
10.3 Call this a new assambler.  

11. Use the assembler to transform our DataFrame to two columns: label and features

12. Create an object for line regression model.

13. Fit the model for the data and call this model lrModel


14. Print the coefficients and intercept for the linear regression

15. Summarize the model on the training set print the output of some metrics!
Use our model's .summary method to create an object called trainingSummary

16. Show the residuals values, the RMSE, the MSE, and also the R ^ 2.


## Solutions: 
