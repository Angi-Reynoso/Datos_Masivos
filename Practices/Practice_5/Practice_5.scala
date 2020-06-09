//Aggregate Functions
//1. def approx_count_distinct(columnName: String, rsd: Double): Column 
df.select(approx_count_distinct("Sales")).show()

//2. def avg(columnName: String): Column 
df.select(avg("Sales")).show()

//3. def collect_list(columnName: String): Column 
df.select(collect_list("Sales")).show()

//4. def covar_pop(columnName1: String, columnName2: String): Column 
df.select(covar_pop("Sales", "Company")).show()

//5. def first(columnName: String, ignoreNulls: Boolean): Column 
df.select(first("Sales")).show()

//6. def kurtosis(columnName: String): Column 
df.select(kurtosis("Sales")).show()

//7. def last(columnName: String, ignoreNulls: Boolean): Column 
df.select(last("Sales")).show()

//8. def skewness(columnName: String): Column 
df.select(skewness("Sales")).show()

//9. def corr(columnName1: String, columnName2: String): Column  
df.select(corr("Sales","Company")).show()
    
//10. def count(columnName: String): TypedColumn[Any, Long]  
df.select(count("Sales")).show()

//11. def covar_samp(columnName1: String, columnName2: String): Column  
df.select(covar_samp("Sales","Company")).show()

//12. def stddev_pop(columnName: String): Column    
df.select (stddev_pop("Sales")).show()

//13. def stddev_samp(columnName: String): Column    
df.select (stddev_samp("Sales")).show()

//14. def var_pop(columnName: String): Column    
df.select (var_pop("Sales")).show()  

//15. def var_samp(columnName: String): Column   
df.select (var_samp("Sales")).show()

