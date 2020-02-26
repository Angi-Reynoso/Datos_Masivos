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
