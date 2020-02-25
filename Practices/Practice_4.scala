
//1.  
df.select(avg($"Close")).show()

//2.  
df.select(collect_list("Close")).show() 

//3.  
df.select(collect_set("Close")).show()

//4.  
df.select(first("Close")).show()

//5.  
df.select(last("Close")).show()

//6.  
df.select(max("Close")).show()

//7.  
df.select(min("Close")).show()

//8.  
df.select(skewness("Low")).show()

//9.  
df.select(sum("High")).show()

//10. 
df.select(kurtosis("Low")).show()

//11. 
df.select(countDistinct("Volume")).show()

//12. 
df.select(mean("High")).show()

//13. 
df.select(var_pop("Close")).show()

//14. 
df.select(concat($"High", $"Low")).show()

//15. 
df.select(reverse($"Date")).show()

//16. 
df.select(current_date()).show()

//17. 
df.select(dayofmonth($"Date")).show()

//18. 
df.select(dayofweek($"Date")).show()

//19. 
df.select(month($"Date")).show()

//20. 
df.select(year($"Date")).show()
