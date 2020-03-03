# Exam: Unit 1 - Part 1  
<br>

~~~~
//Declaration of function 'diagonalDifference'  
def diagonalDifference (arr1: Array [Array [Int]]): Int = {
~~~~
Here the "diagonalDifference" function is declared to receive the array and process it until the absolute diagonal 
difference is obtained.  
`Array [Array [Int]]`: is used to receive an array consisting of arrangements.  

~~~~
    // Calculate array size  
    var length = arr1.length  
~~~~
".length": is a function of the arrangements, which allows to calculate the size of them.  

~~~~
    // Calculate sum of diagonals  
    var d1 = 0  
    var d2 = 0  
~~~~
The variables to save the sum of the diagonals are initialized at 0.  

~~~~
    // Cycle to calculate the sum of the N values of the diagonals  
    for (i <- 0 to (length - 1)) {  
~~~~
The 'for' cycle is declared where the counter 'i' starts at 0, and will take values until reaching the size of the 
array minus one.  

~~~~
        d1 = d1 + arr1 (i) (i) 
~~~~
The variable 'd1' will save the value of the current position of the array: `arr1 (i) (i)`, and will add the other 
values as it progresses.  

~~~~
        d2 = d2 + arr1 (length - 1 - i) (i)  
~~~~
In the case of the second diagonal, as the positions to be obtained are reversed, it will be necessary to start with 
the largest value in the arrangement, and then decrease: `arr1 (length - 1 - i) (i)`.  

~~~~
    }  
    // Calculate the absolute value of the difference between d1 and d2  
    return math.abs (d1 - d2)
~~~~
The `math.abs` function is used to calculate the absolute value of a variable, in this case the difference between 
the diagonals (d1 and d2).
~~~~
}  
~~~~

~~~~
//Declaration of arrangements with square matrices  
val arr = Array (Array (11,2,4), Array (4,5,6), Array (10,8, -12))
val arr = Array (Array (1,2,3), Array (4,5,6), Array (7,8,9))
val arr = Array (Array (1,2), Array (4,6))
val arr = Array (Array (1,2,3,4), Array (5,6,7,8), Array (9,10,11,12), Array (13,14,15,16))
~~~~
These are some tests done with square matrices of different sizes to verify that the function works.  

~~~~
//Send arrangement to function  
diagonalDifference (arr)  
~~~~
Finally, the function is called by its name: 'diagonalDifference', and in parentheses the parameter (s) is sent, in this 
case the array with the **square matrix**: `(arr)`.  
<br>
<br>
**Square Matrix:**  
A square matrix is a very basic matrix typology that is characterized by having the same order of both rows and columns.  
In other words, a square matrix has the same number of rows (n) and the same number of columns (m).  
We can create infinite combinations of square matrices as long as we respect the restriction that the number of columns 
and rows has to be the same.  
