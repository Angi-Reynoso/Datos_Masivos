
## Instructions  
1. Develop an algorithm in scala that calculates the radius of a circle.
2. Develop an algorithm in scala that tells me if a number is is a prime number.
3. Given the variable bird = "tweet", use string interpolation to print "I am writing a tweet".
4. Given the variable message = "Hi Luke, I'm your father!" use slice to extract the "Luke" sequence.
5. What is the difference between value and a variable in scala?.
6. Given the tuple (2,4,5,1,2,3,3.1416,23) return the number 3.1416.

## Solutions: 

### Solution 1
Creation of variables: `var Name: Data type = value`.  
To print values: `println()`.
~~~~
println("1. Radius of a circle")
var D: Double = 15
var R: Double = (D/2)
println("Operation = " + {D} + "/" + "2" + " = " + R)
println("")
~~~~
### Solution 2  
The "if" statement consists of a Boolean expression followed by one or more statements.  
To insert a variable into a string use: `s"${variable} rest of the text"`.  
The operator "| |" is used to indicate option (OR).  
The operator "%" is used to return the rest of a division.  
~~~~
println("2. Prime numbers")
var num: Double = 1
if(num == 2 || num == 3 || num == 5 || num == 7){
           var message = s"${num} it's a prime number"
           println(message)
        } else if(num % 2 == 0){
           var message = s"${num} it's not a prime number"
           println(message)
        } else if(num % 3 == 0){
           var message = s"${num} it's not a prime number"
           println(message)
        } else if(num % 5 == 0){
           var message = s"${num} it's not a prime number"
           println(message)
        } else if(num % 7 == 0){
           var message = s"${num} it's not a prime number"
           println(message)
        } else if(num == 1){
           var message = s"${num} it's not a prime number"
           println(message)
        } else {
           var message = s"${num} it's a prime number"
           println(message)
        }
println("")
~~~~  
### Solution 3
String interpolation: it allows to include references to variables directly in “processed” text strings.  
For example: `s"Text ${var}"`  
~~~~
println("3. String interpolation")
val bird = "tweet"
val res = s"I am writing a ${bird}"
println(res)
println("")
~~~~
### Solution 4
The slice() method returns the selected elements in an array, as a new array object.  
The slice() method selects the elements starting at the given start argument, and ends at, but does not include, the given end argument.  
~~~~
val message = "Hi Luke, I'm your father"
message slice (3,7)
~~~~
### Solution 5
To make comments on scala use "//".
~~~~
// Value: Used to assign an immutable value (which cannot be changed, static)
// Variable: The value assigned can change
~~~~
### Solution 6
A "tuple" is a value that contains a fixed number of elements, each with a distinct type. Tuples are immutable.  
One way of accessing tuple elements is by position. The individual elements are named `_1`, `_2`, and so forth.  
~~~~
val tuple =((2,4,5),(1,2,3),(3.1416,23))
tuple ._3._1
~~~~
