## The Fibonacci sequence 

The Fibonacci sequence, sometimes also known as the Fibonacci sequence or incorrectly as the Fibonacci series, is itself an infinite mathematical sequence. Constant of a series of natural numbers that add up to 2, from 0 and 1. Basically, the 
Fibonacci sequence is done by always adding the last 2 numbers (All the numbers present in the sequence are called the 
Fibonacci numbers) of as follows: <br>
0,1,1,2,3,5,8,13,21,34 ...<br>

Therefore in the following practice the development of the 6 fibonacci algorithms will be carried out.

`package math`  
This package contains methods for performing basic numeric operations such as elementary exponential, logarithmic, root and trigonometric functions.  
To make use of any of the methods in this package it is necessary to put "math." first, and then the name designated for the method; for example for square root "sqrt" is used.  

`import util.control.Breaks._`  
Scale library that allows you to stop a cycle while.  

**1st Algorithm: Recursive version descending**  
Methods are always defined with the reserved word "def": `def method (variable: data type): data type = `  

*If the number entered in the function is less than 2, the number entered will be returned to n.*  
*In any case that is greater than 2, the function will perform a series of operations, and return the result depending 
on the number entered.*  
~~~~
def algorithm1(n: Int): Int =
{   
    if (n<2)
    {
        return n
    }
    else
    {
        return algorithm1 (n-1) + algorithm1(n-2)
    }
}
println(algorithm1(1))
~~~~

**2nd Algorithm: Version with explicit formula**  
`math.sqrt`: Returns the square root of a Double value.  
`math.pow`: Returns the value of the first argument raised to the power of the second argument.  
~~~~
def algorithm2(n: Double): Double =  
{   
    if (n<2)
    {
        return n
    }
    else
    {
        var phi = ((1 + math.sqrt(5)) / 2)
        var j = ((math.pow(phi,n) - math.pow((1 - phi),n)) / (math.sqrt(5)))
        return j
    }
}
println(algorithm2(2))
~~~~

**3rd Algorithm: Iterative version**  
A "for" loop is a repetition control structure which allows us to write a loop that is executed a specific number of times.  
The loop enables us to perform n number of steps together in one line.  
~~~~
def algorithm3 (n: Int): Int =
{
var a = 0
var b = 1
var c = 0
    for (k <- 1 to n)
    {   
        println(a)
        c = b + a
        a = b
        b = c
    }
    return(a)
}
println(algorithm3(3))  
~~~~

**4th Algorithm: Iterative version-2 variables**  
~~~~
def algorithm4(n: Int): Int =
{
    var a = 0
    var b = 1
    for(k <- 1 to n)
        {
            b = b + a
            a = b - a
        }
        return(a)
}
println(algorithm4(3))
~~~~

**5th Algorithm: Iterative vector version**  
"Range" allows us to give you the range we want our arrangement to have.  
~~~~
def algorithm5(n: Int): Int = 
{   
    if (n<2)
    {
        return n
    }
    else
    {
        val arr = Array.range(0, n+2) 
        arr(0) = 0
        arr(1) = 1
        for(k <- 2 to n+1)
        {
            arr(k) = arr(k-1) + arr(k-2)
        }
        return arr(n)
    }
}
println(algorithm5(5))
~~~~

**6th Algorithm: Version Divide and Conquer**  
A "while" loop repeats a statement or group of statements while a given condition is true. It tests the condition before executing the loop body.  
A while loop statement repeatedly executes a target statement as long as a given condition is true.  
~~~~
def algorithm6(n: Int): Int = 
{   
    if (n<=0)
    {
        return 0
    }
    var i = n-1
    var a = 1
    var b = 0
    var c = 0
    var d = 1
    var t = 0
    while(i > 0){
        if (i % 2 == 1)
        {
            t = d*(b+a) + c*b
            a = d*b + c*a
            b = t
        }
        t = d*(2*c + d)
        c = c*c + d*d
        d = t
        i = i/2
    }
    return a + b    
}
println(algorithm6(6))
~~~~
