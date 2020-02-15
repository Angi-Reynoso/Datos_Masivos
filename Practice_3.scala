
// 1st Algorithm 
//Recursive version descending
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


// 2nd Algorithm 
//Version with explicit formula 


// 3rd Algorithm 
//Iterative version
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


// 4th Algorithm 
//iterative version-2 variables
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

// 5th Algorithm
//Iterative vector version (Complexity)


//6th Algorithm 
//Version Divide and Conquer
