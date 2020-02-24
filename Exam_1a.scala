//Declaration of function 'diagonalDifference'
def diagonalDifference(arr1:Array[Array[Int]]): Int={
    //Calculate array size
    var length = arr1.length
    //Calculate sum of diagonals
    var d1 = 0
    var d2 = 0
    //Cycle to calculate the sum of the N values of the diagonals
    for(i <- 0 to (length - 1)){
        d1 = d1 + arr1(i)(i)
        d2 = d2 + arr1(length - 1 - i)(i)
    }
    //Calculate the absolute value of the difference between d1 and d2
    return math.abs(d1 - d2)
}

//Declaration of arrangements with square matrices
val arr = Array(Array(11,2,4),Array(4,5,6),Array(10,8,-12))
val arr = Array(Array(1,2,3),Array(4,5,6),Array(7,8,9))
val arr = Array(Array(1,2),Array(4,6))
val arr = Array(Array(1,2,3,4),Array(5,6,7,8),Array(9,10,11,12),Array(13,14,15,16))

//Send arrangement to function
diagonalDifference(arr)
