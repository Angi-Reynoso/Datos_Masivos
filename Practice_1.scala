// Practice 1
//1. Develop an algorithm in scala that calculates the radius of a circle
println("1. Radius of a circle")
var D: Double = 15
var R: Double = (D/2)
println("Operation = " + {D} + "/" + "2" + " = " + R)
println("")

//2. Develop an algorithm in scala that tells me if a number is prime
    // Prime numbers: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 
    // 53, 59, 61, 67, 71, 73, 79, 83, 89 y 97.
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
      
//3. Given the variable bird = "tweet", use string interpolation to
//   print "I am writing a tweet"
println("3. String interpolation")
val bird = "tweet"
val res = s"I am writing a ${bird}"
println(res)
println("")

//4. Given the variable message = "Hi Luke, I'm your father" use slice to extract the
//sequence "Luke"
val message = "Hi Luke, I'm your father"
message slice (3,7)

//5. What is the difference between value and a variable in scala?
// Value: Used to assign an immutable value (which cannot be changed, static)
// Variable: The value assigned can change

 
// 6. Given the tuple ((2,4,5), (1,2,3), (3,114,23))) return the number 3.1416
val tuple =((2,4,5),(1,2,3),(3.1416,23))
tuple ._3._1

