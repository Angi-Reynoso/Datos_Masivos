// Assessment 1
//1. Develop an algorithm in scala that calculates the radius of a circle
println("1. Radius of a circle")
var diametro: Double = 15
var radio: Double = (diametro/2)
println("Operacion = " + {diametro} + "/" + "2" + " = " + radio)
println("")

//2. Develop an algorithm in scala that tells me if a number is prime
    // Prime numbers: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 
    // 53, 59, 61, 67, 71, 73, 79, 83, 89 y 97.
println("2. Prime numbers")
var numero: Double = 1
if(numero == 2 || numero == 3 || numero == 5 || numero == 7){
           var mensaje = s"${numero} es un numero primo"
           println(mensaje)
        } else if(numero % 2 == 0){
           var mensaje = s"${numero} no es un numero primo"
           println(mensaje)
        } else if(numero % 3 == 0){
           var mensaje = s"${numero} no es un numero primo"
           println(mensaje)
        } else if(numero % 5 == 0){
           var mensaje = s"${numero} no es un numero primo"
           println(mensaje)
        } else if(numero % 7 == 0){
           var mensaje = s"${numero} no es un numero primo"
           println(mensaje)
        } else if(numero == 1){
           var mensaje = s"${numero} no es un numero primo"
           println(mensaje)
        } else {
           var mensaje = s"${numero} es un numero primo"
           println(mensaje)
        }
println("")
      
//3. Given the variable bird = "tweet", use string interpolation to
//   print "Estoy ecribiendo un tweet"
println("3. String interpolation")
val bird = "tweet"
val imprimir = s"Estoy escribiendo un ${bird}"
println(imprimir)
println("")

//4. Given the variable message = "Hola Luke yo soy tu padre" use slilce to extract the
//sequence "Luke"
val message = "Hola Luke yo soy tu padre"
message slice (5,9)

//5. What is the difference between value and a variable in scala?
// Value: Used to assign an immutable value (which cannot be changed, static)
// Variable: The value assigned can change

 
// 6. Given the tuple ((2,4,5), (1,2,3), (3,114,23))) return the number 3.1416
val tuple =((2,4,5),(1,2,3),(3.1416,23))
tuple ._3._1

