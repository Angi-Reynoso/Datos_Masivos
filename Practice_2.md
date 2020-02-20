## Instructions: 

1. Create a list called "list" with the elements "red", "white", "black".  
2. Add 5 more items to "list" "green", "yellow", "blue", "orange", "pearl".  
3. Bring the "list" "green", "yellow", "blue" items.  
4. Create a number array in the 1-1000 range in 5-in-5 steps. 
5. What are the unique elements of the "list" List (1,3,3,4,6,7,3,7) use conversion to sets.  
6. Create a mutable map called names that contain the following "José", 20, "Luis", 24, "Ana", 23, "Susana", "27".  
  6a. Print all the map keys.  
  6b. Add the following value to the map ("Miguel", 23).  

## Solutions: 

### Solution 1:
A List was created, which has the name "list".
~~~
val list = List ("red", "white", "black")
println (list)
~~~
### Solution 2:  
The operator ": :" is used to build lists, in addition to the form used in the previous exercise.
~~~
val list1 = "green" :: "yellow" :: "blue" :: "orange" :: "pearl" :: list
~~~
### Solution 3:
The `slice()` method returns the selected elements in an array, as a new array object.  
The `slice()` method selects the elements starting at the given start argument, and ends at, but does not include, the given end argument.
~~~
list.slice (0,3)
~~~
### Solution 4:
"Range" allows us to give you the range we want our arrangement to have.
~~~
Array.range (1,1000,5)
~~~
### Solution 5:
"toSet" gives us data that is not repeated or duplicated.
~~~
val list = List(1,3,3,4,6,7,3,7)
list.toSet
~~~
### Solution 6:
"Map# is a collection of key-value pairs. In other words, it is similar to dictionary.  
Keys are always unique while values need not be unique.  
Key-value pairs can have any data type. However, data type once used for any key and value must be consistent throughout.  
Maps are classified into two types: mutable and immutable.  
By default Scala uses immutable Map, in order to use mutable Map, we must import `scala.collection.mutable.Map` class explicitly.
~~~
val names = collection.mutable.Map (("Jose", 20), ("Luis", 24), ("Ana", 23), ("Susana", 27))
~~~
### Solution 6a:
"keys": This method returns an iterable containing each key in the map.
~~~
names.keys
~~~
### Solution 6b:
We can insert new key-value pairs in a mutable map using "+=" operator followed by new pairs to be added or updated.
~~~
names += ("Miguel" -> 23)
~~~
