## Instructions: 

1. Create a list called "list" with the elements "red", "white", "black".
2. Add 5 more items to "list" "green", "yellow", "blue", "orange", "pearl". Bring the "list" "green", "yellow", "blue" items.
3. Create a number array in the 1-1000 range in 5-in-5 steps. What are the unique elements of the List list (1,3,3,4,6,7,3,7) to     use conversion to sets.  
4. Create a mutable map called names that contain the following "José", 20, "Luis", 24, "Ana", 23, "Susana", "27".
5. Print all the map keys.
6. Add the following value to the map ("Miguel", 23).

## Solutions: 

### Solution 1:
~~~
a list was created, which has the name "list"
val list = List ("red", "white", "black")
println (list)
~~~
### Solution 2:
~~~
val list1 = "green" :: "yellow" :: "blue" :: "orange" :: "pearl" :: list
~~~
### Solution 3:
The slice() method returns the selected elements in an array, as a new array object.
The slice() method selects the elements starting at the given start argument, and ends at, but does not include, the given end argument.
~~~
list.slice (0,3)
~~~
### Solution 4:
"Range" allows us to give you the range we want our arrangement to have
~~~
Array.range (1,1000,5)
~~~

### Solution 5:
"toSet" gives us data that is not repeated or duplicated
~~~
val list = List(1,3,3,4,6,7,3,7)
list.toSet
~~~

### Solution 6:

~~~
val names = collection.mutable.Map (("Jose", 20), ("Luis", 24), ("Ana", 23), ("Susana", 27))
~~~
### Solution 6a:
~~~
names.keys
~~~
### Solution 6b:
~~~
names += ("Miguel" -> 23)
~~~
