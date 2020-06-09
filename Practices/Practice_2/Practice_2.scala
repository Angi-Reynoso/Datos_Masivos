// 1. Create a list called "list" with the elements "red", "white", "black"
val list = List ("red", "white", "black")
println (list)

// 2. Add 5 more items to "list" "green", "yellow", "blue", "orange", "pearl"
val list1 = "green" :: "yellow" :: "blue" :: "orange" :: "pearl" :: list

// 3. Bring the "list" "green", "yellow", "blue" items
list.slice (0,3)

// 4. Create a number array in the 1-1000 range in 5-in-5 steps
Array.range (1,1000,5)

// 5. What are the unique elements of the "list" List (1,3,3,4,6,7,3,7) use conversion to sets
val list = List(1,3,3,4,6,7,3,7)
list.toSet

// 6. Create a mutable map called names containing the following
// "Jose", 20, "Luis", 24, "Ana", 23, "Susana", "27"
val names = collection.mutable.Map (("Jose", 20), ("Luis", 24), ("Ana", 23), ("Susana", 27))

// 6 a. Print all map keys
names.keys

// 6 b. Add the following value to the map ("Miguel", 23)
names += ("Miguel" -> 23)
