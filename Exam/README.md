## Instrucciones  
Desarrolle las siguientes instrucciones en Spark con el lenguaje de programación Scala.  

## Objetivo  
El objetivo de este examen práctico es tratar de agrupar los clientes de regiones específicas de un distribuidor al mayoreo. Esto en base a las ventas de algunas categorías de productos.  
La fuente de los datos se encuentra en el repositorio: https://github.com/jcromerohdz/BigData/blob/master/Spark_clustering/Wholesale%20customers%20data.csv  

## Pasos  
**1. Importar Sesión**  
Importar una simple sesión Spark.  
~~~
import org.apache.spark.sql.SparkSession
~~~  

**2. Minimizar errores**  
Utilizar las líneas de código para minimizar errores.  
~~~
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
~~~  

**3. Crear Sesión**  
Crear una instancia de la sesión Spark.  
~~~
val spark = SparkSession.builder().getOrCreate()
~~~  

**4. Importar Kmeans**  
Importar la librería de Kmeans para el algoritmo de agrupamiento.  
~~~
import org.apache.spark.ml.clustering.KMeans
~~~  

**5. Cargar Dataset**  
Cargar el dataset de Wholesale Customers Data.  
~~~
val dataset = spark.read.option("header","true").option("inferSchema","true").csv("Wholesale customers data.csv")
~~~  
> Utilizamos `option("header","true")` para conservar los nombres de las columnas del dataset original.  
> Utilizamos `.option("inferSchema","true")` para obtener el tipo de dato de cada columna según el dataset original.  

**6. Seleccionar columnas**  
Seleccionar las siguientes columnas: Fresh, Milk, Grocery, Frozen, Detergents_Paper y Delicassen, y llamar a este conjunto feature_data.  
~~~
val feature_data = dataset.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen")
~~~  
> Seleccionamos las columnas mencionadas anteriormente con el nombre del dataset ('dataset') y la función `select`.  
> Para indicar que se trata de una columna es necesario poner primero el signo '$', seguido del nombre de la columna entre comillas dobles.  

**7. Importar Vector Assembler y Vectors.**  
~~~
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
~~~  

**8. Crear objeto Vector Assembler**  
Crear un nuevo objeto Vector Assembler para las columnas de características (columna features) como un conjunto de entrada, recordando que no hay etiquetas (columna label).  
~~~
val assembler = new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")).setOutputCol("features")
~~~  
> Utilizamos la función `VectorAssembler()` para crear el objeto. Enseguida utilizamos `setInputCols` para especificar las columnas de entrada, y `setOutputCol` para indicar el nombre de la columna de salida.  
> En este caso los nombres de las columnas se declaran en forma de arreglo, por lo cual no es necesario utilizar el signo '$' como en uno de los pasos anteriores.  

**9. Transformación de Datos**  
Utilizar el objeto assembler para transformar feature_data.  
~~~
val training_data = assembler.transform(feature_data).select("features")
~~~  
> Utilizamos la función `transform` para transformar los datos seleccionados (features), y se guarda el resultado dentro de la variable training_data.  

**10. Crear Modelo**  
Crear un modelo Kmeans con K=3.  
Se ajusta el modelo a los datos (training_data).  
~~~
val kmeans = new KMeans().setK(3).setSeed(1L)
val model = kmeans.fit(training_data)
~~~  
> Se utiliza la función `KMeans()` para crear el modelo. Enseguida se utiliza `setK()` para establecer el número de 'K' o grupos a crear por el modelo, y `setSeed()` para dar aleatoriedad a los datos a seleccionar.  
> Una vez creado el modelo, este se ajusta a los datos (training_data) utilizando la función `fit`.  

**11. Evaluar Grupos**  
Evaluar los grupos utilizando Within Set Sum of Squared Errors (WSSSE) e imprimir los centroides.  
<br>
_Within Set Sum of Squared Errors_ es igual a:  
* La sumatoria de las distancias al cuadrado de cada observación al centroide de su grupo.  
* La medida para indicar cuán bien los centroides representan a los miembros de su grupo.  

En cada iteración que realiza el modelo, se intenta reducir el valor obtenido de esta suma.  
<br>
Un _Centroide_:
* Es el promedio de los puntos que componen a un grupo.  
* Se le denomina así al representante de cada grupo.  
~~~
val WSSSE = model.computeCost(training_data)
println(s"Within Set Sum of Squared Errors = $WSSSE")

println("Cluster Centers: ")
model.clusterCenters.foreach(println)
~~~  
> Se utiliza la función `computeCost` para calcular el WSSSE, indicando los datos a usar (training_data), y se imprime el resultado.  
> Se utiliza la función `clusterCenters` para obtener los centroides de cada grupo creado por el modelo, y se usa `foreach` para imprimir en forma de arreglos separados los resultados de cada grupo; como se puede observar en los resultados a continuación.  

Resultados:  
~~~
Within Set Sum of Squared Errors = 8.095172370767671E10
Cluster Centers: 
[7993.574780058651,4196.803519061584,5837.4926686217,2546.624633431085,2016.2873900293255,1151.4193548387098]
[9928.18918918919,21513.081081081084,30993.486486486487,2960.4324324324325,13996.594594594595,3772.3243243243246]
[35273.854838709674,5213.919354838709,5826.096774193548,6027.6612903225805,1006.9193548387096,2237.6290322580644]
~~~  

### Conclusión  
Como resultado del modelo se obtuvo un total de 6 iteraciones; esto se puede verificar al contar el número de centroides obtenidos para cada grupo o "K", los cuales se encuentran representados en forma de arreglos (encerrados entre '[ ]').  
