import org.apache.spark.sql.SparkSession
val spar = SparkSession.builder().getOrCreate()

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

val df = spark.read.option("header", "true").option("inferSchema","true")csv("iris.csv")
df.na.drop().show()

import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

df.columns

df.printSchema()

df.show(5)

df.head(5)
for(row <- df.head(5)){
    println(row)
}

df.describe().show()



val data = df.select(df("species").as("label"), $"sepal_length", $"sepal_width", $"petal_length", $"petal_width")

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

val assembler = new VectorAssembler().setInputCols(Array("sepal_length", "sepal_width","petal_length","petal_width")).setOutputCol("features")
val features = assembler.transform(data)
features.show()

import org.apache.spark.ml.feature.StringIndexer
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(features)

import org.apache.spark.ml.feature.VectorIndexer
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(features)

val splits = features.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)


val layers = Array[Int](4, 5, 4, 3)

val trainer = new MultilayerPerceptronClassifier()
.setLayers(layers)
.setLabelCol("indexedLabel")
.setFeaturesCol("indexedFeatures")
.setBlockSize(128)
.setSeed(1234L)
.setMaxIter(100)

import org.apache.spark.ml.feature.IndexToString

val labelConverter = new IndexToString()
.setInputCol("prediction")
.setOutputCol("predictedLabel")
.setLabels(labelIndexer.labels)

import org.apache.spark.ml.Pipeline

val pipeline = new Pipeline()
.setStages(Array(labelIndexer, featureIndexer, trainer, labelConverter))

val model = pipeline.fit(train)


val predictions = model.transform(test)
val predictionAndLabels = predictions.select("prediction", "label")

val evaluator = new MulticlassClassificationEvaluator()
.setLabelCol("indexedLabel")
.setPredictionCol("prediction")
.setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
