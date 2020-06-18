import org.apache.spark.sql.SparkSession
val spar = SparkSession.builder().getOrCreate()
 
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
 
val dataset = spark.read.option("header","true").option("inferSchema","true").csv("bank-additional-full.csv")
 
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
 
val data = dataset.select(dataset("y").as("label"), $"age",$"duration",$"campaign",$"pdays",$"previous",$"emp_var_rate",$"cons_price_idx",$"cons_conf_idx",$"euribor3m",$"nr_employed")
 
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
 
val assembler = new VectorAssembler().setInputCols(Array("age","duration","campaign","pdays","previous","emp_var_rate","cons_price_idx","cons_conf_idx","euribor3m","nr_employed")).setOutputCol("features")
 
val df = assembler.transform(data)
 
import org.apache.spark.ml.feature.StringIndexer
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(df)
 
import org.apache.spark.ml.feature.VectorIndexer
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(df)
 
val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3), seed = 1234L)
 
val t1 = System.nanoTime
 
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
 
import org.apache.spark.ml.feature.IndexToString
val labelConverter = new IndexToString()
.setInputCol("prediction")
.setOutputCol("predictedLabel")
.setLabels(labelIndexer.labels)
 
import org.apache.spark.ml.Pipeline
val pipeline = new Pipeline()
.setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
 
val model = pipeline.fit(trainingData)
 
val predictions = model.transform(testData)
predictions.select("predictedLabel", "label", "features").show(5)
 
val evaluator = new MulticlassClassificationEvaluator()
.setLabelCol("indexedLabel")
.setPredictionCol("prediction")
.setMetricName("accuracy")
 
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))
 
val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
 
val duration = (System.nanoTime - t1) / 1e9d
