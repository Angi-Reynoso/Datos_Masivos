import org.apache.spark.sql.SparkSession
val spar = SparkSession.builder().getOrCreate()
 
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
 
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.evaluation.MulticlassMetrics 
 
val dataset  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("bank-additional-full.csv")
 
val coly = when($"y".contains("yes"), 1.0).otherwise(0.0)
val df = dataset.withColumn("y", coly)
 
val data = df.select(df("y").as("label"), $"age",$"duration",$"campaign",$"pdays",$"previous",$"emp_var_rate",$"cons_price_idx",$"cons_conf_idx",$"euribor3m",$"nr_employed")
 
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
 
val assembler = (new VectorAssembler().setInputCols(Array("age","duration","campaign","pdays","previous","emp_var_rate","cons_price_idx","cons_conf_idx","euribor3m","nr_employed")).setOutputCol("features"))
 
val Array(training, test) = data.randomSplit(Array(0.7, 0.3), seed = 1234L)
 
val t1 = System.nanoTime
 
import org.apache.spark.ml.Pipeline
 
val lr = new LogisticRegression().setMaxIter(100)
 
val pipeline = new Pipeline().setStages(Array(assembler,lr))
val model = pipeline.fit(training)
val results = model.transform(test)
val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
val metrics = new MulticlassMetrics(predictionAndLabels)
 
println("Confusion matrix:")
println(metrics.confusionMatrix)
 
metrics.accuracy
 
val duration = (System.nanoTime - t1) / 1e9d

