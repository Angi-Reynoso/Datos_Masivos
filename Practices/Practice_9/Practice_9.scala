package org.apache.spark.examples.ml
import  org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import  org.apache.spark.mllib.util.MLUtils

import org.apache.spark.sql.SparkSession

val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")

val Array(training, test) = data.randomSplit(Array(0.6, 0.4))

val model = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial")

val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))

val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()
model.save(sc, "target/tmp/myNaiveBayesModel")

val sameModel = NaiveBayesModel.load(sc, "target/tmp/myNaiveBayesModel")
