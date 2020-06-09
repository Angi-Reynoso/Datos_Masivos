# Index #  
* [Homework 1: Main Types of Machine Learning Algorithms](#homework-1)  
* [Homework 2: VectorAssembler, Vectors, and RootMeanSquaredError](#homework-2)  
* [Homework 3: Pipeline and Confusion Matrix](#homework-3)  

<br>

---

<br>

# Homework 1  
### Main Types of Machine Learning Algorithms

1. **SUPERVISED LEARNING**  
In supervised learning, algorithms work with “labeled data”, trying to find a function that, given the input variables, assigns them the appropriate output tag. The algorithm trains with a "history" of data and thus "learns" to assign the appropriate output tag to a new value, that is, it predicts the output value.  
  
    Supervised learning is often used in:  
    * Classification problems (digit identification, diagnostics, or detection of identity fraud).  
    * Regression problems (weather predictions, life expectancy, growth, etc.).  

    These two main types of supervised learning, classification and regression, are distinguished by the type of target variable. In the cases of classification, it is of the categorical type, while in the cases of regression, the objective variable is of the numerical type.  

    The most common algorithms that apply to supervised learning are:  
    
      **a) Decision trees.**  
      A decision tree is a decision support tool that uses a graph or model similar to a decision tree and its possible consequences, including the results of fortuitous events, resource costs, and profit. They have an appearance like this:  
      <img src="https://image.slidesharecdn.com/l7decision-treetable-130318112451-phpapp01/95/l7-decision-tree-table-21-638.jpg?cb=1363605932" alt="image" width="35%">  
        
      From a business decision-making point of view, a decision tree is the minimum number of yes / no questions that one has to ask, to assess the probability of making a correct decision, most of the time. This method allows you to approach the problem in a structured and systematic way to reach a logical conclusion.  
    
      **b) Naïve Bayes classification.**  
      This algorithm, based on already known elements, is capable of classifying a new element without knowing which group it belongs to.  
      This algorithm is usually used in the following cases:  
      * Autonomous cars: for their ability to determine the speed that the vehicle must carry, starting for example: the state in which the road is or the slope of the same.  
      * Find emails considered spam after analyzing the words that appear in the email.  
      * Classification of articles by the topic they are talking about.  
      These types of classification algorithms are based on Bayes' theorem and classify each value as independent of any other. This allows predicting a class or category based on a given set of characteristics, using probability.  
      Despite its simplicity, the classifier works surprisingly well and is often used because it outperforms more sophisticated classification methods.  
      
      **c) Regression for least squares.**  
      Linear regression can be thought of as the task of fitting a straight line through a set of points. There are several possible strategies for doing this, and the "ordinary least squares" strategy goes like this: you can draw a line and then, for each of the data points, measure the vertical distance between the point and the line, and add them together; The fitted line would be the one in which this sum of distances is as small as possible.  
      <img src="https://1.bp.blogspot.com/-JB9NcFsc_l8/VDbWKSish8I/AAAAAAAAABg/x3SYHQt37L4/s1600/350px-Linear_regression.svg.png" alt="image" width="35%">  
      
      Linear refers to the type of model you are using to fit the data, while least squares refer to the type of error metric you are minimizing.

      **d) Logistic regression.**  
      Logistic regression is a powerful statistical way to model a binomial result with one or more explanatory variables. Measure the relationship between the categorical dependent variable and one or more independent variables by estimating the probabilities using a logistic function, which is the cumulative logistic distribution.  
      <img src="https://www.researchgate.net/profile/Jack_Baker3/publication/228974449/figure/fig5/AS:668748008681483@1536453358991/An-example-of-prediction-of-the-probability-of-collapse-using-logistic-regression-applied.png" alt="image" width="35%">  

      **e) Support Vector Machines (SVM).**  
      SVM is a binary classification algorithm. Given a set of points of 2 types at the N-dimensional location, SVM generates a dimensional (N-1) hyperplane to separate those points into 2 groups. Let's say you have some points of 2 types on a piece of paper that are linearly separable. SVM will find a straight line separating those points into 2 types and located as far as possible from all those points.  
      <img src="https://miro.medium.com/max/921/1*06GSco3ItM3gwW2scY6Tmg.png" alt="image" width="35%">  

      In terms of scale, some of the biggest problems that have been solved using SVMs (with appropriately modified implementations) are on-screen advertising, human splice site recognition, image-based gender detection, large-scale image classification.  

      **f) Ensemble Methods.**  
      Ensemble methods are learning algorithms that build a set of classifiers and then classify new data points by taking a weighted vote of their predictions. The original set method is Bayesian averaging, but the latest algorithms include encoding output correction error.  

<br>

2. **UNSUPERVISED LEARNING**  
Unsupervised learning takes place when "tagged" data is not available for training. We only know the input data, but there are no output data that correspond to a certain input. Therefore, we can only describe the structure of the data, to try to find some type of organization that simplifies the analysis. Therefore, they have an exploratory character.  
This type of algorithm is useful for discovering relationships that are implicit in a data set but are not known. That is, it allows considering that several elements belong to the same group or to different groups thanks to the study of their characteristics.  

    Unsupervised learning is often used in:  
    * Clustering problems  
    * Co-occurrence clusters  
    * Profiling or profiling  

    However, problems involving tasks of finding similarity, predicting links, or reducing data may or may not be supervised.  

    The most common types of algorithm in unsupervised learning are: 
    
    **a) Clustering algorithms.**  
    Clustering is the task of grouping a set of objects such that the objects in the same group (cluster) are more similar to each other than to those of other groups.  
    These algorithms are used to group together the elements that are most similar to each other. That is, they are algorithms that group the elements based on some specific characteristic.  

    In this group we find different types of algorithms, such as:  
    * Centroid-based algorithms: these algorithms, if we represent the elements in a point graph, calculate the midpoint of them that minimizes the distances. K-means is such an algorithm.  
    * Density-based algorithms: These algorithms seek to group the points by proximity to the surrounding points. An algorithm that is included in this family would be K-NN (k-nearest neighbors) or K nearest neighbors.  

    As additional information, mention the existence of Connectivity-based algorithms, Probabilistic, Dimensionality Reduction and Neural networks that would also fall within this group.  

    **b) Principal Component Analysis.**  
    It is a mathematical method used to reduce the number of variables so that we have the minimum number of new variables and that they represent all the old variables in the most representative way possible. That is, if the number of variables is reduced to two or three new ones, the original data can be represented in the plane or in a 3-dimensional graph and, thus, a summary of our data is displayed graphically. The simple fact of having the data in a visible way greatly simplifies understanding what may be happening and helps to make decisions.  
    <img src="https://miro.medium.com/fit/c/1838/551/0*5Iaw94wlYCTp0GuK.png" alt="image" width="58%">  

    Some of the PCA applications include compression, data simplification for easier learning, visualization. Keep in mind that domain knowledge is very important when choosing whether to go ahead with PCA or not. Not suitable in cases where the data is noisy (all PCA components have a fairly high variance).  
    
    **c) Singular Value Decomposition.**  
    The Singular Value Decomposition (hereinafter SVD) is a matrix factorization technique that allows decomposing a matrix A into other matrices U, S, V.  
    
    It should be clarified that there is a property applied to SVD focused on recommendation systems, this is that by reducing the number of singular values of the matrix S to the first k values, an approximation of the original matrix A will be obtained, which allows it to be reconstructed from the reduced versions of the other matrices making a certain mistake but
    decreasing the size.  
    
    This important property is derived from the Eckart-Young theorem that addresses the best approximation to the original matrix A, obtaining it by setting the n smallest singular values to 0, thus reducing the matrices to the number of non-null singular values that the matrix S has. This then results in the transformation of large amounts of data in their reduced representation, being therefore a very important property that allows to considerably reduce the computation time and memory usage for the three matrices.  
    
    **d) Independent Component Analysis.**  
    ICA is a statistical technique for revealing hidden factors underlying sets of variables, measurements, or random signals. ICA defines a generative model for the observed multivariate data, which is usually given as a large sample database. In the model, the data variables are assumed to be linear mixtures of some unknown latent variables, and the mixing system is also unknown. Latent variables are assumed to be non-Gaussian and mutually independent, and are called independent components of the observed data.  
    <img src="https://66.media.tumblr.com/6585ad06a5d2abe875e866bb4bc336da/tumblr_inline_omft2fnIeD1ugmnhq_540.png" alt="image" width="38%">   
    
    ICA is related to PCA, but it is a much more powerful technique that is able to find the underlying factors of sources when these classic methods fail completely. Its applications include digital images, document databases, economic indicators and psychometric measurements.  
    
<br>
    
3. **REINFORCED LEARNING**  
This type of learning is based on improving the model's response using a feedback process. The algorithm learns by observing the world around it. Your input information is the feedback you get from the outside world in response to your actions. Therefore, the system learns on a trial-error basis.  

    It is not a type of supervised learning, because it is not based strictly on a set of labeled data, but on monitoring the response to the actions taken. Nor is it unsupervised learning, since when we model our “apprentice” we know in advance what the expected reward is.  

    Reinforced Learning aims to build models that increase performance based on the result or reward generated by each interaction carried out. This reward is the product of a correct action or returned data set that goes into a specific measure. The agent model uses the reward as an adjustment parameter in its behavior for future actions, so that the new action also meets the objective or correct action and thus obtain a maximum reward.  

    Reinforced learning is publicly recognized for being applied in the AlphaZero program by technology developer Deep Mind. Its programming allowed this Machine Learning agent to know all the possible combinations and plays on a chess board and beat the Stockfish computerized chess engine after only 4 hours of learning.  
    
    * **Neural network algorithms**  
    An artificial neural network (RNA) comprises units arranged in a series of layers, each of which connects to the adjacent layers. RNAs are inspired by biological systems, such as the brain, and how they process information.  
    Thus, they are essentially a large number of interconnected processing elements, working in unison to solve specific problems.  
    They also learn by example and experience, and are extremely useful for modeling nonlinear relationships in high-dimensional data, or where the relationship between the input variables is difficult to understand.  

    * **Deep Learning Algorithms**  
    They are the evolution of Artificial Neural Networks that take advantage of the cheaper technology and the greater execution capacity, memory and disk to exploit large amounts of data in huge interconnected neural networks in various layers that can run in parallel to perform calculations.  
    
      The most popular Deep Learning algorithms are:
        * **Convolutional Neural Networks**  
        Convolutional networks make a deep learning neural network capable of recognizing animals, humans, and objects within images.  
        * **Long Short Term Memory Neural Networks**  
        Unlike standard direct feed neural networks, LSTM has feedback connections. It can not only process individual data points (like images), but also complete data streams (like voice or video).  
          For example, LSTM is applicable to tasks such as handwriting recognition, voice recognition, and detection of anomalies in network traffic or IDS (intrusion detection systems).  
          LSTM networks are suitable for classifying, processing, and making predictions based on time series data, as there may be delays of unknown duration between major events in a time series.  
    
    * **Natural Language Processing**  
    It is a mix between Data Science, Machine Learning and Linguistics. It aims to understand human language. Both in texts and in speech / voice. From analyzing syntactically or grammatically thousands of contents, automatically classifying topics, chatbots and even generating poetry imitating Shakespeare. It is also common to use it for Sentiment Analysis on social networks, (for example regarding a politician) and machine translation between languages. Assistants like Siri, Cortana and the possibility to ask and get answers, or even get movie tickets.  
    
<br>
    
> * Recuero de los Santos, P. (2017). Tipos de aprendizaje en Machine Learning: supervisado y no supervisado. Think Big. Retrieved March 25, 2020, from https://empresas.blogthinkbig.com/que-algoritmo-elegir-en-ml-aprendizaje/  
> * Los 10 Algoritmos esenciales en Machine Learning. (2017). Raona. Retrieved March 25, 2020, from https://www.raona.com/los-10-algoritmos-esenciales-machine-learning/  
> * Algoritmos de Machine Learning y cómo seleccionar el mejor(1/3). (2018). LIS Solutions, Consultoría Logística. Retrieved March 25, 2020, from https://www.lis-solutions.es/blog/algoritmos-de-machine-learning-y-como-seleccionar-el-mejor1-3/  
> * Algoritmo PCA: de lo complejo a lo sencillo. (2018). LIS Solutions, Consultoría Logística. Retrieved March 25, 2020, from  https://www.lis-solutions.es/blog/algoritmo-pca-de-lo-complejo-a-lo-sencillo/  
> * ¿Cuáles son los tipos de algoritmos del machine learning?. (2019). APD España. Retrieved March 25, 2020, from https://www.apd.es/algoritmos-del-machine-learning/  
> * Principales Algoritmos usados en Machine Learning. (2017). Aprende Machine Learning. Retrieved March 25, 2020, from  https://www.aprendemachinelearning.com/principales-algoritmos-usados-en-machine-learning/  
> * Machine Learning | Qué es, tipos, ejemplos y cómo implementarlo. (2019). GraphEverywhere. Retrieved March 25, 2020, from https://www.grapheverywhere.com/machine-learning-que-es-tipos-ejemplos-y-como-implementarlo/  
> * Ramírez, C. A. (2018). Algoritmo SVD aplicado a los sistemas de recomendación en el comercio. TIA, 6(1), pp. 21. Retrieved from https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=7&cad=rja&uact=8&ved=2ahUKEwjsrODbq7XoAhWDt54KHdx4BLsQFjAGegQICRAB&url=https%3A%2F%2Frevistas.udistrital.edu.co%2Findex.php%2Ftia%2Farticle%2Fdownload%2F11827%2Fpdf%2F&usg=AOvVaw19h5qNCfQC8GEJqv_axODl  
> * Long short-term memory. (2020). En.wikipedia.org. Retrieved March 25, 2020, from https://en.wikipedia.org/wiki/Long_short-term_memory  

<br>

---

<br>

# Homework 2   
### VectorAssembler Library  
`VectorAssembler` is a transformer that combines a given list of columns into a single vector column. It is useful for combining raw features and features generated by different feature transformers into a single feature vector, in order to train ML models like logistic regression and decision trees. `VectorAssembler` accepts the following input column types: all numeric types, boolean type, and vector type. In each row, the values of the input columns will be concatenated into a vector in the specified order.  

#### Examples  
Assume that we have a DataFrame with the columns `id`, `hour`, `mobile`, `userFeatures`, and `clicked`:  
~~~~  
 id | hour | mobile | userFeatures     | clicked
----|------|--------|------------------|---------
 0  | 18   | 1.0    | [0.0, 10.0, 0.5] | 1.0
~~~~  

`userFeatures` is a vector column that contains three user features. We want to combine `hour`, `mobile`, and `userFeatures` into a single feature vector called `features` and use it to predict `clicked` or not. If we set `VectorAssembler`’s input columns to `hour`, `mobile`, and `userFeatures` and output column to `features`, after transformation we should get the following DataFrame:  
~~~~  
 id | hour | mobile | userFeatures     | clicked | features
----|------|--------|------------------|---------|-----------------------------
 0  | 18   | 1.0    | [0.0, 10.0, 0.5] | 1.0     | [18.0, 1.0, 0.0, 10.0, 0.5]
~~~~  

<br>

### Vectors Library  
Factory methods for `org.apache.spark.ml.linalg.Vector`. We don't use the name `Vector` because Scala imports `scala.collection.immutable.Vector` by default.  

<br>

### How does rootMeanSquareError work?  
`rootMeanSquareError` returns the root mean squared error, which is defined as the square root of the mean squared error.  

The Mean Squared Error (MSE) is a measure of how close a fitted line is to data points. For every data point, you take the distance vertically from the point to the corresponding y value on the curve fit (the error), and square the value. Then you add up all those values for all data points, and, in the case of a fit with two parameters such as a linear fit, divide by the number of points minus two.[1] The squaring is done so negative values do not cancel positive values. The smaller the Mean Squared Error, the closer the fit is to the data. The MSE has the units squared of whatever is plotted on the vertical axis.  

Another quantity that we calculate is the Root Mean Squared Error (RMSE). It is just the square root of the mean square error. That is probably the most easily interpreted statistic, since it has the same units as the quantity plotted on the vertical axis.  

Key point: The RMSE is thus the distance, on average, of a data point from the fitted line, measured along a vertical line.  

[1]: Using the number of points – 2 rather than just the number of points is required to account for the fact that the mean is determined from the data rather than an outside reference. This is a subtlety, but for many experiments, n is large so that the difference is negligible.  

<br>

> * Extracting, transforming and selecting features - Spark 2.4.5 Documentation. (2020). Spark.apache.org. Retrieved April 28, 2020, from https://spark.apache.org/docs/latest/ml-features.html#vectorassembler  
> * Spark 2.4.5 ScalaDoc. (2020). Spark.apache.org. Retrieved April 28, 2020, from https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.linalg.Vectors$  
> * Spark 2.4.5 ScalaDoc. (2020). Spark.apache.org. Retrieved April 28, 2020, from https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.regression.LinearRegressionTrainingSummary  
> * What are Mean Squared Error and Root Mean Squared Error? - Technical Information Library. (2018). Technical Information Library. Retrieved April 28, 2020, from https://www.vernier.com/til/1014  

<br>

---

<br>

# Homework 3  
### Pipeline and Confusion Matrix  

#### PIPELINE LIBRARY  
**Main concepts in Pipelines**  
MLlib standardizes APIs for machine learning algorithms to make it easier to combine multiple algorithms into a single pipeline, or workflow. This section covers the key concepts introduced by the Pipelines API, where the pipeline concept is mostly inspired by the scikit-learn project.  
* **DataFrame:** This ML API uses DataFrame from Spark SQL as an ML dataset, which can hold a variety of data types. E.g., a DataFrame could have different columns storing text, feature vectors, true labels, and predictions.  
* **Transformer:** A Transformer is an algorithm which can transform (`transform()` method) one DataFrame into another, generally adding one or more columns. E.g., an ML model is a Transformer which transforms a DataFrame with features into a DataFrame with predictions.  
* **Estimator:** An Estimator is an algorithm which can be fit (`fit()` method) on a DataFrame to produce a Transformer. E.g., a learning algorithm is an Estimator which trains on a DataFrame and produces a model.  
* **Pipeline:** A Pipeline chains multiple Transformers and Estimators together to specify an ML workflow.  
* **Parameter:** All Transformers and Estimators now share a common API for specifying parameters.  

**Pipeline**  
In machine learning, it is common to run a sequence of algorithms to process and learn from data. E.g., a simple text document processing workflow might include several stages:  
* Split each document’s text into words.  
* Convert each document’s words into a numerical feature vector.  
* Learn a prediction model using the feature vectors and labels.  
MLlib represents such a workflow as a Pipeline, which consists of a sequence of PipelineStages (Transformers and Estimators) to be run in a specific order.  

**How it works**  
A Pipeline is specified as a sequence of stages, and each stage is either a Transformer or an Estimator. These stages are run in order, and the input DataFrame is transformed as it passes through each stage. For Transformer stages, the `transform()` method is called on the DataFrame. For Estimator stages, the `fit()` method is called to produce a Transformer (which becomes part of the PipelineModel, or fitted Pipeline), and that Transformer’s `transform()` method is called on the DataFrame.  

We illustrate this for the simple text document workflow. The figure below is for the training time usage of a Pipeline.  
<img src="https://spark.apache.org/docs/latest/img/ml-Pipeline.png" alt="image" width="55%">  

Above, the top row represents a Pipeline with three stages. The first two (Tokenizer and HashingTF) are Transformers (blue), and the third (LogisticRegression) is an Estimator (red). The bottom row represents data flowing through the pipeline, where cylinders indicate DataFrames. The `Pipeline.fit()` method is called on the original DataFrame, which has raw text documents and labels. The `Tokenizer.transform()` method splits the raw text documents into words, adding a new column with words to the DataFrame. The `HashingTF.transform()` method converts the words column into feature vectors, adding a new column with those vectors to the DataFrame. Now, since LogisticRegression is an Estimator, the Pipeline first calls `LogisticRegression.fit()` to produce a LogisticRegressionModel. If the Pipeline had more Estimators, it would call the LogisticRegressionModel’s `transform()` method on the DataFrame before passing the DataFrame to the next stage.  

A Pipeline is an Estimator. Thus, after a Pipeline’s `fit()` method runs, it produces a PipelineModel, which is a Transformer. This PipelineModel is used at test time; the figure below illustrates this usage.  
<img src="https://spark.apache.org/docs/latest/img/ml-PipelineModel.png" alt="image" width="55%">  

In the figure above, the PipelineModel has the same number of stages as the original Pipeline, but all Estimators in the original Pipeline have become Transformers. When the PipelineModel’s `transform()` method is called on a test dataset, the data are passed through the fitted pipeline in order. Each stage’s `transform()` method updates the dataset and passes it to the next stage.  

Pipelines and PipelineModels help to ensure that training and test data go through identical feature processing steps.  

<br>

#### CONFUSION MATRIX  
In the field of artificial intelligence, a confusion matrix is a tool that allows the visualization of the performance of an algorithm that is used in supervised learning. Each column in the array represents the number of predictions for each class, while each row represents the instances in the actual class. One of the benefits of confusion matrices is that they make it easy to see if the system is confusing the different classes or results of the classification.  

If the number of samples from different classes changes greatly in the input data, the classifier error rate will not be representative. If for example there are 990 samples with result 1, but only 10 with result 2, the classifier will have a bias to classify towards class 1. If the classifier classifies all the samples as class 1, its precision will be 99%. This does not mean that it is a good classifier, as it had a 100% error in classifying class 2 samples.  

Let's see it represented in the following table:  
<img src="https://miro.medium.com/max/712/1*Z54JgbS4DUwWSknhDCvNTQ.png" alt="image" width="35%">  

* **TP** is the number of positives that were correctly classified as positive by the model.  
* **TN** is the number of negatives that were correctly classified as negative by the model.  
* **FN** is the number of positives that were incorrectly classified as negative. Type 2 error (False Negatives)  
* **FP** is the number of negatives that were incorrectly classified as positive. Type 1 error (False positives)  

<br>

**Brief explanation of the Metrics**  
1. **Accuracy and Precision**  
  1.1 **Accuracy**  
    It refers to how close the result of a true value measurement is. In statistical terms, accuracy is related to the bias of an estimate. Also known as True Positive (or True positive rate). It is represented by the proportion between the real positives predicted by the algorithm and all the positive cases.  
    Practically, Accuracy is the number of positive predictions that were correct.  
    * It is calculated as: (TP + TN) / (TP + FP + FN + TN)  

    1.2 **Precision**  
      It refers to the dispersion of the set of values obtained from repeated measurements of a magnitude. The smaller the dispersion, the greater the precision. It is represented by the ratio between the number of correct predictions (both positive and negative) and the total number of predictions.  
      In practical form it is the percentage of positive cases detected.  
      * It is calculated as: TP / (TP + FP)  

2. **Sensitivity (Recall) and Specificity**  
Sensitivity and specificity are two values that indicate the capacity of our estimator to discriminate positive cases from negative ones. Sensitivity is the fraction of true positives, while specificity is the fraction of true negatives.  

    2.1 **Sensitivity (Recall)**  
      Also known as True Positive Rate or TP. It is the proportion of positive cases that were correctly identified by the algorithm.  
      * It is calculated: TP / (TP + FN), or what would be the same in terms of health: True positives / Total Sick (it is the ability to correctly detect the disease among the sick).  

    2.2 **Specificity**  
      Also known as the True Negative Rate or TN. These are the negative cases that the algorithm has correctly classified. It expresses how well the model can detect this class.  
      * It is calculated: TN / (TN + FP), or what would be the same in terms of health: True Negatives / Total Healthy (it is the ability to identify the cases of healthy patients among all healthy).  

3. **F-Score (F-measure)**  
It is difficult to compare two models with low precision and high recall or vice versa. So to make them comparable, we use F-Score. F-score helps to measure Recall and Precision at the same time. It uses Harmonic Mean in place of Arithmetic Mean by punishing the extreme values more.  
    * It is calculated: (2 * (Recall * Precision)) / (Recall + Precision)

&nbsp;

#### Check results of confusion matrix (Example: logregression.scala)  
<img src="https://github.com/Angi-Reynoso/Datos_Masivos/blob/Unidad_2/Images/Confusion Matrix.png" alt="image" width="95%">  

**Confusion Matrix:**  
~~~~  
   TP   |  FP  |       | 
--------|------|-------|
  114   |  16  | = 130 |
--------|------|-------|
   27   |  61  | = 88  |
--------|------|-------|
   FN   |  TN  |       |
--------|------|-------|
 = 141  | = 77 |       |
~~~~  

**1. Accuracy:**  
~~~  
(TP + TN) / (TP + FP + FN + TN) = (114 + 61) / (114 + 16 + 27 + 61) = 175/218 = 0.8027
~~~  
The obtained value coincides with the one thrown in the terminal when using the metrics.accuracy function, which indicates that the matrix has an approximate 80% accuracy; not bad, but could be better.  

**2. Precision = % of Correct Predictions:**  
~~~  
TP / (TP + FP) = 114 / (114 + 16) = 114/130 = 0.8769
TN / (TN + FN) = 61 / (61 + 27) = 61/88 = 0.6931
~~~  
The precision result confirms how effective the predictions made in the model are; In this case, for the classification of true, there is an approximate of 87%, while for the false it is an approximate of 69%. This indicates that the results for the "false" case may not be very reliable or accurate.  

**3. Sensitivity (Recall) = True Positive Rate:**  
~~~  
TP / (TP + FN) = 114 / (114 + 27) = 114/141 = 0.8085
~~~  
According to the obtained result, the percentage of the true positive rate is approximately 80%; This means that most of the positive cases were classified correctly.  

**4. Specificity = True Negative Rate:**  
~~~  
TN / (TN + FP) = 61 / (61 + 16) = 61/77 = 0.7922
~~~  
Similar to the previous calculation, with the true negative rate, the percentage of negative cases that were classified correctly is obtained; in this case it is approximately 79%, a little lower than in the case of positives.  

**Conclusion:**  
It can be said that the model works correctly, however, considering the percentages obtained in the previous calculations (from 69% to 87%), it can also be inferred that some improvements still need to be made; for example in the case of negatives, since this is where the lowest results were obtained.  

<br>

> * ML Pipelines - Spark 2.4.5 Documentation. (2020). Spark.apache.org. Retrieved May 1, 2020, from https://spark.apache.org/docs/latest/ml-pipeline.html  
> * Barrios Arce, J. (2019). La matriz de confusión y sus métricas – Inteligencia Artificial –. Juan Barrios. Retrieved May 1, 2020, from https://www.juanbarrios.com/matriz-de-confusion-y-sus-metricas/  
> * Narkhede, S. (2018). Understanding Confusion Matrix. Medium. Retrieved May 1, 2020, from https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62  

<br>

---

<br>
