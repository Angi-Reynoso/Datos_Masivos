# Index #  
* [Homework 1: Main Types of Machine Learning Algorithms](#homework-1)  
* [Homework 2: ](#homework-2)  

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
      <img src="https://student.mohammed.mx/tutoriales/analisis/541.gif" 
alt="image" width="35%">
        
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
      <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRbYRG_Q6Sn7-TeaXysKVJe2pAObFph2ZD5SGX3BAYxQ0bkiCnf&usqp=CAU" alt="image" width="35%">  

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
### 
