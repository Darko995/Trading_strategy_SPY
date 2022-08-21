# Trading_strategy_SPY
The trading strategy was developed in a python programming language.

# Data collection

Data of the Standard & Poor's 500 ETF (Ticker Symbol: SPY) from Jan 1st 2000 to Dec 31st 2010 was downloaded from Yahoo Finance site. Data consists of 2676 rows and 6 columns (High, Low, Open, Close, Volume, Adj Close).
Before constructing the model we will look at what the data look like:

![image](https://user-images.githubusercontent.com/53577768/185779923-bb8a9674-c8b1-4577-bd44-125e1b4cf19a.png)

On the y axis is stock price at the end of the day. We can notice that the value of these stocks has fallen in this time period, so if we invested money in the beginning we would eventually lose a certain amount of money.

The idea is to form and train a classification model on the basis of historical data, on the basis of which we will predict the movement of stocks in the future with as much accuracy as possible and on the basis of these predictions buy or sell stocks.

# Random Forest Classifier

First, a classification model was formed. The model used for this classification task is Random Forest Classifier. This is Machine Learning, supervised, ensemble method. Ensembles are a divide-and-conquer approach used to improve performance. The main principle behind ensemble methods is that a group of “weak learners” can come together to form a “strong learner”. Each classifier, individually, is a “weak learner,” while all the classifiers taken together are a “strong learner”. Random forests provide an improvement over bagging by doing a small tweak that utilizes de-correlated trees. In bagging, we build a number of decision trees on bootstrapped samples from training data, but the one big drawback with the bagging technique is that it selects all the variables. By doing so, in each decision tree, order of candidate/variable chosen to split remains more or less the same for all the individual trees, which look correlated with each other. Variance reduction on correlated individual entities does not work effectively while aggregating them.

# Feature Engineering

One of the most important steps in model development is feature engineering. The goal is to form features that will contribute to the correct prediction of the target variable. Eight features has been created for this problem: ho (distance between highest and opening daily price- numerical feature), lo (distance between lowest and opening daily price-numerical feature), feat1 (take two possible values -1 and -1, depending on whether today's value at the end of the day is greater than the value of shares at the beginning and the previous day, if yes than this feature take 1, otherwise -1-ordinal feature), feat2 and feat3 by the same logic, only the values at the end of today and two or three days before are compared, momentum (represents a measure of the short-term tendency of stock price growth. the number was obtained as the mean of the winnings of the previous five days), volatility (represents a measure of stock price oscillations in the short past. Each number was obtained by the standard deviation of prices of the previous five days), ma (Another ordinal feature. The idea is based on moving average trading strategies, gains a value of 1 when the average value of the stock price of the previous 5 days becomes higher than the average value of the stocks of the previous 10 days). Given that this is a supervised model of machine learning, a known target variable is needed. Here it is a variable outcome (will get a value of 1 when the difference between the closing price and the opening price next day is greater than zero, or when it would be worthwhile to invest in these stocks, otherwise get a value of -1).

# Model Development

As already mentioned, Random Forest Classifier will be used for this classification task. First, all data were divided into a training set and a test set, so that they could assess the model’s ability to generalize. Data was divided in proportion 70%-30% using python sklearn library.
The way this model works is shown in the figure below:

![image](https://user-images.githubusercontent.com/53577768/185780028-456f489e-27fc-48ad-bf4b-726f1e9fe691.png)

Two main principles were used in this model: bootstrapping and aggregation. The bootstrap method is a resampling technique used to estimate statistics on a population by sampling a dataset with replacement. The bootstrap method can be used to estimate a quantity of a population. This is done by repeatedly taking small samples, calculating the statistic, and taking the average of the calculated statistics. The Bootstrap Aggregation algorithm for creating multiple different models from a single training dataset. In this concrete model, every model is Decision Tree. Bootstrapping and aggregation together make is a simple and very powerful ensemble method. Bootstrapping and aggregation together make Bagging method. Bootstrap Aggregation is a general procedure that can be used to reduce the variance for those algorithm that have high variance. An algorithm that has high variance are decision trees.
Before we start training our model, we need to choose the parameters of this model. We choose 3 parameters: n_estimators(number of decision trees used in model) max_depth(max depth of every tree), max_features (every estimator was trained on different features - this will give us one by-product of this model. we will see which predictors are most often chosen and based on that we can see which ones are most useful to us). These parameters are selected using grid
 
search (sklearn). For each of the parameters several values were selected, then the model was trained on each of the combinations of parameters and the combination that results in the highest accuracy of classification was choosen. In this example the following parameters are selected: n_estimators = 10, max_depth = 6 and max_features = 5.

# Training and evaluation Model

After determining the optimal parameters, we train the model again and check the accuracy of this model with test data.
Accuracy of this model is 85.32 %. Since this classification problem has 2 classes into which it can classify examples if we randomly chose one of the two classes, the probability of the correct classification would be 50%. Model improvement should be sought in higher quality of feature engineering, gathering more data and the potential use of a more complex model.
There are several other indicators that show the quality of the model that has been developed:

Classification_report shows some other important indicators of classification quality such as precision, recall and f1 score. Classification report for this model is shown below:

	          precision	  recall	  f1-score	  support
-1	          0.90	       0.84	      0.87	      488
1	            0.80	       0.87	      0.83	      343
accuracy			                        0.85	      831
macro avg	    0.85	       0.86	      0.85	      831
weighted avg	0.86	       0.85	      0.85	      831

Classification accuracy alone can be misleading if you have an unequal number of observations in each class or if you have more than two classes in your dataset.
Calculating a confusion matrix can give you a better idea of what your classification model is getting right and what types of errors it is making:

                                Confusion Matrix 
                                  [[1049 114]
                                  [ 100 673]]

                                Accuracy 0.8894628099173554
                                
The number of correct and incorrect predictions are summarized with count values and broken down by each class. This is the key to the confusion matrix. It gives us insight not only into the
 
errors being made by your classifier but more importantly the types of errors that are being made. This Confusion matrix was made on training examples. Accuracy on training examples is about 88.9%. Each row of the matrix corresponds to a predicted class. Each column of the matrix corresponds to an actual class.
Next the success indicator of the model is ROC curve. AUC - ROC curve is a performance measurement for the classification problems at various threshold settings. ROC is a probability curve and AUC represents the degree or measure of separability. It tells how much the model is capable of distinguishing between classes. Higher the AUC, the better the model is at predicting 0 classes as 0 and 1 classes as 1. ROC curve of this model was shown below:

![image](https://user-images.githubusercontent.com/53577768/185780197-14d41c3c-59b8-48ed-887a-944851f97958.png)

As mentioned earlier, Random Forest Classifier has one useful by-product, we could determine the most important predictors based on how many times they were selected from the back of each individual estimator. Feature importance is shown below:

![image](https://user-images.githubusercontent.com/53577768/185780212-f645b37b-846f-47c5-bbe5-28e8dda17c50.png)

This can be significant for a large number of features, in addition to reducing computationaly cost the omission of non-informative predictors can increase the accuracy of the model.
Below are all the quality indicators of this model on the test data set:

	                       precision	recall	f1-score	support
-1	                        0.90	   0.84	     0.87	    488
1	                          0.80	   0.87	     0.83	    343
accuracy			                                 0.85	    831
macro avg	                  0.85	   0.86	     0.85	    831
weighted avg	              0.86	   0.85	     0.85	    831

                                Confusion Matrix 
                                    [[412 76]
                                    [ 46 297]]

                                Accuracy 0.8531889290012034




