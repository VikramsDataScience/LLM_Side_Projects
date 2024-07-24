# ECommerce Customer Churn Propensity Model (POC Complete)
The dataset I'm working with was sourced from Kaggle. It's a customer dataset from an <a href='https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction/data'>Ecommerce</a> company that contains anonymised customer ID numbers, and their activity prior to churning. Since, it's a customer churn dataset, it also contains the boolean flag for those customers who did or didn't churn.<br>
&nbsp;&nbsp;With this model I'm hoping to develop a churn propensity model that will predict the likelihood of a customer churning based on the activity that's measured in the provided dataset. Obviously the reasons why a customer may decide to cancel their membership with a company is highly prone to confounding factors that may not be measurable. As such, one of the many limitations of such modelling does imply that confounding factors (or variables if they're measurable) can significantly affect the model's predictive accuracy. 


## Mathematical background & justifications for model selection and hyperparameter tuning
- <b>On Stratification:</b> The class label/target variable 'Churn' is an imbalanced data set that favours those who didn't churn (~83% did not churn) - which stands to reason, since if we had a balanced number of customers that did consistently churn we may not be working for a viable business!<br>
&nbsp;&nbsp;This imbalance in the class label/target variable presents a problem for any Machine Learning model, since any predictions will favour those who did not churn. The answer is to perform a Stratification on the class label such that the train and test sets have a roughly similar proportion of positive and negative samples in both sets. On this note, I've opted for the simpler version of stratification that comes with the `train_test_split()` function in the `sklearn` class, and not the more advanced `StratifiedKFold` or `StratifiedShuffleSplit` capabilities that are also part of sklearn. This was intended as the conditions for the initial experiment for these 3 models. However, the F1 scores, especially for the `RandomForestClassifier()` and `XGBClassifier()` were so high in the out-of-sample test predictions (~0.96 for each model) that I didn't feel the need to further experiment with the more advanced stratification methods.<br>
- <b>On Regularization:</b> Prior to discussing the algorithms chosen, let's first discuss Regularization. Specifically L2 Regularization. Mathematically, this is defined as follows:<br>
$$L_2=||w||_2^2=w_1^2+w_2^2+...+w_n^2$$
Where:<br>
- $w$ represents the weights of all the features in the model
<br><br>In the above, the equation takes the absolute value of the sum of squares of all the feature weights in the model to arrive at a penalty. In other words, the higher the summed  weight value the more the model is said to be overfitted against the training set (and therefore won't generalise well in the out-of-sample test sets). The regularization calculation will penalise that complexity to prevent overfitting.
<br>&nbsp;&nbsp;It should also be noted that L2 regularization is sensitive to outlier weights. For example in the following set of weights:
$$\{w_1=1, w_2=0.5, w_3=4, w_4=0.25\}$$
The L2 calculation would be:
$$L_2=1^2+0.5^2+4^2+0.25^2 = 17.31$$
But, we can see that 16 out of 17.31 comes from $w_3$ and 1.31 comes from the remaining 3 weights. It's a just a point to note when working with L2 regularization in the Machine Learning world, as this is quite key to accounting for (and penalising) large weights in training sets 😁.

### Logistic Regression
For Logistic Regression, stratification at the train/test split stage is not sufficient (this was experimentally validated when using the 'class_weight'=None to return a much lower accuracy). So the `class_weight`='balanced' was used to further stratify the class labels. Please note that this stratification is performed at the insample training stage (when the `fit()` method is called). From sklearn's documentation, the equation used to calculate this is as follows:<br>
$$\frac{n_{\text{samples}}}{n_{\text{classes}} \times \text{np.bincount}(y)}$$
<br>The Optimization algorithm of choice was the Coordinate Descent (solver='liblinear' argument in the `LogisticRegression()` class). We've already discussed L2 Regularization. Following on from that, the following Optimisation equation allows us to minimise Loss as an Objective function:<br>
$$\min_{w} \left[ \frac{1}{n} \sum_{i=1}^n \log(1 + \exp(-y_i(X_i \cdot w + b))) + \frac{\lambda}{2} \|w\|^2 \right]$$
Where:<br>
- $X_i$ are the feature vectors
- $y_i$ are the labels
- $w$ is the weight vector
- $b$ is the bias term
- $n$ are the number of samples
- ${\lambda}$ is the regularization parameter