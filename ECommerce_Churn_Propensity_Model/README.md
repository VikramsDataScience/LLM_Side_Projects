# ECommerce Customer Churn Propensity Model (POC Complete)
The dataset I'm working with was sourced from Kaggle. It's a customer dataset from an <a href='https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction/data'>Ecommerce</a> company that contains anonymised customer ID numbers, and their activity prior to churning. Since, it's a customer churn dataset, it also contains the boolean flag for those customers who did or didn't churn.<br>
&nbsp;&nbsp;With this model I'm hoping to develop a churn propensity model that will predict the likelihood of a customer churning based on the activity that's measured in the provided dataset. Obviously the reasons why a customer may decide to cancel their membership with a company is highly prone to confounding factors that may not be measurable. As such, one of the many limitations of such modelling does imply that confounding factors (or variables if they're measurable) can significantly affect the model's predictive accuracy. 


## Mathematical background & justifications for model selection and hyperparameter tuning
- <b>On Stratification:</b> The class label/target variable 'Churn' is an imbalanced data set that favours those who didn't churn (~83% did not churn) - which stands to reason, since if we had a balanced number of customers that did consistently churn we may not be working for a viable business!<br>
&nbsp;&nbsp;This imbalance in the class label/target variable presents a problem for any Machine Learning model, since any predictions will favour those who did not churn. The answer is to perform a Stratification on the class label such that the train and test sets have a roughly similar proportion of positive and negative samples in both sets. On this note, I've opted for the simpler version of stratification that comes with the `train_test_split()` function in the `sklearn` class, and not the more advanced `StratifiedKFold` or `StratifiedShuffleSplit` capabilities that are also part of sklearn. This was intended as the conditions for the initial experiment for these 3 models. However, the F1 scores, especially for the `RandomForestClassifier()` and `XGBClassifier()` were so high in the out-of-bag test predictions (~0.96 for each model) that I didn't feel the need to further experiment with the more advanced stratification methods.<br>
- <b>On Regularization:</b> Prior to discussing the algorithms chosen, let's first discuss Regularization. Specifically L2 Regularization. Mathematically, this is defined as follows:<br>
<!-- Centered equation -->
$$
L_2=||w||_2^2=w_1^2+w_2^2+...+w_n^2
$$
<!-- End centered equation -->
<br>Where:<br>
- $w$ represents the weights of all the features in the model
<br><br>In the above, the equation takes the absolute value of the sum of squares of all the feature weights in the model to arrive at a penalty. In other words, the higher the summed  weight value the more the model is said to be overfitted against the training set (and therefore won't generalise well in the out-of-sample test sets). The regularization calculation will penalise that complexity to prevent overfitting.
<br>&nbsp;&nbsp;It should also be noted that L2 regularization is sensitive to outlier weights. For example in the following set of weights:
<!-- Centered equation -->
$$\{w_1=1, w_2=0.5, w_3=4, w_4=0.25\}$$
<!-- centered equation -->
<br>The L2 calculation would be:<br>
<!-- Centered equation -->
$$L_2=1^2+0.5^2+4^2+0.25^2 = 17.31$$
<!-- centered equation -->
<br>But, we can see that 16 out of 17.31 comes from $w_3$ and 1.31 comes from the remaining 3 weights. It's a just a point to note when working with L2 regularization in the Machine Learning world, as this is quite key to accounting for (and penalising) large weights in training sets 😁.

### Logistic Regression
For Logistic Regression, stratification at the train/test split stage is not sufficient (this was experimentally validated when using the 'class_weight'=None to return a much lower accuracy). So the `class_weight`='balanced' was used to further stratify the class labels. Please note that this stratification is performed at the insample training stage (when the `fit()` method is called). From sklearn's documentation, the equation used to calculate this is as follows:<br>
<!-- centered equation -->
$$\frac{n_{\text{samples}}}{n_{\text{classes}} \times \text{np.bincount}(y)}$$
<!-- centered equation -->
<br>The Optimization algorithm of choice was the Coordinate Descent (solver='liblinear' argument in the `LogisticRegression()` class). We've already discussed L2 Regularization. Following on from that, the following Optimisation equation allows us to minimise Loss as an Objective function:<br>
<!-- centered equation -->
$$\min_{w} \left[ \frac{1}{n} \sum_{i=1}^n \log(1 + \exp(-y_i(X_i \cdot w + b))) + \frac{\lambda}{2} \|w\|^2 \right]$$
<!-- centered equation -->
<br>Where:<br>
- $X_i$ are the feature vectors
- $y_i$ are the labels
- $w$ is the weight vector
- $b$ is the bias term
- $n$ are the number of samples
- ${\lambda}$ is the regularization parameter

The above equation looks a little intimidating, but it's essentially stating that the objective function is the minimisation of the weight vector $w$ as $n$ samples increase. Earlier, we discussed the calculation of L2 regularisation, whereby the weights are sensitive to outliers. That sensitivity to account for outlier weights is most effective in the above equation's ${\lambda}$ parameter. That sensitivity is most penalized here as $n$ samples increase. 
<br>&nbsp;&nbsp;For the purpose of the model that's been developed here, the Coordinate Descent Optimisation solver has outperformed the other choices for Logistic Regression by way of the F1-score. In saying that, this data has also proved to be quite challenging for Logistic Regression to handle as the F1 scores appeared to find a 'steady state' of sorts at around ~0.78 during the out-of-sample test samples (i.e. when calling the `predict()` method), regardless of the choice of Solver or additional Hyperparameter tuning.

###  Mathematical explanations of the Random Forest Classifier
I understand that both RF and XGBoost Classifiers are quite famously used owing to their general ease of use and forgiveness towards missing data. But, I thought it would also be a good idea to do a bit of a deep dive into the mathematical reasoning behind how they operate under the hood.
<br>&nbsp;&nbsp;At its heart the Random Forest classifier is an ensemble (a forest of trees) that grow recursively until a class label prediction is reached. At the conclusion of the following conditions, a vote is taken that bases its prediction on the mode value of which class label is the winner:
- <b>Bootstrapping</b>: In leiu of using validation sets, the RF Classifier uses a bootstrapping technique that samples $B$ bootstraps from the total training set $N$. This process is randomized to prevent overfitting, and is done with replacement from the original data.
- <b>Tree growth</b>: This is based on the bootstrapped sample randomly selecting $m$ features from the total feature space $p$ so long as $m < p$. This bootstrapped sample will need to be split based on either an impurity measure such as the Gini Impurity score or Entropy. In this assessment we can say that the following impurity calculations assess the probability of misclassification at the point of split (i.e. the higher the score the higher the probability of misclassification). 
<br>&nbsp;&nbsp;These are the equations for each splitting method:<br>
<!-- centered equation -->
$$\text{Gini}(t) = \sum_{i=1}^c p_i(1 - p_i)$$
<!-- centered equation -->
<!-- centered equation -->
$$\text{Entropy}(t) = - \sum_{i=1}^c p_i \log(p_i)$$
<!-- centered equation -->
- <b>Feature Importance:</b> This is assessed by examining each feature's contribution to the <i>reduction</i> of impurity at each split, and aggregating the total contribution by each feature across the forest of trees to arrive at a final feature importance score.
- <b>Out-of-Bag (OOB) Error:</b> With bootstrap sampling about 1/3 of the data is left 'out-of-bag' so to speak. These samples can later be used to get an unbiased sample of the model's error without requiring a validation set.
- <b>Voting</b>: The aforementioned point around the voting that determines the final prediction $\hat{y}$ is determined by the following equation. This equation bases its prediction on the mode value of which class label is the winner. This mode value is said to be the majority vote and ultimately becomes the predicted class label (i.e. 0 or 1 for binary classification cases):<br>
<!-- centered equation -->
$$\hat{y}=\text{mode}{{\{h_b(x):b=1,2,...,B}\}}$$
<!-- centered equation -->

###  Mathematical explanations of the XGBoost Classifier
<a href='https://arxiv.org/pdf/1603.02754'>XGBoost (eXtreme Gradient Boosting)</a> doesn't have too many similarities to Random Forest. Instead of growing a forest of trees, XGBoost is a greedy algorithm that starts from a single leaf and iteratively adds branches to a single tree that has a structure defined by $q$. 
<br>&nbsp;&nbsp;These are definitely different algorithms, albeit both are equally brilliant and useful across a wide variety of use cases. XGBoost's computational approach and splitting method to how it grows a decision tree and arrives at predictions is quite different. Here are some of the differences:
- <b>Additive Node Splitting:</b> RF Classifer uses ensembling techniques and votes on the eventual class label (prediction). However, XGBoost uses a 'Greedy Algorithm' to arrive at class label predictions. The implication here is that it starts with a single leaf and iteratively adds branches to the tree. This is done in a synchronous manner (the previous computation must be completed before the next computation can commence).
- <b>Quality scoring for tree structure:</b> Whilst RF Classifier uses the Gini Impurity or Log Loss to calculate Entropy, XGBoost uses a scoring function that is similar to calculating impurity but instead calculates scores across a wider range of objective functions to assess the quality of tree structure. This is the equation:<br>
<!-- centered equation -->
$$\tilde{\mathcal{L}}^{(t)}(q) = -\frac{1}{2} \sum_{j=1}^T \frac{(\sum_{i \in I_j} g_i)^2}{\sum_{i \in I_j} h_i + \lambda} + \gamma T$$
<!-- centered equation -->
Where:<br>
- $\tilde{\mathcal{L}}^{(t)}(q)$ is the objective function at iteration $t$ with respect to the tree structure denoted by $q$
- $T$ are the total number of leaves in the tree
- $I_j$ is the instance set of leaf $j$
- $g_i$ is the first-order gradient statistic (first order derivative) of the loss function with respect to the prediction for data point $i$
- $h_i$ is the second-order gradient statistic (second order derivative) of the loss function with respect to the prediction for data point $i$. One point to note about $h_i$ is that this is an <i>approximate</i> Hessian. It's approximate because the true mathematical definition of a Hessian matrix is a <i>partial</i> 2nd order derivative. Whereas in gradient boosting this is the second derivative of the loss function at a point in time.
- $\lambda$ is one of the regularisation parameters that was discussed earlier
- $\gamma$ is also a regularisation parameter, but is used to control the number of leaves $T$<br>
<br>There's a bit more to tree splitting that's defined in the following equation:<br>
<!-- centered equation -->
$$L_{\text{split}} = \frac{1}{2} \left[\frac{(\sum_{i\in I_L} g_i)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{(\sum_{i \in I_R} g_i)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{(\sum_{i \in I} g_i)^2}{\sum_{i \in I} h_i + \lambda} \right] - \gamma$$
<!-- centered equation -->
<br>&nbsp;&nbsp;In XGBoost the 'greedy algorithm' starts with a single leaf and iteratively adds branches to the tree. In practice the equation above is used to evaluate split candidates. Where:
- $I_L$ are the instances of the left node after the split
- $I_R$ are the instances of the right node after the split
- $I$ is the union of the left and right nodes after the split (i.e. $I = I_L \cup I_R$)
- The remaining terms in the equation have all been defined previously and remain the same in the above equation as well

### Random Forest & XGBoost Classifier performance on the Ecommerce data set
Unlike the Logistic Regression model both decision tree models performed extremely well with respect to F1-scores (~0.95-0.96).
<br>&nbsp;&nbsp;The RF/XGBoost Classifiers use the same stratification method that was discussed in the Logistic Regression section. However, unlike Logistic Regression performing an additional stratification beyond the stratification performed at the `train_test_split()` stage, proved to have a negative effect on the F1-scores for the RF Classifier (and indeed the XGBoost Classifier, as well). This implies that, according to the sklearn documentation, each weight in the class labels (i.e. 'class_weight=None' argument setting) are treated with an equal weight of 1 across all class labels. 
<br>&nbsp;&nbsp;My theory is that, unlike Logistic Regression, it appears that for the RF Classifier, performing stratification at the in-sample training stage (i.e. train_test_split() stage) <i>and</i> attempting to use 'balanced' weights assigned to the class labels at the Out-of-Bag (OOB) stage appears to skew the randomness that's required when RF performs its training on the bootstrapped OOB sample. I'm not sure if this is correct, but that's my theory as it applies to this dataset. For other datasets this may be quite a different story, since decision tree algorithms are not simple and straightforward, but complex and possess stochasticity!

### Flask app and the simplified model based on Feature Importance
A simple Flask app has also been created, and can be very easily launched on a local host by running the app.py module - i.e. `python app.py`. This command will launch the app on your local machine in a debug state (unless otherwise specified by changing the `app.run(debug=False)`).
<br>&nbsp;&nbsp;The app itself is rendered as a form submission of which all the JavaScript, HTML and CSS have been written. The form submission is meant for stakeholders to input values as radio buttons across the simplified feature set. Upon submission this will generate a propensity score of churn.
<br>&nbsp;&nbsp;The reason I used a simplified feature set was only to simplify the form submission. Since there were 19 features in total (but many more since OneHot Encoding was used to dummify the categorical feature set), I felt that for stakeholder simplicity I would create a less accurate model based only on the features that possessed the highest feature importances. Whilst, this would simplify the form fill, it would create a less accurate propensity score. In saying this, I would happily use the full feature set, if asked. But, this would certainly make for quite a long form fill, and would probably test the patience of most stakeholders!