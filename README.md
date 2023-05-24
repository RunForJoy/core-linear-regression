# Linear regression model

Summary: This project is about supervised learning and particularly about linear model, regularization, overfitting and underfitting and metrics for quality estimation.

## Contents

1. [Chapter I. Preamble](#chapter-i-preamble)
2. [Chapter II. Introduction](#chapter-ii-introduction) \
    2.1. [Regression problem](#regression-problem) \
    2.2. [Linear regression](#linear-regression) \
    2.3. [Gradient Descent](#gradient-descent) \
    2.4. [Overfitting and underfitting](#overfitting-and-underfitting) \
    2.5. [Quality metrics](#quality-metrics) \
    2.6. [Alternative linear regression problem formulation](#alternative-linear-regression-problem-formulation)
3. [Chapter III. Goal](#chapter-iii-goal) 
4. [Chapter IV. Instructions](#chapter-iv-instructions)
5. [Chapter V. Task](#chapter-v-task)


## Chapter I. Preamble

In the last project, we discussed about what ML is and what problems this area of science solves. 
And also a detailed series of examples of the most difficult tasks, and by what group they can be divided. 
The goal of the open project is one of this group and will get acquainted with the first child - a linear model.

But before we start, I would like to briefly describe how they usually approach the description of all tools. 
Further, we will consider the formulation of the problem from a mathematical point of view. 
And the main tool for solving these statements will be the following algorithm:

We limit the pool of possible solutions to a certain set (set of functions).

For example, we want to predict temperature y from pressure x with only 1 observation. 
We restrict ourselves to the set of linear functions:

$`y≃\hat y=f(x)=ax`$

As a rule, each solution in this set can be uniquely determined using parameters..
In our example, this is the parameter *a*

We choose a loss function that clearly demonstrates our desire to find a solution that satisfies the goal of the problem.

In our example, this will be the standard deviation, which takes on high values if our prediction is far from the true value..

$`L(y,\hat y)=(y - \hat y)^2`$

Thus, the search for our optimal solution will be reduced to finding a solution for which the loss function is minimal. 
But since the function will depend on the parameter, we can say that the goal is to find such parameters for which the loss 
function is minimal.

In our example, formally the task will sound like this:

$`arg \min\limits_{\hat y} L(y,\hat y)=arg \min\limits_{a} L(y,ax)`$

The fact that the loss function depends on the parameters will allow us to calculate the derivatives and 
use optimization methods (including gradient descent) to find the optimal values for the parameters

In our example:

$`{\delta L \over {\delta a}} = -2a(y-ax) = 0 \Rightarrow a=\{y/x, 0\}`$

Often it will be possible to look at the same problem mathematically from a different angle by solving 
it in an alternative formulation. This will open up new meanings in the solution, which will help you understand 
the details even better. Therefore, we strongly recommend that you do not dwell on the materials mentioned in this 
course and periodically improve your understanding of one or another approach.

In this chapter we will discuss regression problems limiting the pool of possible solutions with linear models. 
Together with the model itself we consider things that are closely related to the model development process: 
definitions of overfitting/underfitting, how to handle these and how to estimate the quality of models.

## Chapter II. Introduction

### Regression problem

Suppose we have a set of inputs *X*  which are corresponded to outputs *y*. 
The goal is to find a mapping function f from *X* to *y*. 
In regression problems $`y ∈ R`$ and our $`X ∈ R^n`$. In other words:

$`f: X \rightarrow y`$, where $`y \in R`$ is target, *X* is training data.

There are a lot of examples for real-world regression problems. Let’s consider the most popular:
* Predict the age of a viewer watching TV or video channels.
* Predict amplitude, frequency and other characteristics of heartbeat
* Predict the required amount of taxies due to specific weather conditions
* Predict the temperature at any location inside a building using weather data, time, door sensors, etc.

In case of linear regression the key property is that the expected value of the output is assumed to be a linear 
function of the input, $`f(x) = w^Tx`$. This makes the model easy to fit the data and easy to interpret. 

For regression, the most common choice is to use quadratic loss, or $`l2`$ loss, or MSE – mean squared error:

$`l2(y, \hat y) = (y - \hat y)^2`$

where $`\hat y`$ is the estimation of the model. We use loss to minimize errors and to fit model. 
Let’s dive into linear regression model.

### Linear regression

One of the most widely used, but not always effective algorithms for regression’s problem solution is linear 
regression model. Main goal of this model is to estimate the linear function of the training set with the least error.

We can fit this data using a simple linear regression model of the form 

$`f(x; \theta) = b + wx`$

where *w* is the slope, *b* is the offset, and $`\theta = (w,b)`$ are all the parameters of the model. 

By adjusting $`\theta`$, we can minimize the sum of squared errors until we find the least squares solution:

![](misc/images/mse.jpg)

where MSE is mean squared error.

Lets deep with formal way, linear regression model could be wrote as:

![](misc/images/linear_regression_model.jpg)

where $`w^Tx`$ represents the inner or scalar product between the input vector *x* and the model’s weight vector *w*, 
and *ε* is the residual error between our linear predictions and the true response.

From a mathematical point of view, we can define our task as minimizing MSE. 
To solve these problems we can use optimization methods such as gradient descent or Newton's method 
or even write down analytical solution that contain matrix inversion (after you will complete your first task, 
which we will discuss later in the chapter - you will find that matrix inversion is a heavy calculating operation). 
First task for you will be to find an analytical solution for the regression task.

### Gradient Descent

It is a common fact that gradient is the direction of fastest increase of function value and in the opposite case 
anti-gradient will show fastest decrease direction. Thus Gradient descent is about finding a local minimum of a 
differentiable function. And since our main goal in almost any machine learning task is to reduce the value of 
loss function gradient descent is perfectly fit for this needs.

To do it we should iteratively repeat next steps:

1. Calculate the value of gradient

    ![](misc/images/gradient.jpg)

2. Update parameter values using formula:

    ![](misc/images/update_formula.jpg)
    
    Where  is the learning rate that defines how far we should go and where 0 is usually defined randomly.

![](misc/images/gradient_descent.jpg)

There are a lot of modifications of gradient descent for more accurate results and fast calculation. 
We consider primary modification – stochastic gradient descent. Strong side of this method is that we stochastically 
choose one object (or batch of objects) and calculate gradient only on this object. It helps to save a lot of time, 
but qualitative efficiency doesn't have huge dropdowns. 

We recommend you to read about more modifications of GD such as Adagrad, RMSProp, Adam and momentum method, 
also you can read about secondary order methods like Newtons method.

### Overfitting and underfitting

A fundamental problem with MLE is that it will try to pick parameters that minimize loss on the training set, 
but this may not result in a model that has low loss on future data. This is called overfitting.

Thus when we fit highly flexible models, we need to be careful that we do not overfit the data, that is, 
we should avoid trying to model every minor variation in the input, since this is more likely to be noise than true signal. 
This is illustrated in the figure below, where we see that using a high degree polynomial results in a curve that is 
very “wiggly”. It is unlikely that the true function has such extreme oscillations. Thus, using such a model might 
result in accurate predictions of future outputs

![](misc/images/graphics.jpg)

Underfitted model is the opposite to overfitted - the model that does not learn dependencies from training dataset 
or learn it in a wrong way. Which again results in poor performance on the test dataset. For example, we need to 
classify handwritten digits, and it happens that all images of 8’s have 67 dark pixels, while all other examples don’t. 
The model might decide to exploit this accidental regularity, thereby correctly classifying all the training examples 
of 8’s, without learning the true regularities. And when we pass some new images of 8’s that will be written with 
the other scale, we get very poor performance since the number of dark pixels will differ. The model does not 
learn required patterns.

In other words we need to fit the model balancing between these 2 cases. This is also known as bias-variance tradeoff. 
The error that we produce when fitting and evaluating the model can be splitted into 2 parts (actually 3, but we 
can’t influence on the third one): bias and variance. To understand what does every type of error means refer to 
figure from [Wikipedia](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff):

![](misc/images/low_low.jpg)
![](misc/images/high_low.jpg)

![](misc/images/low_high.jpg)
![](misc/images/high_high.jpg)

> By Bernhard Thiery - Own work, CC BY-SA 3.0

* “The bias error is an error from erroneous assumptions in the learning algorithm. High bias can cause an algorithm to miss the relevant relations between features and target outputs (underfitting).”
* “The variance is an error from sensitivity to small fluctuations in the training set. High variance may result from an algorithm modeling the random noise in the training data (overfitting).”

Hopefully, the majority of models in machine learning deal with bias error and could have problems mainly with variance. 
But there are a lot of special techniques for reducing model overfitting. Here we want to introduce one of them 
that is called regularization. 

### Regularization

Usually the regularization comes from the fact that it simply consumes the additive in the loss function, 
which depends on the parameters of the model.

![](misc/images/regularization_first_formula.jpg)

The idea of this addition is that the models do not seek to learn complex patterns and try to simplify the 
solution. In other words, if the model begins to overfit in the process of finding a solution, then the addition in 
the error function will begin to increase and this will force the model to stop training.

For linear regression, 2 variants of these additives are usually used:
* L2 - then the linear model is called the Ridge model 

    ![](misc/images/ridge_model.jpg)

* L1 - then the linear model is called Lasso model

    ![](misc/images/lasso_model.jpg)

* Sklearn has also a separate class when these 2 regularizations are combined and such a model is called Elastic

Think about how the algorithm will changes for finding a solution for a linear model if you add L1 and L2 
regularization to the loss function. Provide an answer inside your project.

### Quality metrics

Lets dive into quality metrics for regression problem estimation. We already have learned mean square error. 
And after research of definition  ℓ1 regularization you already have known what is mean absolute error. 
But these metrics are not relative. What does it mean?

Let's imagine that we are solving a client income task, so our target variable is continuous. 
And our model’s MAE is equal to 5000. How could we estimate, is our model good enough?

Take a break and think for several minutes. Here we will discuss a few ways for this problem solution. 
First and  the simplest one compares our model with native forecasts. For example we could find the mean of our 
target variable and set it as a prediction for a test sample for a whole set of clients. And finally compare MAE 
of our model and native forecast. Here we recommend finding the best native forecast for MSE and MAE. 
Also think how we could improve native prediction for better quality. 

The second way we will discuss here is to reconstruct relative metrics to absolute ones. 
The Mean Absolute Percentage Error (MAPE) is one of the widely used regressive problem metrics. 
MAPE is the sum of the individual absolute errors divided by the demand (each period separately). 
It is the average of the percentage errors. Compared to MAE, MAPE has an exact range. 
For better understanding think out few examples and measure different metrics, also we recommend to define 
MSLE and R squared coefficient.

### Alternative linear regression problem formulation

Lets see on linear regression from the probability point of view and we can rewrite model with connection 
between linear regression and Gaussians in the following form:

![](misc/images/linear_regression_and_gaussian.jpg)

where  $`N(µ, \sigma^2)`$ is **Gaussian** or **normal** distribution with µ is the **mean** and $`\sigma^2`$ is the **variance**.

This makes it clear that the model is a **conditional probability density**. 
In the simplest case, we assume µ is a linear function of *x*, so $`µ= w^Tx`$, and that the noise is fixed, $`\sigma^2(x) = \sigma^2`$.

In this case, $`\theta = (w,\sigma^2)`$  are the **parameters** of the model. 

For example, suppose the input is 1 dimensional. We can represent the expected response as follows:

$`µ(x)= w_0+ w_1x = w^Tx`$

where $`w_0`$ is the intercept or bias term, $`w_1`$ is the slope, and where we have defined the vector *x = (1, x)*.

Let's try to find out how to determine parameters of the  linear regression model.

A common way to estimate the parameters of a statistical model is to compute the MLE - 
Maximum Likelihood Estimation. MLE is a method for estimating an unknown parameter by maximizing a 
Likelihood function, we can define it as: 

$`\hat \theta = arg \max\limits_{\theta} L(\theta) = arg \max\limits_{\theta} \prod_{i=1}^l P(y_i|x_i, \theta)`$

Where $`L(\theta)`$ – Likelihood function.

It is common to assume the training examples are independent and identically distributed. 
This means we can write the log-likelihood as follows:

![](misc/images/log_likelihood.jpg)

Instead of maximizing the log-likelihood, we can equivalently minimize the negative log likelihood or NLL:

![](misc/images/nll.jpg)

The NLL formulation is sometimes more convenient, since many optimization software packages are designed to 
find the minima of functions, rather than maxima. Now let us apply the method of MLE to the linear regression 
setting. Inserting the definition of the Gaussian into the above, we find that the log likelihood is given by

![](misc/images/new_log_likelihood.jpg)

RSS stands for residual sum of squares and is defined by

![](misc/images/rss.jpg)

The RSS is also called the sum of squared errors, or SSE, and SSE/N is called the mean squared error or MSE. 
It can also be written as the square of the ℓ2 norm of the vector of residual errors:

![](misc/images/sse.jpg)

Linear regression can be made to model non-linear relationships by 
replacing x with some non-linear function of the inputs, φ(x). That is, we use

![](misc/images/non_linear.jpg)

where could be degrees of features - ![](misc/images/feature_degrees.jpg)

Maximum likelihood estimation can result in overfitting. In some cases, our task – minimizing RSS could have infinite solutions. 
Could you give us a few examples of such cases?  Take a break and think. Answer: 
So, several common cases for this problem – high correlation features and cases when dimension of features more 
than dimension of object. Subsequently there are problems with the huge value of weights and overfitting. 
To solve this problem we can add regularization parameter for our minimizing equation. 
There are two basic regularization parameter ℓ2 and ℓ1. 

$`l2 = ||w||_2 = \sum_{l=1}^d w_i^2`$  
$`l1 = ||w||_1 = \sum_{l=1}^d |w_i|`$

From probability point of view, we can define regularization through MAP (Maximum A Posterior estimate) estimation 
with a zero-mean Gaussian prior on the weights, 

![](misc/images/ridge_regression.jpg)

This is called ridge regression. In more detail, we compute the MAP estimate as follows: 

![](misc/images/map.jpg)

> Source: [Kevin Murphy, Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html)

is the ℓ2 norm of the vector w. Thus we are penalizing weights that become too large in magnitude. 
In general, this technique is called ℓ2 regularization or weight decay, and is very widely used.

## Chapter III. Goal

The goal of this task is to get a deep understanding of the linear models for regression. 

## Chapter IV. Instructions

* This project will only be evaluated by humans. You are free to organize and name
your files as you desire.
* Here and further we use Python 3 as the only correct version of Python.
* For training deep learning algorithms you can try [Google Colab](https://colab.research.google.com). It offers kernels
(Runtime) with GPU for free that is faster than CPU for such tasks.
* The norm is not applied to this project. Nevertheless, you are asked to be clear
and structured in the conception of your source code.
* Store the datasets in the subfolder data

## Chapter V. Task

We will continue our practice with a problem from Kaggle.com. In this chapter we will implement 
all models that were described upstairs. Measure quality metrics on train and test parts. 
Will detect overfitted models and regularize these. And dive more deeply with native model estimation and comparison.

1. Answer to the questions
   1. Derive an analytical solution for the regression task. Use a vector form of equation.
   2. What changes in solution when adding L1 and L2 regularization to the loss function.
   3. Explain why L1 regularization is commonly used to select features. Why are there a lot of weights that are equal to 0 after the model is fitted?
   4. Explain how you can use the same models (Linear Regression, Ridge, etc) but make it possible to fit nonlinear dependencies?
2. Introduction  - make all preprocess staff from the previous lesson
   1. Import libraries. 
   2. Read train and test parts.
   3. Preprocess “interest level” feature.
3. Intro data analysis part 2
   1. Lets generate additional features for better model quality. Consider a column named “features”. It consists of a list of highlights of the current flat. 
   2. Remove unused symbols ([,],’,” and space) from the column.
   3. Split values in every row with separator “,” and collect the result in one huge list for the whole dataset. You may use DataFrame.iterrows().
   4. How many unique values consist of a result list?
   5. Let's get to know the new library - Collections. With this package you could effectively get quantity statistics about your data. 
   6. Count the most popular features of our huge list and take the top 20 for this moment.
   7. If everything correct you should get next values:  'Elevator', 'HardwoodFloors', 'CatsAllowed', 'DogsAllowed', 'Doorman', 'Dishwasher', 'NoFee', 'LaundryinBuilding', 'FitnessCenter', 'Pre-War', 'LaundryinUnit', 'RoofDeck', 'OutdoorSpace', 'DiningRoom', 'HighSpeedInternet', 'Balcony', 'SwimmingPool', 'LaundryInBuilding', 'NewConstruction', 'Terrace'.
   8. Now generate 20 new features based on top 20 values: 1 if value in “feature” column else 0.
   9. Extend our features set with  'bathrooms',  'bedrooms', 'interest_level' and lets create a special variable feature_list with all feature names. Now there are 23 values. All models should be trained on these 23 features.
4. Models implementation - linear regression
   1. Implement python class for a linear regression algorithm with two basic methods - fit and predict. Use stochastic gradient descent to find optimal weights of the model. For better understanding we recommend additionally implementing separate versions of the algorithm with the analytical solution and non-stochastic gradient descent under the hood.
   2. Give definition for R squared (R2) coefficient and implement function for calculation.
   3. Make predictions with your algorithm and estimate model with MAE, RMSE and R2 metrics.
   4. Initialize LinearRegression() from sklearn.linear_model, fit model and predict train and test parts, same as previous lesson.
   5. Compare quality metrics and make sure that difference is small. (Between your implementations and sklearn).
   6. Save metrics as in the previous lesson in a table with columns model, train, test for MAE table, RMSE table and R2 coefficient.
5. Regularized models implementation - Ridge, Lasso, ElasticNet    
   1. Implement Ridge, Lasso, ElasticNet algorithms: extend loss function with L2, L1 and both regularization correspondingly.
   2. Make predictions with your algorithm and estimate the model with MAE, RMSE and R2 metrics.
   3. Initialize Ridge(), Lasso() and ElasticNet() from sklearn.linear_model, fit model and make prediction for train and test samples, same as previous lesson.
   4. Compare quality metrics and make sure that difference is small. (Between your implementations and sklearn).
   5. Save metrics as in the previous lesson in a table with columns model, train, test for MAE table, RMSE table and R2 coefficient.
6. Feature normalization
   1. First of all, please write several examples why and where features normalization is mandatory and vice versa.
   2. Here let's consider the first of classical normalization methods - MinMaxScaler. Please write a mathematical formula for the method.
   3. Implement your own function for MinMaxScaler feature normalization.
   4. Initialize MinMaxScaler() from sklearn.preprocessing.
   5. Compare feature normalization with your own method and from sklearn.
   6. Repeat steps from b to e for one more normalization method StandardScaler.
7. Fit models with normalization
   1. Fit all models - Linear regression, Ridge, Lasso and ElasticNet with MinMaxScaler.
   2. Fit all models - Linear regression, Ridge, Lasso and ElasticNet with StandardScaler.
   3. Add all results to our dataframe with metrics on samples.
8. Overfit models
   1. Let's consider an overfitted model in practice. After theory, you know that polynomial regression is easy to overfit. So let's create toy example and see how regularization works in real life.
   2. In the last lesson we created polynomial features with degree equals 10. Here repeat these steps from the previous lesson, remember, with only 3 basic features - 'bathrooms’, 'bedrooms’, ‘'interest_level'.
   3. And train and fit all our implemented algorithms - Linear regression, Ridge, Lasso and ElasticNet on a set of polynomial features.
   4. Store results of quality metrics in the result dataframe.
   5. Analyze results, and choose the best model for your opinion.
   6. Addition try different alpha parameters of regularization in algorithms, choose best and analyze results.
9. Native models
   1. Calculate metrics for mean and median from previous lesson and add results to final dataframe.
10. Compare results
    1. Print your final tables
    2. What is the best model?
    3. What is the most stable model?
11. Addition task
    1. There are some tricks with target variable for better model quality. If we have a distribution with a heavy tail, you can use some monotonic function to “improve” distribution. In practice you can use logarithmic functions. We recommend you do this exercise and compare results. But don’t forget to perform inverse transformation when you will compare metrics.
    2. Next trick is outliers. The angle of the linear regression line strongly depends on outlier points. And often you should remove these points from !allert! only train data. You should explain why it was removed only from the train sample.  We recommend you do this exercise and compare results.
    3. Also it will be a useful exercise to realize linear regression algorithm with batch training.

### Submission

Save your code in python JupyterNotebook. Your peer will load it and compare it with the basic solution. 
Your code should contain the answers to all mandatory questions. Task ‘additional’ is on your own. 


💡 [Tap here](https://forms.yandex.ru/cloud/646b4693e010db2a75f1f5e6/) **to leave your feedback on the project**. Pedago Team really tries to make your educational experience better.
