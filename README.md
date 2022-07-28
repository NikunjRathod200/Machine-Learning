# Machine-Learning

Pandas library gives us excel like structure.

import pandas as pd

pd.read_csv(path).
pd.read_excel(path).
pd.read_json(path).

df= pd.read_csv(path) -- to save the data in df.

df.head()--- First five line of the data.


Modeling --- two types;

1st is SUpervised and 2nd is Unsupervised 

Supervised- Labeled data for target & independent variable.

Unsupervised -- Unlabeled data.

Supervised is further divided into two types

1st is if y=continues(regression) and y=descrete(catogarical)(classification)


Regression.
ex- rain fall as a function of temperature
  - prices of petrol
  - stock prises
  - gdp
  - sales
  - tourism


Linear-Regression(most used) --> error
K nearest neighbor--> euclidian distance
Support vector machine--> kernal/margin
Decision tree-- >entropy
Random forest-- > averaging/ensamble


Regression --Forecast / PRedict and make policy decision

there will be best fit line,
there will be error points.

all the points above the lines are positive errors and below the lines are negitive errors.

So which is the best line , How to find it..
Sol--> the Line for which the minerror is minimum.

minerror = sum of all errors with their actual signs. 

Ordinary Least Square method--> individually square all the errors and add them then whichever line gives us the minimum sum of the errors will be our best fit line.

spet-1 import the library


step-2 define X and Y.
y= df['element'] but x= [['element']] because x is column type and y is row type.



step-3 Define model.

from sklearn.linear_model import LinearRegression.
Ordinary least square method.

when y is in the from of binary, we use the logistic regression.


step-4 train model.
step-5 prediction.

y=b0 + b1*(Exp) + c
where y=salary
	b0 = intercept
	b1= slope
	Exp = experience
	c= Constant
this is for 1 x-axis value  --> line

y=b0 + b1*(exp) + b2*(Edu) + c
this is for 2 x-axis value --> plane

y=b0 + b1*(exp) + b2*(Edu) + b3*(gender) + c
this is for more than 2 x-axis value --> Hyperplane


step-6 find the model Accuracy.

(from sklearn.metrics import mean_absolute_percentage_error
mean_absolute_percentage_error(y_test,y_pred))




Linear regression-- Straight line curve.
Logistic regression -- 'S' tpe curve (bounded between (0,1)) { actually it gives the probability of whatever the y is meant for.}


In the Logistic Regression, the max iteration limit default set is 100.



Avoid two highly correlated features in the model to avoid multicolinearity.


White box : Tells you why/how ?

Black Box: Only Accuracy (ensenble tech)/(deep Learning) 


Classification --> { SVM, Discriminant Ana, Naive Bayse, Nearest Neighbour}

Regression --> { Linear Regresssion,GLM, SVR,GPR, Ensemble Methods , Decision Trees ,Neural Networks}

Clustring --> { K-means, Kmedodis, Fuzzy C-means, Heirarchical, Gaussian Mixture , neural Networkd, Hidden Markov Model}


Inbuild datasets - {sklearn->[from sklearn.datasets import ....],
 and seaborn->[ df = sns.load_datasets("name")] }



Regression || overall accuracy

Classification || category and class accuracy

The more we do cross validation the more it will take the machine time to calculate.

