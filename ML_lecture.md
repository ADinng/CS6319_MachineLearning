# Lecture 1

0. Introduction
	1. cs6405 Data mining
	2. cs6421 Deep learning
	3. cs6426 Data Visualization for Analytis Applications
	4. Jupyter
	5. calab
	6. mid-timetable 40%
	7. end-of-semester 40%
	8. lab assignment 20%
1. Cover topics
	1. Data Wrangling 数据整理
		1. reading and manipulating datasets.
		2. python(pandas, NumPy)
	2. Supervised Learning
		1. given a labelled dataset
			1. each sample has a label
			2. a real number in regression and an integer in classification.
		2. cover methods
			1. support vector machines.
			2. naive Bayes
			3. random forests
		4. necessary ideas
			1. accuracy metrics
			2. over-and under-fitting
			3. cross-validation
	3. Unsupervised Learning
		1. use unlabelled data
		2. Clustering
			1. grouping similar objects together
		3. ARM(association rule mining)
			1. finding patterns such as "customers who buy this also often buy that"
	4. Dimensionality Reduction
		1. reducing the number of dimensions or features in data
		2. Feature selection
			1. choosing a subset of the features
		3. feature learning
			1. combining existing features into a smaller set of new ones.
		4. cover methods
			1. principal components analysis
			2. auto encoders
	5. Reinforcement Learning
		1. learning to making sequential decisions to maximise a reward.
		2. methods
			1. Q-learning
			2. Monte Carlo tree search
	6. Deep Learning
		1. use of neural networks and is a huge topic.
	7. learning outcomes
		1. Identify what types of machine learning fits an application
		2. Select an appropriate software tool to solve it.
		3. Interpret and present the solution effectively
		4. work with real-world data sets from diverse domains and formats.
2. Some ML applications
	1. analyzing images of products on a production line to automatically classify them.
	2. Automatically classifying news articles.
	3. Forecasting a company's revenue.
	4. ..
3. Software packages.
	1. Scikit-learn
		1. ML libraries for python.
	2. Colab
	3. Numpy
		1. NumPy is the main array programming library for Python and it avoids inefficient Python loops.
	4. Matplotlib(plotting graphs and other displays)
	5. SciPy
		1. Python scientific computing library, built on top of NumPy and Matplotlib.
	6. Pandas
		1. for data manipulation and analysis.
		2. provides data structures and operations for manipulating numerical tables and time series, and interacts with NumPy.
		3. merging, reshaping and selecting. data cleaning and data wrangling features.
		4. 2 main data structures
			1. a series is like a single column of values.
			2. DataFrame contains 1 or more series and is like a 2D table.
4. Supervised learning: Regression
	1. Regression
		1. have a dataset of labelled instances.
		2. data samples are labelled in some way: the labels might be numbers or classes.
		3. train an ML algorithm to recreate these labels, and extrapolate to unseen data.
		4. variables called attributes or features.
	2. Linear regression
```python

# Linear regression
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression

  

np.random.seed(0)
n = 20 #n is the number of data points (20)
x = np.linspace(0, 10, n)
y = x*2 + 1 + np.random.randn(n) # y=2x+1


linreg = LinearRegression()
linreg.fit(x[:,np.newaxis], y)
yfit = linreg.predict(np.transpose([x]))


plt.plot(x, y, 'o')
plt.plot(x, yfit)
```

# Lecture 2
1. Polynomial regression, under- and overfitting(多项式回归，欠拟合和过拟合)
	1. underfifitting
``` python
# lecture 02 0115/2025
# Polynomial regression, under- and overfitting
n=20
x=np.linspace(0, 4, n)
y=-x*x*x + 6*x*x -9*x + 2 + 0.2 * np.random.randn(n)
 

linreg.fit(np.transpose([x]),y)
yfit=linreg.predict(np.transpose([x]))

plt.plot(x, y,'o')
plt.plot(x, yfit)
		
		```

	2.again badly underfitting the data.
``` python
linreg.fit(np.transpose([x,x*x]),y)
yfit=linreg.predict(np.transpose([x,x*x]))

# again badly underfitting the data.
plt.plot(x, y,'o')
plt.plot(x, yfit)
```
	1. shows a curve that fits the data well.
``` python
linreg.fit(np.transpose([x,x*x,x*x*x]),y)
yfit=linreg.predict(np.transpose([x,x*x,x*x*x]))


# overfitting
plt.plot(x, y,'o')
plt.plot(x, yfit)
```
2. Real-world data
	1. Categorical(or qualitative)
		1. Nominal: no ordering of classes(staff/patient, or windows/max/linux)
		2. Ordinal: ordered(young/old, small/medium/large)
	2. Numerical(or quantitative)
		1. Discrete
		2. Continuous
	3. possible problems
		1. missing values
		2. outliers
		3. very atypical values
	4. often need to do:
		1. cleaning and transforming
		2. data wrangling
		3. preprocessing
3. Californian housing dataset
	1. Observe the data 观察数据
``` python
%% observe the data %%
import pandas as pd

housing = pd.read_csv("https://github.com/ageron/handson-ml/raw/master/datasets/housing/housing.csv")

housing.head()
housing.info()
housing["ocean_proximity"].value_counts()
housing.describe()
```

	2. Data visualisation 
``` python
# Data visualisation
housing.hist(bins=5, figsize=(20, 15))
housing.plot(kind="scatter", x="longitude", y="latitude")
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, c="median_house_value", cmap=plt.get_cmap("jet"))
```

	3.Create training and testing data
``` python
from sklearn.model_selection import train_test_split
trainset, testset = train_test_split(housing, test_size=0.2)
print(trainset)
print(testset)
```

# lecture 03
Californian housing dataset
1. Separate the labels
	1. if columns (axis 1) or rows (axis 0)
``` python
# separate the labels
traindata = trainset.drop("median_house_value", axis=1)
trainlabs = trainset["median_house_value"].copy()

```
2.  Separate categorical and numerical attributes

``` python
# Separate categorical and numerical attributes
traincat = traindata[["ocean_proximity"]]
trainnum = traindata.drop("ocean_proximity", axis=1)
print(traincat)
print(trainnum)
```

3. Handling missing value

	mean均值，median中位数，most_frequent众数
```python

# Handling missing values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
imputer.fit(trainnum)
X = imputer.transform(trainnum)
trainnum= pd.DataFrame(X, columns=trainnum.columns)
```

4. Feature scaling
	min-max scaling(normalisation)
		(x-min)/(max-min)
		适用于数据分布已知且范围固定的情况
	standardisation: subtract the mean value µ and divide by the standard deviation σ
		(x-µ)/σ
		  μ 是均值，σ 是标准差
		适用于数据分布未知或存在异常值的情况
		- 数据分布范围较大，且部分特征存在右偏分布和异常值。
		- StandardScaler` 对异常值不敏感，适合线性模型（如线性回归、逻辑回归）和距离-based 模型（如 KNN、SVM）
	
```python
# # 选择一种方法进行数据缩放
# # 方法 1: MinMaxScaler (将值缩放到 [0, 1])
# scaler = MinMaxScaler()
# trainnum = pd.DataFrame(scaler.fit_transform(trainnum), 
# columns=trainnum.columns)
# # 方法 2: StandardScaler (将数据标准化为均值 0 和标准差 1)
# scaler = StandardScaler()
# trainnum = pd.DataFrame(scaler.fit_transform(trainnum), 
# columns=trainnum.columns)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
trainnumr = scaler.fit_transform(trainnum)
```

5. Regression on the numerical attributes
	rmse 越小，模型预测越准确 ,关注异常值
	mae, 稳健易解释的指标
```python
# import

from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
linreg.fit(trainnumr, trainlabs)
predictions=linreg.predict(trainnumr)


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(trainlabs, predictions)
rmse = np.sqrt(mse)
print("Train numeries LIN error = ", rmse)
%% Train numerics LIN error = 69082.78464568175 %%
```

6. Combining numerical attributes
```python
trainnum["rooms_per_household"]=trainnum["total_rooms"]/trainnum["households"]

trainnum["bedrooms_per_room"]= trainnum["total_bedrooms"]/trainnum["total_rooms"]

trainnum["population_per_household"]=trainnum["population"]/trainnum["households"]

imputer.fit(trainnum)

X = imputer.transform(trainnum)
trainnum = pd.DataFrame(X, columns=trainnum.columns)
trainnumr = scaler.fit_transform(trainnum)



linreg.fit(trainnumr, trainlabs)
predictions = linreg.predict(trainnumr)
mse = mean_squared_error(trainlabs, predictions)
rmse = np.sqrt(mse)

print("Train extended numericas LIN error = ", rmse)
%% Train extended numericas LIN error = 68559.36183563367 %%
```

7. Feature selection
	1. Spearman's rank correlation 皮尔逊相关系数衡量两个变量之间的线性关系
		1. 1 完全正相关
		2. -1 完全负相关
		3. 0 无线性相关
	2. 一般选择绝对值较大的特征：
		1. |0.3-0.5| 中等相关性
		2. |>0.5| 强相关性
	3. anything correlated with itself gives a value of 1 so we can ignore that value
```python
corr_matrix=trainset.drop("ocean_proximity", axis=1).corr()
print(corr_matrix["median_house_value"])

%% longitude -0.047998 latitude -0.143619 housing_median_age 0.111647 total_rooms 0.140739 total_bedrooms 0.055268 population -0.018309 households 0.071222 median_income 0.693173 median_house_value 1.000000 Name: median_house_value, dtype: float64 %%


housing.plot(x="median_income", y="median_house_value", kind="scatter", alpha=0.1)



trainnum1 = trainnumr[:, [1, 2, 3, 7, 9, 10]] linreg.fit(trainnum1, trainlabs) predictions = linreg.predict(trainnum1) 
mse = mean_squared_error(trainlabs, predictions) 
rmse = np.sqrt(mse) print("Train selected numericas LIN error = ", rmse) 
%% Train selected numericas LIN error = 77316.06904450871 %%

``` 


# lecture 04
1.Use categorical attributes
	one-hot encoding, creates a binary attribute for each category.
		a one-hot-encoder normally returns a SciPy sparse matrix, which stores only the non-zero values to save memory.
```python
# 01.22 2025 lecture 04

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
trainhot = cat_encoder.fit_transform(traincat).toarray()

trainnew = np.concatenate([trainnumr, trainhot], axis=1)
# trainnew = np.hstack((trainnumr, trainhot))

linreg.fit(trainnew, trainlabs)
predictions = linreg.predict(trainnew)
mse = mean_squared_error(trainlabs, predictions)
rmse = np.sqrt(mse)
print("Train combined arrributes LIN error = ", rmse)
# Train combined arrributes LIN error = 67765.74947521035
```
2.other regressors
	decision tree
		they can handle numerical and categorical attributes so there's no need for the one-hot-encoding here.
		the result is 0, probably that the model has badly overfitted the data.
```python
# decsion tree

from sklearn.tree import DecisionTreeRegressor

treereg = DecisionTreeRegressor()
treereg.fit(trainnew, trainlabs)
predictions = treereg.predict(trainnew)
mse = mean_squared_error(trainlabs, predictions)
rmse = np.sqrt(mse)
print("Train combined attributes DT error = ", rmse)
# Train combined attributes DT error = 0.0
```

	The testding data is for evaluation, and the training data for learning.
```python
testdata = testset.drop("median_house_value", axis=1)

testlabs = testset["median_house_value"].copy()

  

testnum = testdata.drop("ocean_proximity", axis=1)

testcat = testdata[["ocean_proximity"]]

  

testnum["rooms_per_household"] = testnum["total_rooms"]/testnum["households"]

testnum["bedrooms_per_room"] = testnum["total_bedrooms"]/testnum["total_rooms"]

testnum["population_per_household"] = testnum["population"] / testnum["households"]

  

X = imputer.transform(testnum)

testnum = pd.DataFrame(X, columns=testnum.columns)

  

testnumr = scaler.transform(testnum)

  

testhot = cat_encoder.fit_transform(testcat).toarray()

testnew = np.concatenate([testnumr, testhot], axis=1)

predictions = treereg.predict(testnew)

mse = mean_squared_error(testlabs, predictions)

rmse = np.sqrt(mse)

print("Test combined attributes DT error = ", rmse)
#Test combined attributes DT error = 70878.16784855767
```

	random forest
```python
from sklearn.ensemble import RandomForestRegressor 

rfreg = RandomForestRegressor()
rfreg.fit(trainnew, trainlabs)

#Testing

predictions = rfreg.predict(testnew)

mse = mean_squared_error(testlabs, predictions)

rmse = np.sqrt(mse)

print("Test combined attributes RF error = ", rmse)
# Test combined attributes RF error = 51474.77108788628
```

3.What else can we do?
	1. a reasonable approach is to try a lot of methods with default values, or perhaps try a bit of tuning, and narrow the search down to (say) 2-5 most promising methods.
	2. try adding more attributes.
	3. for rescaling we could try normalisation instead of standardisation
	4. we could try deleting outliers.
	5. we could use stratified sampling to split the data.
	6. regression methods can also give us useful feedback, the random forest tells us the relative improtance of each attribute, we could use that for feature selection.

4.Notes on Colab
5.Notes on Lab 1
6.**Notes on data wrangling**
	1. Data preparation
	2. Python arrays
		1. python lists
			1. mylist【start:stop:step】
			2. mylist.extend([x, y])
			3. mylist[len(mylist):]=[x, y]
		2. Python tuples
			4. tuples are immutable, they can't be changed.
		3. python dictionaries
	3. Numpy arrays


7.complete code
```python
import pandas as pd

housing = pd.read_csv("https://github.com/ageron/handson-ml/raw/master/datasets/housing/housing.csv")

  

from sklearn.model_selection import train_test_split

trainset, testset = train_test_split(housing, test_size=0.2)

  

traindata = trainset.drop("median_house_value", axis=1)

trainlabs = trainset["median_house_value"].copy()

  

trainnum = traindata.drop("ocean_proximity", axis=1)

traincat = traindata[["ocean_proximity"]]

  
  

# Combining numerical attributes

trainnum["rooms_per_household"]=trainnum["total_rooms"]/trainnum["households"]

trainnum["bedrooms_per_room"]= trainnum["total_bedrooms"]/trainnum["total_rooms"]

trainnum["population_per_household"]=trainnum["population"]/trainnum["households"]

  

# Handling missing values

from sklearn.impute import SimpleImputer

# Imptuation

imputer = SimpleImputer(strategy="median")

imputer.fit(trainnum)

X = imputer.transform(trainnum)

trainnum = pd.DataFrame(X, columns=trainnum.columns)

  

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

trainnumr = scaler.fit_transform(trainnum)

  

from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()

trainhot = cat_encoder.fit_transform(traincat).toarray()

trainnew = np.concatenate([trainnumr, trainhot], axis=1)

  

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

  

linreg = LinearRegression()

linreg.fit(trainnumr, trainlabs)

predictions = linreg.predict(trainnumr)

mse = mean_squared_error(trainlabs, predictions)

rmse = np.sqrt(mse)

print("Train extended numericas LIN error = ", rmse)
```
# Lecture 05
## 1.Numpy arrays
```python
b = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(b[0:2, 1:4])
# b[0:2, 1:4] 切片操作，[行切片，列切片]
# 0:2 从0行到第2行(不包括第2行)
# 1:4 从第1列到第4列(不包括第4列)

y = a.view() #修改a,也会修改b，共享相同的数据内存

c = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
d = c.reshape(4, 3)

a = np.array([[1, 2, 3], [4, 5, 6]])
b = a.reshape(-1) #[1 2 3 4 5 6]

# conbine two 1D arrays into a 2D array we can stack them:
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.stack((a,b), axis=1)
print(c)

#[[1 4] [2 5] [3 6]]
```

## 2.Pandas Series and DataFrames
	developed by statisticians
```python
import pandas as pd
a = np.array([6,7,8])
s = pd.Series(a)
s = pd.Series(a, index=['a','z','hello'])
print(s)
x = s.values
print(x)
```
## 3.Cross-validation
1.K-fold cross-validation
	Shuffle the dataset randomly
	Split the dataset into K groups G1, . . . , GK
	For each group Gi :
	Fit a model to training set G1 ∪ . . . ∪ Gi−1 ∪ Gi+1 ∪ . . . , GK
	Evaluate the model on test set Gi
	Take the mean (and stddev) of the scores
	数据集划分更固定，适用于数据量较小或希望每个数据点都被测试一次的场景
```python
# Cross-validataion

# k-fold cross-validation

from sklearn.model_selection import cross_val_score

# scoring,指定评分标准负均方误差
# cv= 10, 10折交叉验证，划分10个子集（每次9个子集训练，1个子集验证）
scores = cross_val_score(linreg, trainnew, trainlabs, scoring="neg_mean_squared_error", cv=10)
# scoressq = cross_val_score(rfreg, trainnew, trainlabs,scoring="neg_mean_squared_error", cv=10)

rmse = np.sqrt(-scores)

print("XVAL LIN RMSE mean =",rmse.mean()," stddev =",rmse.std())
#XVAL LIN RMSE mean = 67929.13193437677 stddev = 2602.8838783869105
```

2.Shuffle split cross-validataion
	适合数据量大，多次随机划分评估模型性能的场景
	randomly split the training data into(say) 80% and 20% subsets, fit to the 80% and test on the 20%. repeat this several times, with different random splits, and find the average error.
```python
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

split = ShuffleSplit(n_splits=10, test_size=0.2)

scores = cross_val_score(linreg, trainnew, trainlabs, scoring="neg_mean_squared_error", cv=split)

rmse=np.sqrt(-scores)

print("XVALSHUF LIN RMSE mean = ",rmse.mean()," stddev =",rmse.std())
```

3.Leave-one-out cross-validation
	1. dataset so small so need this method
	2. use all but one row for training, and test on that one row.
	3. this method gives accurate means but with high variance, as each figuire has a single test sample.
4.The holdout method
	1. split off part of the training data to form a validation set (or development set). train on the reduced training data, perhaps using several regressors or several parameter settings. 
	2. Then choose the best one.
	3. Finally evaluate that one on the test set, and report the result.
	4. Holdout can be used to avoid overfitting.
5.Notes on cross-validation

## 4.Notes on correlation
1. if we know the values of other variables that correlate with it, then we can use them to predict it.
2. Pearson's correlation $$
r_{X,Y} = \frac{\text{cov}(X, Y)}{\text{stddev}(X)\text{stddev}(Y)} = \frac{E[(X - \mu_X)(Y - \mu_Y)]}{\sigma_X \sigma_Y}
$$

	1. If rX,Y = 0 then X, Y are independent variables, if rX,Y > 0 they’re positively correlated, and if rX,Y < 0 they’re negatively correlated. Usually we’re interested in correlated variables, whether positive or negative
	2. linearly related.
3. Spearman's correlation
	1. no linear relationship
	2. if we have (X,Y) pairs(1,2),(4,3),(16,100)
	3. then $r_s$ is 1 because they both increase together. If decrease as X increase then $r_s$=-1
4. Spurious correlation(伪相关)
	1. a confounding variable
	2. correlation does not imply causation
	3. correlation is just a coincidence
5. Correlation on time series
	1. we should only apply it to the right kind of data.
	2. it shouldn't be applied to time series data.
	3. time series we should use another measure such as cross-correlation.
6. Correlation on ratios
	1. we also shouldn't use the usual kind of correlation between two variables that are ratios with another variable as common denominator.
7. Milton Friedman's thermostat
	1. correlation doesn't imply causation.
8. 


# Lecture 06 Supervised learning: classification

## 1. Supervised learning: classification
1. the dependent variable is nominal instead of numeric
2. we want to predict a class: yes/no, cat/dog/horse and so on.
### 1.1 Binary classification: electrical grid stability
1. Binary classification means that there are just 2 possible labels, which is the most common situation.

``` python
# binary classification: electrical grid stability

import pandas as pd

import numpy as np

import sklearn

griddata = pd.read_csv("https://archive.ics.uci.edu/ml/machinelearning-databases/00471/Data_for_UCI_named.csv")

griddata = griddata.drop("stab", axis=1)

gridvars = griddata.drop("stabf", axis=1)

gridlabs = griddata["stabf"].copy()

  

# check for missing values

gridvars.info()

  

gridvars.describe()

  

# use normalisation or standardisation to make all variables between 0 and 1.

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

gridvarsr = scaler.fit_transform(gridvars)

  

# sklearn classifiers expect class labels to be in the form 0,1,2,...

# we need to transform the stable/unstable values.

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

gridlabse = le.fit_transform(gridlabs)

  
  

# linear support vector classifier, and rmse is inappropriate here: instead use accuracy

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import cross_val_score

from sklearn.svm import LinearSVC

  

classifier = LinearSVC()

split = ShuffleSplit(n_splits=10, test_size=0.2)

scores = cross_val_score(classifier, gridvarsr, gridlabse, scoring="accuracy", cv=split)

print("LS acc mean =",scores.mean()," stddev =",scores.std())

  
  

# decision tree

from sklearn.tree import DecisionTreeClassifier

  

classifier = DecisionTreeClassifier()

split = ShuffleSplit(n_splits=10, test_size=0.2)

scores = cross_val_score(classifier, gridvarsr, gridlabse, scoring="accuracy", cv=split)

print("DT acc mean =",scores.mean()," stddev =",scores.std())

  

# random forest

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier()

split = ShuffleSplit(n_splits=10, test_size=0.2)

scores = cross_val_score(classifier, gridvarsr, gridlabse, scoring="accuracy", cv=split)

print("RT acc mean =",scores.mean()," stddev =",scores.std())

  

# Support vector classifier

from sklearn.svm import SVC

classifier = SVC()

split = ShuffleSplit(n_splits=10, test_size=0.2)

scores = cross_val_score(classifier, gridvarsr, gridlabse, scoring="accuracy", cv=split)

print("SV acc mean =",scores.mean()," stddev =",scores.std())

  

# K-Nearest-Neighbours

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()

split = ShuffleSplit(n_splits=10, test_size=0.2)

scores = cross_val_score(classifier, gridvarsr, gridlabse, scoring="accuracy", cv=split)

print("KN acc mean =",scores.mean()," stddev =",scores.std())

```
1. sklearn have 3 interfaces: an estimator, a predictor and a transformer.
2. an estimator can be a regressor, a classifier, or something else such as an imputer or a rescaler.
3. It's common to use capital X to indicate that there might be several x values, and lower-case y to indicate that these labels are single values. 

```python
# sklearn have interfaces: an estimator, a predictor and a transformer.

myclassifier = DecisionTreeClassifier()

  

# specify some parameter values at this point

myclassifier = DecisionTreeClassifier(gama=2)

  
  

# train the estimator by fitting it to training data.

# It's common to use capital X to indicate that there might be several x values, and lower-case y to indicate that these labels are single values.

myclassifier.fit(Xtrain, ytrain)

  

# A predictor applies the trained estimator to new X values (test data)to predict y

ypredict = myclassifier.predict(Xtest)

  

# A transformer is something used by some estimators to transform the data, eg after creating and training a standard scaler

myscaler = StandardScaler()

myscaler.fit(Xtrain)

Xtrain = myscaler.transform(Xtrain)

  

# combine fitting and transforming

myscaler.fit(Xtrain).transform(Xtrain)

  

# more efficiently

myscaler.fit_transform(Xtrain)

```
### 1.2 Classification and regression methods
1. Linear methods
	1. multiple linear regression
	2. LinearSVC(the linear support vector classifier)
2. Decision trees
	1. can handle both numerical and categorical data.
	2. but in sklearn they can only handle numerical data. so we have to use one-hot encoding or some other conversion method.
	3. choosing a variable and splitting its range of values. at some point it stops doing this, and makes its prediction, whether numerical or categorical.
	4. how do we decide which variables to choose, what value to use for splitting.
		1. it uses clever heuristics to make these choices. which aim of minimising the training error.
	5. advantages:
		1. they can't extrapolate to values they didn't see before.
		2. In sklearn, they can't be used when there are missing values so we need to impute.
	6. advantages:
		1. they are easy to understand.
		2. they're quite immune to outliers.
		3. they can model very nonlinear relationships.
		4. they're fast to train.
	7. decision trees are prone to overfitting, and they heuristics they use can make bad choices.
	8. use for regression and classification.
3. Random forests
	1. combines multiple decision trees to make a more accurate and robust prediction.
		1. train several trees, with randomness introduced to get some variation, then take an average prediction.
	2. random forests are an example of **ensemble learning**: we can do the same with any type of classifier or regressor, and even mix them to get different points of view.
	3. use for regression and classification.
4. support vector machines
	1. maximum-margin hyperplane
	2. only measure the distance between the plane and the nearest points, which are called support vectors.
		1. the idea is to improve the classifies's ability to generalise, and avoid overfitting.
	3. hinge loss.
	4. we can use 1-dimensional or 2-dimensional to separate dataset.
	5. Replacing (x, y) by (say) (x, y, $x^2$ + $y^2$ ) means that there’s a plane separating them.
	6. kernel trick
		1. linear for a linear classifier
		2. poly for a ploynomial classifier, the degree of the polynomial(2 for quadratic, 3 for cubic, etc) is controlled by the degree parameter, default 3
		3. rbf for a radial basis function, whose value depends only on the distances between points and a fixed point such as the origin
		4. sigmoid for a tanh sigmodial function
	7. sklearn provides SVC for classification and SVR for regression.
	8. support vector regression: fit a hyperplane with maximum margin, with a kernel for nonlinear margins.
	9. SVMs can be very accurate and often avoid overfitting, but they don't scale well to big data and are less effective when the classes overlap.



# L7

## Classification and regression methods
1. Naive Bayes
	1. There are several Naive Bayes(NB) classifiers, which find the most likely class based on different assumptions about the distributions of the variables.
	2. mulitnomial NB : used for spam detection and document classification
		1. assumes that the variables are integers that count something.
	3. Bernoulli NB assumes that the variables are binary(or true/false), and the Bernoulli classifier reduces to simply taking a weighted sum of the binary varoables and testing whether the result is > or < a threshold value.
	4. Gaussian NB assumes that the variables are continuous and normally distributed.  It's possible to handle combinations of these variables, because they all deal with probabilities.
2. Logistic regression
	1. this is a classification method. not a regressor.
	2. it can be used when we have a **binary dependent variable**
	3. it's known that logistic regression tends to be more accurate than naive Bayes, given enough data.
	4. but in practice NB often does better when there's less training data, NB is also faster to train.
3. K-nearest neighbours
	1. training means simply remembering each example, and testing means finding the nearest training example and using its label as a prediction
	2. Euclidean.
		1. $\sqrt{(x-a)^2+(y-b)^2+(z-c)^2}$
	3. Manhattan distance
		1.  |x-a| + |y-b|+|z-c|
	4. Hamming distance
		1. |x-a| + |y-b|+|z-c|
	5. KNN classifier is popular because it makes no assumptions about the data
	6. on the other hand it's very sensitive to outliers, it's can't handle missing values, and it's use a lot of memory.
4. Neural networks
5. Ensemble methods
	1. Bagging
		1. bootstrapping aggregating
		2. well-suited to weak learners with low bias and high variance, such as deep decision trees, a random forest uses bagging to combine decision trees.
	2. Boosting
		1. sequentially
		2. adaboost, Gradient boosting, XGBoost, LightGBM
	3. Stacking
		1. combines different learning algorithms in parallel.
		2. for example , choose KNN, logistic regression, SVM,We train a meta-model such as a neural network to make predictions based on their predictions 
6. Symbolic classification and regression
	1. symbolic regression has been called "the forgotten ML method"
```python
!pip install gplearn

from gplearn.genetic import SymbolicClassifier
sc = SymbolicClassifier()

from sklearn.metrics import mean_absolute_error
sc.fit(X,y)
y1 = sc.predict(X)
mae= mean_absolute_error(y, y1)
```
7. Few-one, and zero-shot classifiers
	1. learn form small data
	2. one-shot learning is a special case in which we have just one training example.
8. One-class classifiers
	1. outlier detection
## Multiple classes and imbalanced datasets

0. MNIST dataset
	1. using accuracy isn't appropriate for imbalanced data.
```python
import pandas as pd

from sklearn.preprocessing import LabelEncoder

mnisttra = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra",header=None)

mnisttes = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes",header=None)

# split label

mnisttrad = mnisttra.drop(64, axis=1)

mnisttesd = mnisttes.drop(64, axis=1)

mnisttral = mnisttra[64].copy()

mnisttesl = mnisttes[64].copy()

ytra = (mnisttral == 5)

ytes = (mnisttesl == 5)

  

# label encoder

le = LabelEncoder()

ytrae = le.fit_transform(ytra)

from sklearn.svm import LinearSVC

from sklearn.model_selection import ShuffleSplit, cross_val_score

  

classifier = LinearSVC()

split = ShuffleSplit(n_splits=10, test_size=0.2)

score = cross_val_score(classifier, mnisttrad, ytrae, scoring="accuracy", cv=split)

print("LS accuracy mena=", score.mean())

# using accuracy isn't appropriate for imbalance data

```
1. Confusion matrix
```python
real = [0, 1, 0, 1, 0, 0, 0, 0]

pred = [1, 1, 0, 1, 0, 0, 0, 0]

  

from sklearn.metrics import confusion_matrix

print(confusion_matrix(real, pred))


from sklearn.model_selection import cross_val_predict

ypred = cross_val_predict(classifier, mnisttrad, ytrae, cv=3)

print(confusion_matrix(ypred, ytrae))
```

2. Alternative to accuracy & confusion matrices
	precision = tp/(tp+fp)
	recall = tp/(tp+fn)
	f1 = 2/(1/precision + 1/recall)
```python
from sklearn.metrics import precision_score, recall_score, f1_score

real = [0,1,0,1,0,0,0,0]

pred = [1,1,0,1,0,0,0,0]

  

print(precision_score(real, pred))

print(recall_score(real, pred))

print(f1_score(real, pred))



from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()

scores = cross_val_score(classifier, mnisttrad, ytrae, scoring="f1", cv = split)

print("DT f1 mean = ", score.mean()," stddev = ", scores.std())


from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier()

scores = cross_val_score(classifier, mnisttrad, ytrae, scoring="f1", cv = split)

print("DT f1 mean = ", score.mean()," stddev = ", scores.std())




from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()

scores = cross_val_score(classifier, mnisttrad, ytrae, scoring="f1", cv = split)

print("DT f1 mean = ", score.mean()," stddev = ", scores.std())

```

3. Precision/recall trade-off
	1. precision and recall get  F1 score
	2. F1 has be criticised because it ignores true negatives.
	3. another problem is its lack of symmetry

# L8

## 1.Multiclass classification
1. classifiers 
	1. mulitclass
		1. Bayes, KNN, neural networks, decision tree and random forests
	2. binary
		1. support vector machines, linear classifiers
	3. binary classifiers can be adapted to the muliticlass case using two methods
		1. One-versus-all(OvA, also called one-veruss-the-rest or OvR)
			1. train a binary classifier for each class, whichever class get the highest score is the selected as the prediction
			2. for other binary classifiers we usually use OvA
		2. One-versus-one(OvO)
			1. train the binary classifier for every possible pair of classes. whichever class wins most often is selected as the prediction. we need train the classifier N(N-1)/2 times
			2. SVM scales poorly to bid dataset, so we usually use Ovo.
	4. generate a confusion matrix of multiclass problems
		1. row represent actual classes
		2. columns represent predicted classes
``` python
mnisttrale = le.fit_transform(mnisttral)

classifier = DecisionTreeClassifier()

split = ShuffleSplit(n_splits=10, test_size=0.2)

scores = cross_val_score(classifier, mnisttrad, mnisttrale, scoring="accuracy", cv=split)

print("DT accuracy mean = ",scores.mean(), " stddev = ", scores.std())


# generate a confusion matrix for multiclass problem 
ypred = cross_val_predict(classifier, mnisttrad, mnisttrale, cv=3)

conf = confusion_matrix(ypred, mnisttrale)

print(conf)


import matplotlib as plt
plt.pyplot.matshow(conf, cmap=plt.cm.gray)
rowsums = conf.sum(axis=1, keepdims=True)
normconf = conf/rowsums
np.fill_diagonal(normconf, 0)

plt.pyplot.matshow(normconf, cmap=plt.cm.gray)


```

2. Case today
```python
firewall = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00542/log2.csv")

firewall['Action'].value_counts()

from sklearn.model_selection import train_test_split
firetrain, firetest = train_test_split(firewall, test_size=0.2)

# iloc[row index, colum index] 左闭又开，traindata 只取5-10列

traindata = firetrain.iloc[:,5:11]

trainlabs = firetrain.iloc[:,4]

testdata = firetest.iloc[:,5:11]

testlabs = firetest.iloc[:,4]



from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

traindatar = scaler.fit_transform(traindata)

testdatar = scaler.transform(testdata)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

trainlabse = le.fit_transform(trainlabs)

testlabse = le.transform(testlabs)


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
classifier = LinearSVC()

split = ShuffleSplit(n_splits=10, test_size=0.2)

scores = cross_val_score(classifier, traindatar, trainlabse, scoring="f1_macro", cv=split)

print("LS f1 mean =", scores.mean()," stddev =",scores.std())

```
3. Grid search
```python
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

params =[{'n_neighbors': [3, 5, 8, 13],

'weights': ['uniform','distance'],

'p':[1,2]}]

classifier = KNeighborsClassifier()

search = GridSearchCV(classifier, params, cv=5, scoring ="f1_macro")

search.fit(traindatar, trainlabse)

  

print("best params: ", search.best_params_)

print("best score: ", search.best_score_)


```


# L9
# L10 02122025
4. Lab 3
	1. sun 9 march
	2. pandas
	3. d = pd.DataFrame()
5. 


# L12 0219

6. Anomacl detection
7. classifier preprocessing
8. semi-supervised learning
9. forestry
10. ANnN conpression
11. Association rule mining
	1. {onious, potatoes}==> {burgers}
	2. market basket anacisis
	3. {bead, egg}==> {milk}
		1. support:
		2. confidence
		3. lift
	4. downward closure
	5. aprion
	6. eclat
	7. fp-growth













