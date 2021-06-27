# Data-Science-Practicum-Project-House-Price-Prediction
House Price Prediction using various Machine Learning Algorithms

**Abstract‌**

Housing prices are a crucial reflection of the economy of the country, and housing value ranges are of great interest for both buyers and sellers. In this project, house costs will be predicted given informative variables that cover many aspects of residential houses. We will be implementing various regression techniques including Linear Regression, SVM regression, Random Forest regression, XGBoost regression and CatBoost Regression techniques. We will also perform PCA to further reduce the error in prediction.

The goal of this project is to form a regression model to accurately estimate the price of the house given the features and to identify the important features that feed the model&#39;s predictive power.

**Keywords:** Multiple linear regression, Lasso Regression, XGBoost Regression, CatBoost Regression, Machine Learning, House Price Prediction.

# **Table of Contents**

1. Dataset Introduction

1. Exploratory Data Analysis

1. Data Preparation Methodology

1. Baseline Models

1. Feature Engineering

1. Feature Selection

1. HyperParameter Tuning And Results

1. References

# **Dataset Introduction**

The dataset contains the costs and features of residential houses sold from 2006 to 2010 in Ames, Iowa, obtained from the Ames Assessor&#39;s workplace. This dataset consists of seventy-nine house features and 1460 houses with sold prices. Although the dataset is comparatively tiny with solely 1460 examples, it contains seventy-nine features like areas of the homes, types of floors, and numbers of bathrooms. Such massive amounts of features enable us to explore varied techniques to predict house costs.

The dataset consists of options in varied formats. it&#39;s numerical features like prices and numbers of bathrooms/bedrooms/living rooms, as well as categorical

features like zone classifications for sale, which can be &#39;Agricultural&#39;, &#39;Residential High Density&#39;, &#39;Residential Low Density&#39;, &#39;Residential low-density Park&#39;, etc. in order to form this knowledge with totally different format usable for our algorithms, categorical knowledge was converted into

separated indicator data, which expands the amount of features during this dataset. We split our dataset into a training and testing set with a roughly 80/20 split, with 1000 coaching examples and 460 testing examples. Besides, {there were|there have been} some features that had values of N/A; we replaced them with the mean of their columns so that they don&#39;t influence the distribution.

# **Exploratory Data Analysis**

Data exploration is the first step in data analysis and typically involves summarizing the main characteristics of a data set, including its size, accuracy, initial patterns in the data, and other attributes. It is commonly conducted by data analysts using visual analytics tools, but it can also be done in more advanced statistical software, Python. Before it can conduct analysis on data collected by multiple data sources and stored in data warehouses, an organization must know how many cases are in a data set, what variables are included, how many missing values there are, and what general hypotheses the data is likely to support. An initial exploration of the data set can help answer these questions by familiarizing analysts with the data with which they are working.

In total there are 1460 training samples in our dataset and 86 different informative features in which thirty-six are quantitative and forty-three are numerical, the rest are &quot;Id&quot; and &quot;SalesPrice&quot; (target column).

The **Quantitave** Features are 1stFlrSF, 2ndFlrSF, 3SsnPorch, BedroomAbvGr, BsmtFinSF1, BsmtFinSF2, BsmtFullBath, BsmtHalfBath, BsmtUnfSF, EnclosedPorch, Fireplaces, FullBath, GarageArea, GarageCars, GarageYrBlt, GrLivArea, HalfBath, KitchenAbvGr, LotArea, LotFrontage, LowQualFinSF, MSSubClass, MasVnrArea, MiscVal, MoSold, OpenPorchSF, OverallCond, OverallQual, PoolArea, ScreenPorch, TotRmsAbvGrd, TotalBsmtSF, WoodDeckSF, YearBuilt, YearRemodAdd, and YrSold.

The **Categorical** Features are Alley, BldgType, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, BsmtQual, CentralAir, Condition1, Condition2, Electrical, ExterCond, ExterQual, Exterior1st, Exterior2nd, Fence, FireplaceQu, Foundation, Functional, GarageCond, GarageFinish, GarageQual, GarageType, Heating, HeatingQC, HouseStyle, KitchenQual, LandContour, LandSlope, LotConfig, LotShape, MSZoning, MasVnrType, MiscFeature, Neighborhood, PavedDrive, PoolQC, RoofMatl, RoofStyle, SaleCondition, SaleType, Street, and Utilities.

The table below shows a quantitative estimate of the percentage of missing values present in the column of the features of the dataset. As we can see that the features **Alley** , **PoolQC, Fence** and **Miscfeature** contains almost the entire column as blank so we decide to drop these features as imputing these features with any value would not make any sense.

![image](https://user-images.githubusercontent.com/16374886/123544347-7a8a4100-d718-11eb-8f06-a4b38ff31a34.png)

The graph below sums up the table fairly showing the four features as the ones having the maximum percentage of values as null values.

![image](https://user-images.githubusercontent.com/16374886/123544407-c0dfa000-d718-11eb-86fd-5a14dfc484ec.png)

Fig. 1: Countplot Showing the Number of missing values in each of the features of the dataset.

Following this, I did Univariate and Bivariate Analysis of Categorical and Numerical features separately. We observe that some categories seem to more diverse with respect to SalePrice than others. The neighborhood has a big impact on house prices. The most expensive seems to be Partial SaleCondition. Having a pool on the property seems to improve the price substantially. There are also differences in variabilities between category values.

When I plotted a countplot of features against target variable, I observe that &#39;BsmtFullBath&#39;, &#39;BsmtHalfBath&#39;, &#39;FullBath&#39;, &#39;HalfBath&#39;, &#39;Bedroom&#39;, &#39;Kitchen&#39;, &#39;KitchenQual&#39;, &#39;TotRmsAbvGrd&#39;, &#39;Fireplaces&#39;, &#39;GarageType&#39;, &#39;MiscVal&#39;, &#39;MoSold&#39;, &#39;YrSold&#39; variables are in int64/float64 type(i.e. They were considered as quantitative features), but they can be treated as categorical. The barplots below for these mentioned features imply the reason for the same.

![image](https://user-images.githubusercontent.com/16374886/123544428-d5239d00-d718-11eb-8ed0-9bdec88594e9.png)

Fig. 2: Countplots of features which were considered quantitative but can be converted to Categorical.

The three figures on the left show that the target column which is the SalesPrice does not follow a normal distribution, so before performing regression it has to be transformed. While log transformation does pretty good job, best fit is unbounded Johnson distribution.

![image](https://user-images.githubusercontent.com/16374886/123544470-069c6880-d719-11eb-99b4-35c39cdf8b88.png)

![image](https://user-images.githubusercontent.com/16374886/123544480-12882a80-d719-11eb-8470-225bc311141d.png)

![image](https://user-images.githubusercontent.com/16374886/123544488-1d42bf80-d719-11eb-9bd0-003b12f477f6.png)

![image](https://user-images.githubusercontent.com/16374886/123544524-44998c80-d719-11eb-8b40-b860a34b31e8.png)

On the left is the graph of Anova Test P values which gives a quick estimate of influence of categorical variables on SalePrice.

For each variable SalePrices are partitioned to distinct sets based on category values. Then check with ANOVA test if sets have similar distributions. If variable has minor impact then set means should be equal. Decreasing pval is sign of increasing diversity in partitions.

Fig. 4 - Anova test P-value graph

![image](https://user-images.githubusercontent.com/16374886/123544537-5418d580-d719-11eb-9f9d-8dcd87d64ba8.png)

Fig. 5: Spearman Correlation values plot of each feature with the &quot;SalesPrice&quot; target feature.

Spearman correlation is better to work with in this case because it picks up relationships between variables even when they are nonlinear. OverallQual is main criterion in establishing house price. Neighborhood has big influence, partially it has some intrisinc value in itself, but also houses in certain regions tend to share same characteristics (confunding) what causes similar valuations.

![image](https://user-images.githubusercontent.com/16374886/123544543-609d2e00-d719-11eb-8f69-ed4408d0a867.png)

Fig. 6: Relationship between overall quality of the house and its sales price. ![image](https://user-images.githubusercontent.com/16374886/123544555-77dc1b80-d719-11eb-8d41-203ad0d6e405.png)

Fig. 7: Relationship between the year in which the house was built and its sales price.

&#39;OverallQual&#39; and &#39;YearBuilt&#39; also seem to be related with &#39;SalePrice&#39;. The relationship seems to be stronger in the case of &#39;OverallQual&#39;, where the box plot shows how sales prices increase with the overall quality.

![image](https://user-images.githubusercontent.com/16374886/123544565-83c7dd80-d719-11eb-8684-b0e236abef9f.png)

Fig.8: Correlation Heatmap Plot

At first sight, there are two red-colored squares that get my attention. The first one refers to the &#39;TotalBsmtSF&#39; and &#39;1stFlrSF&#39; variables, and the second one refers to the &#39;GarageX&#39; variables. Both cases show how significant the correlation is between these variables. Actually, this correlation is so strong that it can indicate a situation of multicollinearity. If we think about these variables, we can conclude that they give almost the same information so multicollinearity really occurs. Heatmaps are great to detect this kind of situations and in problems dominated by feature selection, like ours, they are an essential tool.

Some points to note from the Correlation Plot:

- &#39;OverallQual&#39;, &#39;GrLivArea&#39; and &#39;TotalBsmtSF&#39; are strongly correlated with &#39;SalePrice&#39;.
- &#39;GarageCars&#39; and &#39;GarageArea&#39; are also some of the most strongly correlated variables. However, as we discussed in the last sub-point, the number of cars that fit into the garage is a consequence of the garage area. &#39;GarageCars&#39; and &#39;GarageArea&#39; are like twin brothers. You&#39;ll never be able to distinguish them. Therefore, we just need one of these variables in our analysis (we can keep &#39;GarageCars&#39; since its correlation with &#39;SalePrice&#39; is higher).

**Scatter plots between &#39;SalePrice&#39; and correlated variables**

Fig. 9: Scatterplot of Numerical features vs SalesPrice Variable ![image](https://user-images.githubusercontent.com/16374886/123544605-b5d93f80-d719-11eb-96d5-13a55c233e49.png)

This mega scatter plot gives us a reasonable idea about variables relationships.

One of the figures we may find interesting is the one between &#39;TotalBsmtSF&#39; and &#39;GrLiveArea&#39;. In this figure, we can see the dots drawing a linear line, which almost acts like a border. It totally makes sense that the majority of the dots stay below that line. Basement areas can be equal to the above-ground living area, but it is not expected a basement area bigger than the above-ground living area.

The plot concerning &#39;SalePrice&#39; and &#39;YearBuilt&#39; can also make us think. At the bottom of the &#39;dots cloud&#39;, we see what almost appears to be a shy exponential function (be creative). We can also see this same tendency in the upper limit of the &#39;dots cloud&#39;. Also, notice how the set of dots regarding the last years tend to stay above this limit.

**Data Preparation Methodology**

From the Feature Description file, we could estimate what we want to fill the &#39;nan&#39; values with, so the following points explain how I have filled each feature and their nan samples with:

- The feature &quot;Functional&quot;&#39;s nan values were filled with &quot;Typ&quot; string
- The feature &quot;Electrical&quot;&#39;s nan values were filled with &quot;SBrkr&quot; string
- The feature &quot;KitchenQual&quot;&#39;s nan values were filled with &quot;TA&quot; string
- The feature &quot;Exterior1st&quot;&#39;s nan values were filled with its mode value
- The feature &quot;Exterior2nd&quot;&#39;s nan values were filled with its mode value
- The feature &quot;SaleType&quot;&#39;s nan values were filled with its mode value
- The feature &quot;PoolQC&quot;&#39;s nan values were filled with &quot;None&quot; value
- The feature &quot;Alley&quot;&#39;s nan values were filled with &quot;None&quot; value
- The feature &quot;FirePlaceQu&quot;&#39;s nan values were filled with &quot;None&quot; value
- The feature &quot;Fence&quot;&#39;s nan values were filled with &quot;None&quot; value
- The feature &quot;MiscFeature&quot;&#39;s nan values were filled with &quot;None&quot; value
- The feature &quot;GarageArea&quot;&#39;s nan values were filled with the value zero (0)
- The feature &quot;GarageCars&quot;&#39;s nan values were filled with the value zero (0)
- The feature &quot;GarageType&quot;&#39;s nan values were filled with &quot;None&quot; value
- The feature &quot;GarageFinish&quot;&#39;s nan values were filled with &quot;None&quot; value
- The feature &quot;GarageQual&quot;&#39;s nan values were filled with &quot;None&quot; value
- The feature &quot;GarageCond&quot;&#39;s nan values were filled with &quot;None&quot; value
- The feature &quot;BsmtQual&quot;&#39;s nan values were filled with &quot;None&quot; value
- The feature &quot;BsmtCond&quot;&#39;s nan values were filled with &quot;None&quot; value
- The feature &quot;BsmtExposure&quot;&#39;s nan values were filled with &quot;None&quot; value
- The feature &quot;BsmtFinType1&quot;&#39;s nan values were filled with &quot;None&quot; value
- The feature &quot;BsmtFinType2&quot;&#39;s nan values were filled with &quot;None&quot; value

## **Log Transformation**

For those features that are highly skewed, they are normalized using Log transformation.

The following figures show how the log transformation affects the distribution of SalesPrice values. ![image](https://user-images.githubusercontent.com/16374886/123544647-e02afd00-d719-11eb-9c4a-ad5542394f91.png)

Fig. 10: Before Log transformation, the graphs shows how the data is skewed.

Fig. 11: After Log transformation, the graphs shows how the data is perfectly normalized. ![image](https://user-images.githubusercontent.com/16374886/123544718-2b451000-d71a-11eb-86bf-e6c8b317385a.png)

# **Baseline Models**

We divide the dataset in the ratio 66.67/33.33 into train/validation sets respectively. The model is trained on train dataset and further evaluated on the validation dataset. The evaluation metric used is Root Mean Square Error. The calculation formula for the same is given below:

![image](https://user-images.githubusercontent.com/16374886/123544725-39932c00-d71a-11eb-852b-adbe740141d8.png)

The table below represents a tabular representation of performances of numerous different regressors that I have implemented.

![image](https://user-images.githubusercontent.com/16374886/123544736-444dc100-d71a-11eb-9563-ccd1c972e3e3.png)

Fig. 12: Model Performance on the dataset without any feature Engineering or selection

# **Feature Engineering**

Apart from the initial features, I have created **four** more features that can helpn increase the performance of the model, they are as follows along with their formulas to calculate:

- **SqFtPerRoom =** train\_test[&quot;GrLivArea&quot;] / (train\_test[&quot;TotRmsAbvGrd&quot;] +

train\_test[&quot;FullBath&quot;] +

train\_test[&quot;HalfBath&quot;] +

train\_test[&quot;KitchenAbvGr&quot;])

- **Total\_Home\_Quality =** train\_test[&#39;OverallQual&#39;] + train\_test[&#39;OverallCond&#39;]

- **Total\_Bathrooms =** (train\_test[&#39;FullBath&#39;] + (0.5 \* train\_test[&#39;HalfBath&#39;]) +

train\_test[&#39;BsmtFullBath&#39;] + (0.5 \* train\_test[&#39;BsmtHalfBath&#39;]))

- **HighQualSF =** train\_test[&quot;1stFlrSF&quot;] + train\_test[&quot;2ndFlrSF&quot;]

# **Feature Selection**

After visualizing the baseline model performance without any feature selection or generation, nor any hyperparameter tuning, I decided to do some feature generation and generated four new features that could increase the performance of the model.

Now, in order to do feature selection, we need feature importance first to do see how the algorithm of the model is performing and which features are more important than the others.

I implemented a CatBoost Regressor and fitted it on the train dataset. Following this I could easily get the feature importance of each feature for the model. The barplot below shows a quantitative estimate of feature importance of top twenty features.

![image](https://user-images.githubusercontent.com/16374886/123544753-5596cd80-d71a-11eb-92da-4ebd6c35f770.png)

Fig. 13 - Feature Importance plot of top twenty features w.r.t the CatBoost Regressor.

Catboost comes with a great method: get\_feature\_importance. This method can be used to find important interactions among features. This is a huge advantage because it can give us insights about possible new features to create that can improve the performance.

![image](https://user-images.githubusercontent.com/16374886/123544764-61828f80-d71a-11eb-86d5-dfd1ccc41e6a.png)

Fig. 14 - Feature Interaction Levels

The table on the left shows the interaction level between two features that can be used to generate new features or to remove certain features if they are almost equivalent to each other.

#


#


# **Hyper Parameter Tuning And Results**

In order to achieve the best performance of the model, we need to perform HyperParameter tuning of various different hyperparameter options that are given to us by Cat Boost Regressor.

In order to do this, we do a Random Grid Search in order to get the best combination of parameters, the parameters that we tune are &quot;iterations&quot;, which are used to specify the number of epochs the model will iterate over the training set, &quot;learning\_rate&quot;, &quot;depth&quot; which implies the depth of decision tree that will be formed and &quot;l2\_leaf\_reg&quot; which is the L2 regularization term of the cost function.

The following are the best set of parameters obtained from the Grid Search

- Iterations - 6000
- Learning rate - 0.005
- Depth - 4
- L2\_leaf\_reg - 1
- Evaluation\_metric - RMSE (Root Mean Square Error)
- Verbose - 200
- Early Stopping Rounds - 200

After implementing these Hyperparameters with CatBoost Regressor along with the features generated through feature Engineering, we were successfully able to achieve an RMSE score of **0.1098** on the train set and **0.0553** on the test set. Thus, we are successfully able to build a model that accurately predicts the prices of houses provided we know its feature values.

# **References**

To conduct this project the following tools have been used :

● Python 3.8

● Pandas (Library) : [http://pandas.pydata.org/](http://pandas.pydata.org/)

● Numpy (Library) : [http://www.numpy.org/](http://www.numpy.org/)

● Scikit­learn (Library) : [http://scikit­learn.org/](about:blank)

● Matplotlib (Library) : [https://matplotlib.org/](https://matplotlib.org/)

● Seaborn (Library) : [https://seaborn.pydata.org/](https://seaborn.pydata.org/)

● XGBoost (Library) : [https://xgboost.readthedocs.io/en/latest/](https://xgboost.readthedocs.io/en/latest/)

● CatBoost (Library) : [https://catboost.ai/](https://catboost.ai/)

The techniques used to visualize and preprocess the data has been inspired from the book &quot; **Data Mining Concepts and Technique**&quot;.

The Machine Learning part has been greatly inspired by the **Machine Learning course**

**teached by Andrew Ng** of Coursera (https://www.coursera.org/course/ml) and the book &quot; **An introduction to Statistical Learning**&quot;.
