---
title: "Will it be raining tomorrow in Australia?"
author: "Reza Dwi Utomo"
date: "24/02/2020"
output:
  html_document:
    highlight: zenburn
    number_sections: yes
    theme: flatly
    toc: yes
    toc_depth: 2
    toc_float:
      collapsed: yes
    df_print: paged
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

# Introduction {#intro}

This article aims to accomplish [Regression Model](https://algorit.ma/course/regression-models/) course at Algoritma. The dataset used is obtained from [Kaggle](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package), "Rain in Australia: Predict rain tomorrow in Australia". You could see the source code in my GitHub account [here](https://github.com/utomoreza/C1_LBB).

## Aim

The goal is to model red wine quality based on physicochemical tests.

## Objectives

1. To solve the final model equation

2. To output the statistical values (adjusted) R-squared, Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and MAE (Mean Absolute Error)

3. To examine the model including statistics and visualizations:

+ Assess linearity of model (parameters)
+ Assess serial independence of errors
+ Assess heteroscedasticity
+ Assess normality of residual distribution
+ Assess multicollinearity

4. To interpretate the model

5. To consider other factors, such as:

+ Are there any outliers?
+ Are there missing values?

6. To test the model using dataset test and discuss the results

## Structure

This article is arranged as follows.

1. [Introduction](#intro)
2. [Metadata](#meta)
3. [Preparation](#prep)
4. [Exploratory Data Analysis](#eda)
5. [Modelling](#model)
6. [Model Improvements](#modimprov)
7. [Results and Discussions](#resdis)
8. [Conclusions](#conc)

# Metadata {#meta}

## Context

Predict whether or not it will rain tomorrow by training a binary classification model on target `RainTomorrow`.

## Content

This dataset contains daily weather observations from numerous Australian weather stations. The target variable `RainTomorrow` means: Did it rain the next day? Yes or No.

Note: You should exclude the variable Risk-MM when training a binary classification model. Not excluding it will leak the answers to your model and reduce its predictability. Read more about it [here](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package/discussion/78316).

## Source & Acknowledgements

Observations were drawn from numerous weather stations. The daily observations are available from http://www.bom.gov.au/climate/data. Copyright Commonwealth of Australia 2010, Bureau of Meteorology.

Definitions adapted from http://www.bom.gov.au/climate/dwo/IDCJDW0000.shtml

This dataset is also available via the R package rattle.data and at https://rattle.togaware.com/weatherAUS.csv.
Package home page: http://rattle.togaware.com. Data source: http://www.bom.gov.au/climate/dwo/ and http://www.bom.gov.au/climate/data.

And to see some nice examples of how to use this data: https://togaware.com/onepager/

# Preparation {#prep}

```{r message=FALSE, echo=FALSE}
library(tidyverse) # for data wrangling
library(caret) # for confusion matrix
library(gtools) # for converting log of odds to probs
library(PerformanceAnalytics) # for pair plotting
library(car) # for executing VIF test
library(rsample) # for splitting dataset into train and test with controlled proportion
library(class) # for KNN
```

```{r}
weatherAU <- read.csv("weatherAUS.csv")
weatherAU
```

# Data Wrangling

```{r}
str(weatherAU)
```

Since we will compare the performance of Logistic Regression and KNN and KNN is not suitable to categorical predictor variables, we're going to drop those categorical ones. In addition, we should remove `RISK_MM` and `Date` features as well since `RISK_MM` have to be deleted as explained in the beginning and we do not consider time series forecasting so that we do not need `Date`.

```{r}
categorical <- vector()
for (i in 1:(length(weatherAU)-1)) {
  if (class(weatherAU[,i]) == "factor") {
    category <- colnames(weatherAU)[i]
    categorical <- c(categorical, category)
  }
}
weatherAU[,categorical] # to be removed
DF <- weatherAU %>% 
  select(-(categorical)) %>% 
  select(-RISK_MM)
head(DF)
ncol(DF)
```

Proportion of missing values

```{r}
colSums(is.na(DF))
cat("\n")
colSums(is.na(DF))/nrow(DF)
```

```{r}
summary(DF)
```

```{r warning=FALSE}
# DF %>% select(colnames(.)[1:17]) %>% 
#   pivot_longer(cols = colnames(.)[1:16], names_to = "Type", values_to = "Value") %>% 
#   mutate(Type = as.factor(Type)) %>% 
#   ggplot(aes(x = Type,
#              y = Value)) + 
#   coord_flip() + 
#   geom_boxplot(aes(fill = Type), show.legend = F)
```

Most predictors have outliers (we will deal with this outliers later in [a]()). So, the appropriate approach to fill missing values is by imputating them with median value as median is robust with outliers.

```{r}
for (i in 1:(length(DF)-1)) {
    DF[,i] <- replace_na(data = DF[,i], replace = median(DF[,i], na.rm = T))
}
anyNA(DF)
```

Nice. No missing value anymore.

# Exploratory Data Analysis

## Target Variable Proportion

```{r}
table(DF$RainTomorrow)
prop.table(table(DF$RainTomorrow))
```

Temporarily, we can consider the proportion is sufficiently balanced. If later we will find this is not balanced enough, we will tackle with this problem.

## Multicollinearity

```{r}
# library(GGally)

# GGally::ggcorr(DF %>% select(-RainTomorrow), label = T)
ggcorrplot::ggcorrplot(DF %>% select(-RainTomorrow) %>% cor(),
  hc.order = TRUE, type = "lower",
  lab = TRUE,
  digits = 1,
  ggtheme = ggplot2::theme_dark(),
)

# GGally::ggpairs(DF %>% select(-RainTomorrow))
# pairs.panels(DF %>% select(-RainTomorrow))
```

```{r}
DFcomb <- combn(colnames(DF[,1:16]), 2)
Alpha <- 0.05

for (i in 1:dim(DFcomb)[2]) {
  print(paste0("Cor test between ", DFcomb[1,i], " and ", DFcomb[2,i]))
  corTest <- cor.test(DF[,DFcomb[1,i]], DF[,DFcomb[2,i]])
  ifelse(corTest$p.value < Alpha, 
         print("Multicollinear detected"), 
         print("Safe! No Multicollinear"))
  cat("\n")
}
```

Surprisingly, all predictor variables indicate multicollinearity to each other. But, since we have no other variables except the ones provided, we can keep them all.

# Modelling and Predictions

## Splitting

```{r}
set.seed(1)
idx <- initial_split(DF, prop = 0.8, strata = RainTomorrow)
DF_train <- training(idx)
DF_test <- testing(idx)
prop.table(table(DF_train$RainTomorrow)) # Check train dataset proportion after split
prop.table(table(DF_test$RainTomorrow)) # Check test dataset proportion after split

# Split the predictors and the target of train dataset for KNN model usage
X_train <- DF_train[,-ncol(DF_train)]
y_train <- DF_train[,ncol(DF_train)]

# Split the predictors and the target of test dataset for KNN model usage
X_test <- DF_test[,-ncol(DF_test)]
y_test <- DF_test[,ncol(DF_test)]
```

## Create the Model

### Logistic Regression Model

```{r}
modelDF_log <- glm(formula = RainTomorrow ~ ., data = DF_train, family = "binomial")
summary(modelDF_log)
```

### KNN Data Pre-Processing - Scalling

```{r}
X_train.scaled <- scale(x = X_train)
X_test.scaled <- scale(x = X_test, 
                       center = attr(X_train.scaled, "scaled:center"),
                       scale = attr(X_train.scaled, "scaled:scale"))
```

## Predictions

### Logistic Regression Prediction

```{r}
predict_log <- predict(object = modelDF_log, newdata = DF_test, type = "response")
DF_test$ypred_prob <- predict_log
DF_test$ypred_label <- ifelse(DF_test$ypred_prob > 0.5, "Yes", "No")
```

### KNN Prediction

* Find optimum `K`

```{r}
K <- sqrt(nrow(X_train))
K
```

As the target variable has two classes only (`Yes` or `No`) meaning that it is an even number, we need an odd number of `K`. Therefore, with `K = 337`, it is enough for the prediction.

```{r}
predict_knn <- knn(train = X_train.scaled, test = X_test.scaled, cl = y_train, k = 3)
```

# Evaluation

## Confusion Matrix
```{r}
confusionMatrix(data = as.factor(DF_test$ypred_label), reference = y_test, positive = "Yes")
iconfusionMatrix(data = predict_knn, reference = y_test)
```

## ROC/AUC

```{r}

```

```{r}
library(ROCR)
df_ <- data.frame("prediction" = predict_log, 
                      "trueclass" = as.numeric(y_test == "Yes"))

df__roc <- prediction(df_$prediction, df_$trueclass)  
plot(performance(df__roc, "tpr", "fpr"))

auc <- ROCR::performance(prediction.obj = df__roc, "auc")
auc@y.values[[1]]
```















Prepare the performance indicators and all necessary functions.

```{r warning=FALSE, message=FALSE}
library(MLmetrics)
indicator <- function(model, y_pred, y_true) {
     adj.r.sq <- summary(model)$adj.r.squared
     mse <- MSE(y_pred, y_true)
     rmse <- RMSE(y_pred, y_true)
     mae <- MAE(y_pred, y_true)
     print(paste0("Adjusted R-squared: ", round(adj.r.sq, 4)))
     print(paste0("MSE: ", round(mse, 4)))
     print(paste0("RMSE: ", round(rmse, 4)))
     print(paste0("MAE: ", round(mae, 4)))
}

metrics <- function(y_pred, y_true){
     mse <- MSE(y_pred, y_true)
     rmse <- RMSE(y_pred, y_true)
     mae <- MAE(y_pred, y_true)
     print(paste0("MSE: ", round(mse, 6)))
     print(paste0("RMSE: ", round(rmse, 6)))
     print(paste0("MAE: ", round(mae, 6)))
     corPredAct <- cor(y_pred, y_true)
     print(paste0("Correlation: ", round(corPredAct, 6)))
     print(paste0("R^2 between y_pred & y_true: ", round(corPredAct^2, 6)))
}

CheckNormal <- function(model) {
     hist(model$residuals, breaks = 30)
     shaptest <- shapiro.test(model$residuals)
     print(shaptest)
     if (shaptest$p.value <= 0.05) {
          print("H0 rejected: the residuals are NOT distributed normally")
     } else {
          print("H0 failed to reject: the residuals ARE distributed normally")
     }
}

library(lmtest)
CheckHomos <- function(model){
     plot(model$fitted.values, model$residuals)
     abline(h = 0, col = "red")
     BP <- bptest(model)
     print(BP)
     if (BP$p.value <= 0.05) {
          print("H0 rejected: Error variance spreads INCONSTANTLY/generating patterns (Heteroscedasticity)")
     } else {
          print("H0 failed to reject: Error variance spreads CONSTANTLY (Homoscedasticity)")
     }
}
```

```{r}
redDF <- read.csv("winequality-red.csv", sep = ";")
redDF
```

```{r warning=FALSE, message=FALSE}
library(tidyverse)

redDF %>% is.na() %>% colSums()
```

Perfect. Since the used datasets are originally clean, we will not find any missing value. So, let's move on to explore the data.

# Exploratory Data Analysis {#eda}

In order to explore the dataset, we could use scatter plots, histograms, correlation value, and p-value.

```{r warning=FALSE, message=FALSE}
# library(psych)
# pairs.panels(redDF)

# library(ggcorrplot)
# ggcorrplot(redDF %>% cor(),
#   hc.order = TRUE, type = "lower",
#   lab = TRUE,
#   digits = 1,
#   ggtheme = ggplot2::theme_dark(),
# )

library(PerformanceAnalytics)
chart.Correlation(redDF, hist = T)
```

The preceding figure tells many things. But, in general, it shows four points: scatter plots between each variable, histograms of each variable, correlation values between each value, and p-values between each value against significance value of 0.05.

## Scatter plots {#scat}

Surprisingly, we found something interesting here. The scatter plots of between `quality` and each predictor variable form the same pattern that the target variable `quality` classifies the values into several classess, i.e. 3, 4, 5, 6, 7, and 8. To examine this case, we will go through to assess linear regression to model the data.

Moreover, there are several predictors which have strong relationship, e.g. between `fixed.acidity` and `citric.acid`. They are indicated by their tendency to have inclined or declined line. This case is discussed further in the Correlation values point below.

## Histograms {#hist}
     
Each predictor variable shows values distributed appropriately. However, the target variable exhibits poor distribution. This supports the above finding from scatter plots analysis. We could check the summary of such variable to make sure.

```{r collapse=TRUE}
summary(redDF$quality)
table(redDF$quality)
```

Unsuprisingly, `quality` does have classified values. **Based on this finding, it seems that linear regression is not suitable for this dataset. This is our initial hypothesis.**

## Correlation values {#corr}

The figure above shows that below relationships have a strong correlation.

 + Between `density` and `fixed.acidity` (0.67)
 + Between `fixed.acidity` and `citric.acid` (0.67)
 + Between `fixed.acidity` and `pH` (-0.68)
 + Between `free.sulfur.dioxide` and `total.sulfur.dioxide` (0.67)
     
Those perhaps indicate sufficiently high multicollinearity. We will highlight this issue and discuss it later in the [assumptions](#asum) section.

## P-values {#pval}

In addition, it is only `volatile.acidity` and `alcohol` which have the largest correlation value with `quality`. However, we also need to check the Pearson's correlation test based on the p-value. As seen in the figure above, the red stars in the upper triangle of the matrix indicate the significance. The more the stars exist, the more significant the relationship is. In order to be significant enough to the significance value (we use significance value (alpha) of 0.05), we need at least one star.

In this p-value analysis, we're only interested in considering the p-values of relationship between `quality` and each predictor variable. We can see that all variables have at least one star (meaning p-value less than pre-determined alpha (i.e. 0.05)), except `residual.sugar`. So, we won't consider such variable any longer.

# Modelling {#model}

## Splitting Train Datasets and Test Datasets

By using the dataset, we're going to split it up into 80% of data for train datasets and 20% of data for test datasets.

```{r}
set.seed(1)
sampleSize <- round(nrow(redDF)*0.8)
idx <- sample(seq_len(sampleSize), size = sampleSize)

X.train_red <- redDF[idx,]
X.test_red <- redDF[-idx,]

rownames(X.train_red) <- NULL
rownames(X.test_red) <- NULL
```

## Create the Model

As mentioned in the [exploratory data analysis](#pval), we will employ all predictor variables, except `residual.sugar`, for the model. Let's create linear model from those variables.

```{r}
model_red1 <- lm(quality ~ fixed.acidity + volatile.acidity + citric.acid + chlorides + free.sulfur.dioxide + total.sulfur.dioxide + density + pH + sulphates + alcohol, 
                 data = X.train_red)
summary(model_red1)
```

From the `summary()` function above, it can be seen that approximately a half number of all predictor variables exhibits insignificance. Furthermore, the adjusted R-squared also performs poor result. Before we tackle with this issue, we should check the assumptions of the model.

## Check the assumptions {#asum}

Since the linearity assumption has been discussed earlier in [this section](#corr), here, we're going to use three assumptions, i.e. normality, homoscedasticity, and multicollinearity.

### Normality
     
By employing normality assumption, we'd like to have the residuals of the predicted value to approach the normal distribution. We can check this by plotting the residuals and using Shapiro-Wilk normality test. For the latter, we expect to have p-value more than significance value (i.e. 0.05) so that the null hypothesis is failed to reject.

In the [Preparation](#prep) chapter, we have declared a function to carry out this task called `CheckNormal()`. So, let's use it.
     
```{r}
CheckNormal(model = model_red1)
```

Although it seems the figure above indicates that the residuals tend to gather around 0 number (i.e. approaching to have normal distribution), we are unable to immediately believe this. We also have to check the results of Shapiro-Wilk normality test. And unfortunately, in the case of normality, our model shows poor results. The p-value is so small that H0 is rejected, meaning that the residuals is **not** distributed normally. We don't want this.

### Homoscedasticity
 
In homoscedasticity aspect, we'd like to have residuals spreading constantly randomly, without generating any pattern. We have two approaches to examine this aspect, i.e. plotting the residuals vs the predicted values and performing the Breusch-Pagan test. As the function `CheckHomos()` to carry out this task has been declared already in [Preparation](#prep), we just need to use it.

```{r}
CheckHomos(model = model_red1)
```

As read above, the p-value is so small that null hypothesis is rejected. Moreover, the figure above also points out line-like patterns. This indeed states that the residuals generate patterns, meaning that heteroscedasticity exists. We don't want this.

### Multicollinearity
 
In multicollinearity factor, inside the model, we'd like to have each predictor variable **not** demonstrating strong relationship with each other. We could examine this factor by inspecting their VIF (Variance Inflation Factor) score. We expect to have VIF score not greater than 10. We can perform this task by using the function `vif()` from the `car` package.

```{r warning=FALSE, message=FALSE}
library(car)
vif(model_red1)
```

By reading their score above, we see that the only largest value is `fixed.acidity`, i.e. ± 6.9. Fortunately, such score is still lower than 10. Therefore, in case of multicollinearity, our model performs satisfactorily.

# Model Improvements {#modimprov}

As stated in the previous chapter, by using `summary()` function and checking the assumptions, the model performs poor results, except in case of multicollinearity. Thus, any improvement has to be executed to decrease its drawbacks.

## Check the Outliers

Firstly, let's check outliers of the dataset whether any high leverage high influence exist. We could use four plots here, i.e. Residuals vs Fitted, Normal Q-Q, Cook's Distance, and Residuals vs Leverage. For your information regarding those plots, you could read [here](https://data.library.virginia.edu/diagnostic-plots/).

```{r}
par(mfrow=c(2,2)) # Change the panel layout to 2 x 2 
lapply(c(1,2,4,5), # showing 4 types of plots
       function(x) plot(model_red1, 
                        which = x, 
                        # labels.id = 1:nrow(X.train_red),
                        cook.levels = c(0.05, 0.1))) %>% invisible()
```

From four figures above, we found there are some leverages with high influence, i.e. the observations with index 78, 202, 245, 274, and 1161. We're going to remove those rows.

```{r}
to.rm <- c(78,202,245,274,1161)
# X.train_red[to.rm,]
X.train_red <- X.train_red[-to.rm,]
rownames(X.train_red) <- NULL
```

After the outliers removed, a new model is generated, and also check its summary. 

```{r}
model_red2 <- lm(quality ~ fixed.acidity + volatile.acidity + citric.acid + chlorides + free.sulfur.dioxide + total.sulfur.dioxide + density + pH + sulphates + alcohol, 
                 data = X.train_red)
summary(model_red2)
```

It seems the new model performs more reliable. To make sure, let's check the adjusted R-squared values between two models.

```{r collapse=TRUE}
print("Adjusted R-squared for 1st model:")
ad.r.sq1 <- summary(model_red1)$adj.r.squared
ad.r.sq1
print("Adjusted R-squared for 2nd model:")
ad.r.sq2 <- summary(model_red2)$adj.r.squared
ad.r.sq2
print(paste0("The difference between both is ", round(ad.r.sq2-ad.r.sq1, 5)*100, "%"))
```

Well done. Adjusted R-squared increases by almost 2%. Now, we move on to try feature selection to improve the model.

## Feature Selection Implementation

We're going to employ step-wise algorithm for the feature selection method. We will use three directions of the algorithm, i.e. backward, forward, and both. First of all, we have to define the models for lower and upper threshold of the algorithm.

### Create two models as threshold for the step wise algorithm
     
```{r}
model_redAlc <- lm(quality ~ alcohol, data = X.train_red)
# summary(model_redAlc)
model_redAll <- lm(quality ~ ., data = X.train_red)
# summary(model_redAll)
```

Now, let's carry out three approaches of step-wise algorithm.

### Backward approach

```{r}
step(model_redAll, direction = "backward", trace = F)
```

```{r}
model.back_red <- lm(formula = quality ~ volatile.acidity + chlorides + free.sulfur.dioxide + 
    total.sulfur.dioxide + pH + sulphates + alcohol, data = X.train_red)
summary(model.back_red)
```

### Forward approach

```{r}
step(model_redAlc, scope = list(lower = model_redAlc, upper = model_redAll),
     direction = "forward",
     trace = F)
```

```{r}
model.forw_red <- lm(formula = quality ~ alcohol + volatile.acidity + sulphates + total.sulfur.dioxide + chlorides + pH + free.sulfur.dioxide, 
    data = X.train_red)
summary(model.forw_red)
```

### Both approach
  
```{r}
step(model_redAlc, scope = list(lower = model_redAlc, upper = model_redAll),
     direction = "both",
     trace = F)
```

```{r}
model.both_red <- lm(formula = quality ~ alcohol + volatile.acidity + sulphates + total.sulfur.dioxide + chlorides + pH + free.sulfur.dioxide, 
                     data = X.train_red)
summary(model.both_red)
```

All three approaches have been defined. Now, we're going to compare our all models so far by their adjusted R-squared.

```{r collapse=TRUE}
cat("Adjusted R-squared for 1st model:\n")
ad.r.sq1 <- summary(model_red1)$adj.r.squared
ad.r.sq1
cat("\nAdjusted R-squared for 2nd model:\n")
ad.r.sq2 <- summary(model_red2)$adj.r.squared
ad.r.sq2
cat("\nAdjusted R-squared for model using 'alcohol' variable only:\n")
ad.r.sqAlc <- summary(model_redAlc)$adj.r.squared
ad.r.sqAlc
cat("\nAdjusted R-squared for model using all variables:\n")
ad.r.sqAll <- summary(model_redAll)$adj.r.squared
ad.r.sqAll
cat("\nAdjusted R-squared for model with backward approach:\n")
ad.r.sqBack <- summary(model.back_red)$adj.r.squared
ad.r.sqBack
cat("\nAdjusted R-squared for model with forward approach:\n")
ad.r.sqForw <- summary(model.forw_red)$adj.r.squared
ad.r.sqForw
cat("\nAdjusted R-squared for model with both approach:\n")
ad.r.sqBoth <- summary(model.both_red)$adj.r.squared
ad.r.sqBoth
```

Evidently, after we have performed feature selection, we don't obtain the model with much higher performance. Instead, the best model so far is achieved not from such selection, but from manually including all available predictor variables. **Hence, from now on, the best model used will be the one with all predictor variables, i.e. `model_redAll`**

# Results and Discussions {#resdis}

In this chapter, we're going to discuss the best model so far and use it to predict the test dataset. Firstly, we should interpret the selected model. Subsequently, the performance of the model is discussed and the predictions will be carried out later.

## Model Interpretation

The selected model is the one with all available preditor variables. We defined it as `model_redAll`. It consist of the following equation:

$\hat{Y} = \beta_0+\beta_1X_1+\beta_2X_2+\beta_3X_3+\beta_4X_4+\beta_5X_5+\beta_6X_6+\beta_7X_7+\beta_8X_8+\beta_9X_9+\beta_{10} X_{10}+\beta_{11}X_{11}$ 

where the following are values from the $\beta_0$ to $\beta_{11}$ and from $X_1$ to $X_{11}$:

```{r}
model_redAll$coefficients
```

From the equation above, we can interpret that the line starts from the Cartesian coordinate of (0, 37.87), as pointed by the intercept. Furthermore, along with the increase of any $X_i$, the related $\beta_i$ will adjust the line according to both values.

For example and for simplicity, if we were to have $X_{alcohol} = 1$, then the y coordinate (or so-called the predicted value) would be:
$\hat{Y} = \beta_0 + \beta_{alcohol}.X_{alcohol} = 37.87 + 0.27*1 = 38.14$

## Check the Performances

Here, we're going to check the performances of the chosen model. The metrics used are Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE). We just need the `indicator()` function defined in the beginning.

```{r collapse=TRUE}
indicator(model = model_redAll, y_pred = model_redAll$fitted.values, y_true = X.train_red$quality)
```

With performances as shown above, we will compare them to those of the prediction using test dataset.

## Predictions

Here, we're going to compare the performances of prediction using train dataset to those of using test dataset. The metrics used are MSE, RMSE, MAE, correlation between `y_pred` and `y_true`, and R-squared between `y_pred` and `y_true`. The plots between `y_pred` and `y_true` for each train dataset and test dataset also will be shown.

```{r collapse=TRUE}
cat("Performance using train dataset:\n")
metrics(y_pred = model_redAll$fitted.values, y_true = X.train_red$quality)

redPredict.back <- predict(model_redAll, newdata = X.test_red)
cat("\nPerformances using test dataset:\n")
metrics(y_pred = redPredict.back, y_true = X.test_red$quality)
```
```{r}
redFitted.back <- data.frame(qualityPred = model.back_red$fitted.values,
                                qualityAct = X.train_red$quality)
ggplot(redFitted.back, aes(x = qualityPred,
                       y = qualityAct)) +
     geom_point(aes(color = as.factor(qualityAct)), show.legend = F) +
     geom_smooth(method = "lm", se = F) +
     labs(title = "Predicted vs Actual Values Using Train Dataset",
          x = "Predicted quality",
          y = "Actual quality")
```
```{r}
redPredict.backDF <- data.frame(qualityPred = redPredict.back,
                                qualityAct = X.test_red$quality)
ggplot(redPredict.backDF, aes(x = qualityPred,
                       y = qualityAct)) +
     geom_point(aes(color = as.factor(qualityAct)), show.legend = F) +
     geom_smooth(method = "lm", se = F) +
     labs(title = "Predicted vs Actual Values Using Test Dataset",
          x = "Predicted quality",
          y = "Actual quality")
```

There are several points we can infer from the performance results above:

1. The model overfits the train dataset so that it performs poor when using test dataset.
2. As the model has defective results, it is unable to satisfactorily predict the target variable.
3. The plots verify poin number 2 that the model in fact is ineffective to predict the target variable. 

# Conclusions {#conc}

We have finished this article. Below are the points we can conclude from this article:

* A linear model has been created. The target variable is `quality`, whereas the attributes of physicochemical tests are as the predictor variables.

* The selected model is the one with all available variables. So, the model equation is as follows:

$\hat{Y} = \beta_0+\beta_1X_1+\beta_2X_2+\beta_3X_3+\beta_4X_4+\beta_5X_5+\beta_6X_6+\beta_7X_7+\beta_8X_8+\beta_9X_9+\beta_{10} X_{10}+\beta_{11}X_{11}$ 

* The statistical values (adjusted) R-squared, Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE) of the selected model have been calculated.

```{r}
indicator(model = model_redAll, y_pred = model_redAll$fitted.values, y_true = X.train_red$quality)
```

* The selected model has been examined and improved using statistics tests (the assumptions and feature selection) and visualizations in [Modelling](#model) and [Model Improvements](#modimprov) chapters.

* The selected model has been interpretated in [Results and Discussions](#resdis) chapter.

* The selected model has been tested using test dataset test and discussed in [Results and Discussions](#resdis) chapter.

* The selected model performs ineffective in modelling the train dataset. The best adjusted R-squared value produced is only at 0.3832.

* As the selected model show poor performances, it also demonstrates deficient results when predicting the test dataset.

* As stated earlier in [histogram](#hist) and [scatter plots](#scat) sections that it is found an initial hypothesis that the target variable has classified values instead of continuous values so it seems the linear regression is not suitable with the dataset, such hypothesis has been proven. All results and discussions in this article verify it.

* As mentioned above, therefore, it can be concluded that for the type of this dataset, **in particular a target variable with classified values, it is not recommended to model the data using linear regression.**

* For future study, other algorithms will be applied to model this wine quality dataset.