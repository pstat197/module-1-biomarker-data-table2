library(tidyverse)
library(tidymodels)
library(randomForest)
library(modelr)
library(yardstick)
library(dplyr)
library(caret)
library(glmnet)
library(pROC)
source("scripts/preprocessing.R")
set.seed(42)

# Simpler Panel: LASSO
# Code based off of Week 4 Slides: LASSO Regularization

# read in data
biomarker <- biomarker_clean %>% 
  select(-ados) %>% 
  mutate(across(-group, ~scale(log(.x))[,1]),
         class = as.numeric(group == 'ASD'))

# partition
partitions <- biomarker %>% 
  initial_split(prop = 0.8)

x_train <- training(partitions) %>% 
  select(-group, -class) %>% 
  as.matrix()

y_train <- training(partitions) %>% 
  pull(class)

# removing missing values to use glmnet
pp <- preProcess(x_train, method = "medianImpute")
x_train <- predict(pp, x_train)
y_train <- predict(pp, y_train)

x_train <- as.matrix(x_train)
y_train <- as.matrix(y_train)

# multiple partitioning for lambda selection
cv_out <- cv.glmnet(x_train,
                    y_train,
                    family = 'binomial',
                    nfolds = 5,
                    type.measure = 'deviance')

cvout_df <- tidy(cv_out)

# LASSO estimates
fit <- glmnet(x_train, y_train, family = 'binomial')
fit_df <- tidy(fit)

# Create test set
x_test <- testing(partitions) %>%
  select(-group, -class) %>%
  as.matrix()
y_test <- testing(partitions) %>%
  pull(class)

# Impute missing values in test data
x_test <- predict(pp, x_test)
x_test <- as.matrix(x_test)

# Predict probabilities for ASD (class 1)
best_lambda <- cv_out$lambda.min
pred_probs <- predict(cv_out, newx = x_test, s = best_lambda, type = "response")

# ROC & AUC
roc_obj <- roc(y_test, as.numeric(pred_probs))
auc_value <- auc(roc_obj)

cat("ROC AUC:", round(auc_value, 3), "\n")

# Attempting to refit
best_lambda <- cv_out$lambda.1se
lasso_probs <- as.numeric(predict(cv_out, newx = x_test, s = best_lambda,
                                  type = "response"))
lasso_pred <- factor(lasso_probs > 0.5, labels = c("TD", "ASD"))
# Fix error