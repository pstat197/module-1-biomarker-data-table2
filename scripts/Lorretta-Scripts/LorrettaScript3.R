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
# source("scripts/inclass-analysis.R")
load("data/biomarker-clean.RData")
set.seed(101422) # same seed as inclass-analysis.R

# Task 4: Using LASSO Method to find a simpler panel that achieves comparable 
# classification accuracy and also benchmarks the results against the in-class
# analysis using ROC AUC

library(tidyverse)
library(tidymodels)
library(glmnet)
library(yardstick)
library(pROC)

# Load the preprocessed data
load('data/biomarker-clean.RData')

## LASSO REGULARIZATION FOR BIOMARKER SELECTION
################################################

# Prepare data for LASSO (exclude ADOS, only use protein biomarkers)
biomarker_lasso <- biomarker_clean %>%
  select(-ados) %>%
  mutate(class = (group == 'ASD')) %>%
  select(-group)

# Set seed for reproducibility
set.seed(101422)

# Partition into training and test sets (same split as in-class)
biomarker_split <- biomarker_lasso %>%
  initial_split(prop = 0.8)

train_data <- training(biomarker_split)
test_data <- testing(biomarker_split)

# Prepare matrix format for glmnet
x_train <- train_data %>% select(-class) %>% as.matrix()
y_train <- train_data %>% pull(class) %>% as.numeric()

x_test <- test_data %>% select(-class) %>% as.matrix()
y_test <- test_data %>% pull(class) %>% as.numeric()

## STEP 1: Cross-validation to find optimal lambda
###################################################

# Perform 10-fold cross-validation
set.seed(101422)
cv_lasso <- cv.glmnet(x_train, y_train, 
                      family = "binomial",
                      alpha = 1,  # alpha = 1 for LASSO
                      nfolds = 10,
                      type.measure = "auc")

# Plot cross-validation results
plot(cv_lasso)
title("LASSO Cross-Validation: AUC vs Lambda", line = 2.5)

# Extract optimal lambda values
lambda_min <- cv_lasso$lambda.min      # Lambda with minimum CV error
lambda_1se <- cv_lasso$lambda.1se      # Lambda within 1 SE of minimum

cat("Lambda min (minimum CV error):", lambda_min, "\n")
cat("Lambda 1se (parsimony rule):", lambda_1se, "\n")

## STEP 2: Fit LASSO models with optimal lambdas
#################################################

# Fit with lambda.min (best predictive performance)
lasso_min <- glmnet(x_train, y_train, 
                    family = "binomial",
                    alpha = 1,
                    lambda = lambda_min)

# Fit with lambda.1se (more parsimonious model)
lasso_1se <- glmnet(x_train, y_train, 
                    family = "binomial",
                    alpha = 1,
                    lambda = lambda_1se)

## STEP 3: Identify selected biomarkers
########################################

# Extract coefficients (exclude intercept)
coef_min <- coef(lasso_min)[-1, 1]
coef_1se <- coef(lasso_1se)[-1, 1]

# Identify non-zero coefficients
proteins_lasso_min <- names(coef_min[coef_min != 0])
proteins_lasso_1se <- names(coef_1se[coef_1se != 0])

cat("\n=== BIOMARKER SELECTION RESULTS ===\n")
cat("\nNumber of biomarkers selected (lambda.min):", length(proteins_lasso_min), "\n")
cat("Selected biomarkers:\n")
print(proteins_lasso_min)

cat("\nNumber of biomarkers selected (lambda.1se):", length(proteins_lasso_1se), "\n")
cat("Selected biomarkers:\n")
print(proteins_lasso_1se)

# Show coefficients with values
cat("\nCoefficients (lambda.1se - most parsimonious):\n")
coef_1se_nonzero <- coef_1se[coef_1se != 0]
print(sort(coef_1se_nonzero, decreasing = TRUE))

## STEP 4: Evaluate on test set
################################

# Define metrics
class_metrics <- metric_set(sensitivity, 
                            specificity, 
                            accuracy,
                            roc_auc)

# Predictions for lambda.min model
pred_min <- predict(lasso_min, newx = x_test, type = "response")[,1]
test_results_min <- test_data %>%
  mutate(pred = pred_min,
         pred_class = factor(pred > 0.5, levels = c(FALSE, TRUE)))

metrics_min <- test_results_min %>%
  class_metrics(estimate = pred_class,
                truth = factor(class),
                pred,
                event_level = 'second')

# Predictions for lambda.1se model
pred_1se <- predict(lasso_1se, newx = x_test, type = "response")[,1]
test_results_1se <- test_data %>%
  mutate(pred = pred_1se,
         pred_class = factor(pred > 0.5, levels = c(FALSE, TRUE)))

metrics_1se <- test_results_1se %>%
  class_metrics(estimate = pred_class,
                truth = factor(class),
                pred,
                event_level = 'second')

cat("\n=== TEST SET PERFORMANCE ===\n")
cat("\nLASSO with lambda.min (", length(proteins_lasso_min), " biomarkers):\n", sep="")
print(metrics_min)

cat("\nLASSO with lambda.1se (", length(proteins_lasso_1se), " biomarkers):\n", sep="")
print(metrics_1se)

## STEP 5: Compare with in-class analysis
##########################################

# Note: You'll need to run the in-class analysis to get these results
# This section shows how to compare

cat("\n=== COMPARISON SUMMARY ===\n")
cat("\nIn-class approach: Used multiple testing + random forest feature selection\n")
cat("LASSO approach: Automated feature selection via regularization\n")

# Create comparison table
comparison <- tibble(
  Method = c("LASSO (lambda.min)", 
             "LASSO (lambda.1se)", 
             "In-class (multiple testing + RF)"),
  `N Biomarkers` = c(length(proteins_lasso_min), 
                     length(proteins_lasso_1se), 
                     NA),  # Fill in from in-class results
  `Test ROC AUC` = c(
    metrics_min %>% filter(.metric == "roc_auc") %>% pull(.estimate),
    metrics_1se %>% filter(.metric == "roc_auc") %>% pull(.estimate),
    NA   # Fill in from in-class results
  ),
  `Test Accuracy` = c(
    metrics_min %>% filter(.metric == "accuracy") %>% pull(.estimate),
    metrics_1se %>% filter(.metric == "accuracy") %>% pull(.estimate),
    NA   # Fill in from in-class results
  )
)

print(comparison)

## STEP 6: Visualize ROC curves
################################

# ROC curve for lambda.1se (recommended model)
roc_lasso <- roc(test_results_1se$class, test_results_1se$pred)

plot(roc_lasso, 
     main = "ROC Curve: LASSO Model (lambda.1se)",
     col = "blue", 
     lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")
legend("bottomright", 
       legend = paste0("AUC = ", round(auc(roc_lasso), 3)),
       col = "blue", 
       lwd = 2)

## STEP 7: Regularization path visualization
#############################################

# Fit full regularization path
lasso_path <- glmnet(x_train, y_train, 
                     family = "binomial",
                     alpha = 1)

# Plot coefficient paths
plot(lasso_path, xvar = "lambda", label = TRUE)
abline(v = log(lambda_1se), lty = 2, col = "red")
abline(v = log(lambda_min), lty = 2, col = "blue")
title("LASSO Regularization Path")
legend("topright", 
       legend = c("lambda.1se", "lambda.min"),
       col = c("red", "blue"),
       lty = 2)
