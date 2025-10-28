# Created to cross-match with Cathy's script
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

split <- initial_split(biomarker_clean, prop = 0.80, strata = group)
train  <- training(split)
test   <- testing(split)

train <- train %>%
  mutate(class = factor(group, levels = c("ASD", "TD")))
test  <- test  %>%
  mutate(class = factor(group, levels = c("ASD", "TD")))

trainX_df <- train %>% select(-group, -class)
testX_df  <- test  %>% select(-group, -class)

pp <- preProcess(trainX_df, method = "medianImpute")
trainX_df <- predict(pp, trainX_df)
testX_df  <- predict(pp, testX_df)

trainX <- as.matrix(trainX_df)
testX  <- as.matrix(testX_df)

trainY <- train$class
testY  <- test$class

# Simpler Panel Method: LASSO
cvfit <- cv.glmnet(
  x = trainX, y = trainY,
  family = "binomial",
  alpha = 1,                 
  nfolds = 10,
  type.measure = "auc"
)

lam <- cvfit$lambda.1se

coef_mat <- coef(cvfit, s = lam)
sel_idx  <- which(as.numeric(coef_mat) != 0)
sel_names <- rownames(coef_mat)[sel_idx]
lasso_panel <- setdiff(sel_names, "(Intercept)")
length_lasso <- length(lasso_panel)

phat <- as.numeric(predict(cvfit, newx = testX, s = lam, type = "response"))
roc_lasso <- roc(testY, phat, levels = c("TD","ASD"), direction = "<")
auc_lasso <- as.numeric(auc(roc_lasso))

pred_lbl <- ifelse(phat >= 0.5, "ASD", "TD")
cm_lasso <- caret::confusionMatrix(factor(pred_lbl, levels=levels(testY)), testY)

cat("\n[LASSO] Panel size:", length_lasso, "\n")
cat("[LASSO] AUC:", round(auc_lasso, 3), "\n")
print(cm_lasso$byClass[c("Sensitivity","Specificity")])
cat("[LASSO] First 10 proteins:\n"); print(head(lasso_panel, 10))

train_df <- data.frame(group = trainY, as.data.frame(trainX))
ctrl <- rfeControl(functions = rfFuncs, method = "repeatedcv", number = 5, repeats = 3, verbose = FALSE)

sizes <- c(5, 10, 15, 20, 30, 40, 50)

set.seed(42)
rfe_fit <- rfe(
  x = train_df[, -1],
  y = train_df$group,
  sizes = sizes,
  rfeControl = ctrl
)

opt_vars <- predictors(rfe_fit)     
length_rfe <- length(opt_vars)

rf_final <- randomForest(group ~ ., data = data.frame(group=trainY, as.data.frame(trainX[, opt_vars, drop=FALSE])), importance=TRUE)

rf_phat <- predict(rf_final, newdata = as.data.frame(testX[, opt_vars, drop=FALSE]), type = "prob")[, "ASD"]
roc_rfe <- roc(testY, rf_phat, levels = c("TD","ASD"), direction = "<")
auc_rfe <- as.numeric(auc(roc_rfe))

rf_pred <- predict(rf_final, newdata = as.data.frame(testX[, opt_vars, drop=FALSE]), type = "response")
cm_rfe <- caret::confusionMatrix(rf_pred, testY)

cat("\n[RFE+RF] Panel size:", length_rfe, "\n")
cat("[RFE+RF] AUC:", round(auc_rfe, 3), "\n")
print(cm_rfe$byClass[c("Sensitivity","Specificity")])
cat("[RFE+RF] First 10 proteins:\n"); print(head(opt_vars, 10))

# It does not match with Cathy's
# Cathy's ROC AUC
# Improved Random Forest: 0.171
# Simpler LASSO Panel: 0.083
# Lorretta's ROC AUC
# Improved Random Forest: 0.984
# Simpler LASSO Panel: 0.242