# ============================================================
# MODULE 1 – BIOMARKER (Question 4)
# Goal: Use LASSO logistic regression to identify a simpler biomarker panel
# ============================================================

# --- Load libraries ---
library(tidyverse)
library(tidymodels)
library(glmnet)
library(yardstick)
library(pROC)

# --- Load preprocessed data ---
load("data/biomarker-clean.RData")  # loads biomarker_clean
set.seed(101422)

# ============================================================
# STEP 1–3: DATA PREP + LASSO MODEL (already done above)
# ============================================================

# Prepare data for LASSO (drop ADOS, convert group to binary)
biomarker_lasso <- biomarker_clean %>%
  select(-ados) %>%
  mutate(class = (group == "ASD")) %>%
  select(-group)

# Split into train/test
biomarker_split <- initial_split(biomarker_lasso, prop = 0.8)
train_data <- training(biomarker_split)
test_data  <- testing(biomarker_split)

# Convert class to factor (TD = negative, ASD = positive)
train_data <- train_data %>%
  mutate(class = factor(class, levels = c(FALSE, TRUE),
                        labels = c("TD", "ASD")))
test_data <- test_data %>%
  mutate(class = factor(class, levels = c(FALSE, TRUE),
                        labels = c("TD", "ASD")))

# Prepare matrices for glmnet
x_train <- train_data %>% select(-class) %>% as.matrix()
y_train <- train_data %>% pull(class) %>% as.numeric() - 1
x_test  <- test_data %>% select(-class) %>% as.matrix()

# ============================================================
# STEP 4: LASSO MODEL FITTING AND EVALUATION
# ============================================================

# --- Cross-validation to find optimal lambda ---
cv_lasso <- cv.glmnet(
  x_train, y_train,
  family = "binomial",
  alpha = 1,
  nfolds = 10,
  type.measure = "auc"
)
plot(cv_lasso)
title("LASSO Cross-Validation: AUC vs Lambda", line = 2.5)

lambda_min <- cv_lasso$lambda.min
lambda_1se <- cv_lasso$lambda.1se
cat("Lambda min:", lambda_min, "\nLambda 1se:", lambda_1se, "\n")

# --- Fit final models ---
lasso_min <- glmnet(x_train, y_train, family = "binomial", alpha = 1, lambda = lambda_min)
lasso_1se <- glmnet(x_train, y_train, family = "binomial", alpha = 1, lambda = lambda_1se)

# --- Extract selected biomarkers ---
coef_min <- coef(lasso_min)[-1, 1]
coef_1se <- coef(lasso_1se)[-1, 1]

proteins_lasso_min <- names(coef_min[coef_min != 0])
proteins_lasso_1se <- names(coef_1se[coef_1se != 0])

# Aliases for easier downstream use
proteins_min <- proteins_lasso_min
proteins_1se <- proteins_lasso_1se

cat("\n=== SELECTED BIOMARKERS ===\n")
cat("λ.min:", length(proteins_min), "proteins\n")
print(proteins_min)
cat("\nλ.1se:", length(proteins_1se), "proteins\n")
print(proteins_1se)

# --- Define class metrics ---
class_metrics <- yardstick::metric_set(
  yardstick::sensitivity,
  yardstick::specificity,
  yardstick::accuracy
)

# ---- Evaluate λ.min ----
pred_min <- predict(lasso_min, newx = x_test, type = "response")[, 1]
test_results_min <- test_data %>%
  mutate(
    .pred_ASD = pred_min,
    pred_class = factor(if_else(.pred_ASD > 0.5, "ASD", "TD"),
                        levels = c("TD", "ASD"))
  )

metrics_min_class <- class_metrics(
  data = test_results_min,
  truth = class,
  estimate = pred_class,
  event_level = "second"
)
metrics_min_auc <- yardstick::roc_auc(
  data = test_results_min,
  truth = class,
  .pred_ASD,
  event_level = "second"
)

# ---- Evaluate λ.1se ----
pred_1se <- predict(lasso_1se, newx = x_test, type = "response")[, 1]
test_results_1se <- test_data %>%
  mutate(
    .pred_ASD = pred_1se,
    pred_class = factor(if_else(.pred_ASD > 0.5, "ASD", "TD"),
                        levels = c("TD", "ASD"))
  )

metrics_1se_class <- class_metrics(
  data = test_results_1se,
  truth = class,
  estimate = pred_class,
  event_level = "second"
)
metrics_1se_auc <- yardstick::roc_auc(
  data = test_results_1se,
  truth = class,
  .pred_ASD,
  event_level = "second"
)

# Merge metrics for easy use
metrics_min <- bind_rows(metrics_min_class, metrics_min_auc)
metrics_1se <- bind_rows(metrics_1se_class, metrics_1se_auc)

cat("\n=== TEST SET PERFORMANCE ===\n")
cat("\nLASSO (λ.min):\n")
print(metrics_min)
cat("\nLASSO (λ.1se):\n")
print(metrics_1se)

# ============================================================
# STEP 5: COMPARISON SUMMARY
# ============================================================

cat("\n=== COMPARISON SUMMARY ===\n")
cat("\nIn-class approach: Used multiple testing + random forest feature selection\n")
cat("LASSO approach: Automated feature selection via regularization\n")

comparison <- tibble(
  Method = c("LASSO (λ.min)",
             "LASSO (λ.1se)",
             "In-class (multiple testing + RF)"),
  `N Biomarkers` = c(length(proteins_min),
                     length(proteins_1se),
                     NA),
  `Test ROC AUC` = c(
    metrics_min %>% filter(.metric == "roc_auc") %>% pull(.estimate),
    metrics_1se %>% filter(.metric == "roc_auc") %>% pull(.estimate),
    NA
  ),
  `Test Accuracy` = c(
    metrics_min %>% filter(.metric == "accuracy") %>% pull(.estimate),
    metrics_1se %>% filter(.metric == "accuracy") %>% pull(.estimate),
    NA
  )
)

print(comparison)

# ============================================================
# STEP 6: VISUALIZE ROC CURVES
# ============================================================

roc_lasso_min <- roc(test_results_min$class, test_results_min$.pred_ASD)
roc_lasso_1se <- roc(test_results_1se$class, test_results_1se$.pred_ASD)

plot(roc_lasso_min,
     main = "ROC Curves: LASSO Models (Test Set)",
     col = "blue", lwd = 2)
lines(roc_lasso_1se, col = "red", lwd = 2, lty = 2)
abline(a = 0, b = 1, lty = 3, col = "gray")
legend("bottomright",
       legend = c(paste0("λ.min AUC = ", round(auc(roc_lasso_min), 3)),
                  paste0("λ.1se AUC = ", round(auc(roc_lasso_1se), 3))),
       col = c("blue", "red"), lty = c(1, 2), lwd = 2)

# ============================================================
# STEP 7: REGULARIZATION PATH VISUALIZATION
# ============================================================

lasso_path <- glmnet(x_train, y_train, family = "binomial", alpha = 1)
plot(lasso_path, xvar = "lambda", label = TRUE)
abline(v = log(lambda_1se), lty = 2, col = "red")
abline(v = log(lambda_min), lty = 2, col = "blue")
title("LASSO Regularization Path")
legend("topright",
       legend = c("λ.1se", "λ.min"),
       col = c("red", "blue"),
       lty = 2)
