library(tidyverse)
library(tidymodels)
library(randomForest)
library(modelr)
library(yardstick)

load("data/biomarker-clean.RData")

# Prepare training / test data (80/20 split)

set.seed(42)
split <- initial_split(biomarker_clean, prop = 0.8)
train <- training(split)
test  <- testing(split)

# Convert to binary class variable
train <- train %>% mutate(class = factor(if_else(group == "ASD", "ASD", "TD")))
test  <- test %>% mutate(class = factor(if_else(group == "ASD", "ASD", "TD")))

# LASSO logistic regression – Simpler panel

# Recipe (drop ados and group)
lasso_rec <- recipe(class ~ ., data = train %>% select(-ados, -group))

# Define model with L1 penalty
lasso_mod <- logistic_reg(
  mode = "classification",
  penalty = tune(),
  mixture = 1
) %>%
  set_engine("glmnet")

# Cross-validation folds
folds <- vfold_cv(train, v = 5, strata = class)

# Workflow
lasso_wf <- workflow() %>%
  add_recipe(lasso_rec) %>%
  add_model(lasso_mod)

# Create a regularization grid
lambda_grid <- grid_regular(penalty(range = c(-4, 0)), levels = 20)

# Tune model over grid
set.seed(42)
lasso_tune <- tune_grid(
  lasso_wf,
  resamples = folds,
  grid = lambda_grid,
  metrics = metric_set(roc_auc)
)

# Select best λ by AUC
best_lambda <- select_best(lasso_tune, metric = "roc_auc")

# Finalize workflow
lasso_final <- finalize_workflow(lasso_wf, best_lambda)

# Fit final LASSO model
lasso_fit <- fit(lasso_final, data = train)

# Extract selected (non-zero) coefficients = simpler panel

coef_df <- tidy(extract_fit_parsnip(lasso_fit)) %>%
  filter(term != "(Intercept)", estimate != 0)

lasso_genes <- coef_df %>% pull(term)
cat("Selected LASSO biomarker panel (", length(lasso_genes), " proteins):\n")
print(lasso_genes)

# Evaluate LASSO model on test set

lasso_results <- predict(lasso_fit, test, type = "prob") %>%
  bind_cols(predict(lasso_fit, test)) %>%
  bind_cols(test %>% select(class)) %>%
  rename(.pred_ASD = `.pred_ASD`, pred_class = .pred_class)

metrics_lasso <- metric_set(accuracy, sensitivity, specificity, roc_auc)

lasso_metrics <- metrics_lasso(
  data = lasso_results,
  truth = class,
  estimate = pred_class,
  .pred_ASD,
  event_level = "second"
)

cat("\nPerformance of Simpler LASSO Panel:\n")
print(lasso_metrics)

# Alternative model – Random Forest (potential improvement)

rf_model <- rand_forest(trees = 1000, mtry = 30, min_n = 5) %>%
  set_engine("ranger") %>%
  set_mode("classification")

rf_wf <- workflow() %>%
  add_recipe(recipe(class ~ ., data = train %>% select(-ados, -group))) %>%
  add_model(rf_model)

rf_fit <- fit(rf_wf, data = train)

rf_results <- predict(rf_fit, test, type = "prob") %>%
  bind_cols(predict(rf_fit, test)) %>%
  bind_cols(test %>% select(class)) %>%
  rename(.pred_ASD = `.pred_ASD`, pred_class = .pred_class)

rf_metrics <- metrics_lasso(
  data = rf_results,
  truth = class,
  estimate = pred_class,
  .pred_ASD,
  event_level = "second"
)

cat("\nPerformance of Random Forest Model:\n")
print(rf_metrics)

# Compare baseline vs new models

results_q4 <- bind_rows(
  lasso_metrics %>% mutate(Model = "Simpler LASSO panel"),
  rf_metrics %>% mutate(Model = "Improved Random Forest")
)

results_q4 %>%
  select(Model, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  arrange(desc(roc_auc)) %>%
  print()

# Visualize ROC AUC comparison

results_q4 %>%
  filter(.metric == "roc_auc") %>%
  ggplot(aes(x = Model, y = .estimate, fill = Model)) +
  geom_col() +
  geom_text(aes(label = sprintf("%.3f", .estimate)), vjust = -0.5, size = 4) +
  theme_minimal(base_size = 12) +
  labs(title = "Benchmark: Simpler vs. Improved Biomarker Panels (Question 4)",
       y = "ROC AUC", x = NULL) +
  theme(legend.position = "none")