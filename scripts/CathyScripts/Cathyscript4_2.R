library(tidyverse)
library(tidymodels)
library(randomForest)
library(modelr)
library(yardstick)  

load("data/biomarker-clean.RData")  # loads biomarker_clean

# Prepare training/test sets (80/20 split)

set.seed(42)
split <- initial_split(biomarker_clean, prop = 0.8)
train <- training(split)
test  <- testing(split)

train <- train %>% mutate(class = factor(if_else(group == "ASD", "ASD", "TD")))
test  <- test  %>% mutate(class = factor(if_else(group == "ASD", "ASD", "TD")))

train$class <- fct_relevel(train$class, "TD", "ASD")
test$class  <- fct_relevel(test$class,  "TD", "ASD")

# LASSO Logistic Regression: Simpler panel

lasso_rec <- recipe(class ~ ., data = train %>% select(-ados, -group))

lasso_mod <- logistic_reg(
  mode = "classification",
  penalty = tune(), 
  mixture = 1        
) %>%
  set_engine("glmnet")

folds <- vfold_cv(train, v = 5, strata = class)

lasso_wf <- workflow() %>%
  add_recipe(lasso_rec) %>%
  add_model(lasso_mod)

lambda_grid <- grid_regular(penalty(range = c(-4, 0)), levels = 20)

set.seed(42)
lasso_tune <- tune_grid(
  lasso_wf,
  resamples = folds,
  grid = lambda_grid,
  metrics = metric_set(roc_auc)
)

best_lambda <- select_best(lasso_tune, metric = "roc_auc")
lasso_final <- finalize_workflow(lasso_wf, best_lambda)
lasso_fit <- fit(lasso_final, data = train)

# Extract nonzero coefficients

coef_df <- tidy(extract_fit_parsnip(lasso_fit)) %>%
  filter(term != "(Intercept)", estimate != 0)
lasso_genes <- coef_df$term
cat("Selected LASSO biomarker panel (", length(lasso_genes), "proteins):\n")
print(lasso_genes)

# Evaluate LASSO model on test set

lasso_results <- predict(lasso_fit, test, type = "prob") %>%
  bind_cols(predict(lasso_fit, test)) %>%
  bind_cols(test %>% select(class)) %>%
  rename(.pred_ASD = `.pred_ASD`, pred_class = .pred_class)

# Yardstick metrics to avoid conflict
metrics_lasso <- yardstick::metric_set(
  yardstick::accuracy,
  yardstick::sensitivity,
  yardstick::specificity,
  yardstick::roc_auc
)

lasso_metrics <- metrics_lasso(
  data = lasso_results,
  truth = class,
  estimate = pred_class,
  .pred_ASD,
  event_level = "second"
)

cat("\nPerformance of Simpler LASSO Panel:\n")
print(lasso_metrics)

# Random Forest: Improved model

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

lasso_auc <- yardstick::roc_auc(lasso_results, truth = class, .pred_ASD, event_level = "second")
rf_auc    <- yardstick::roc_auc(rf_results, truth = class, .pred_ASD, event_level = "second")

results_q4 <- tibble(
  Model   = c("Simpler LASSO panel", "Improved Random Forest"),
  roc_auc = c(lasso_auc$.estimate, rf_auc$.estimate)
)

cat("\nComparison of ROC AUC:\n")
print(results_q4)


p1 <- ggplot(results_q4, aes(x = Model, y = roc_auc, fill = Model)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = sprintf("%.3f", roc_auc)), vjust = -0.5, size = 4) +
  theme_minimal(base_size = 12) +
  labs(title = "Benchmark: Simpler vs. Improved Biomarker Panels",
       y = "ROC AUC", x = NULL) +
  theme(legend.position = "none")

print(p1)

# Plot Full ROC Curves for both models

lasso_roc <- yardstick::roc_curve(lasso_results, truth = class, .pred_ASD, event_level = "second") %>%
  mutate(Model = "Simpler LASSO panel")

rf_roc <- yardstick::roc_curve(rf_results, truth = class, .pred_ASD, event_level = "second") %>%
  mutate(Model = "Improved Random Forest")

roc_df <- bind_rows(lasso_roc, rf_roc)

p2 <- ggplot(roc_df, aes(x = 1 - specificity, y = sensitivity, color = Model)) +
  geom_line(size = 1.2) +
  geom_abline(linetype = "dashed", color = "black") +
  theme_minimal(base_size = 13) +
  labs(title = "ROC Curves: LASSO vs. Random Forest",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  coord_equal() +
  scale_color_manual(values = c("red", "blue"))

print(p2)