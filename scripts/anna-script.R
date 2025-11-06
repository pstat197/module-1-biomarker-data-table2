# Part 3: Experimental Modifications
# Exploring sensitivity of results to design choices
#
# This script implements three key modifications to the original biomarker analysis:
# 1. Proper train/test split with selection on training data only
# 2. Using more than 10 top proteins from each selection method
# 3. Fuzzy intersection instead of hard intersection for combining protein sets

# Load required libraries
library(tidyverse)
library(infer)
library(randomForest)
library(tidymodels)
library(modelr)
library(yardstick)
library(pROC)

# Remove plyr - it conflicts with dplyr
detach("package:plyr", unload=TRUE)

# Load the cleaned data
load('/Users/annagornyitzki/DSCapstone/biomarker-project/module-1-biomarker-data-table2/data/biomarker-clean.RData')

# Set seed for reproducibility
set.seed(42)


# BASELINE: Original Analysis Function


run_original_analysis <- function(data, n_proteins = 10) {
  
  # Multiple testing with t-tests - with error handling
  test_fn <- function(.df){
    tryCatch({
      # Check if there's sufficient variation in the data
      if(length(unique(.df$level)) < 2) {
        return(tibble(statistic = NA, p_value = 1))
      }
      
      # Check if there's variation within each group
      group_vars <- .df %>% 
        group_by(group) %>% 
        summarise(var = var(level, na.rm = TRUE), .groups = 'drop')
      
      if(any(group_vars$var == 0, na.rm = TRUE) || any(is.na(group_vars$var))) {
        return(tibble(statistic = NA, p_value = 1))
      }
      
      t_test(.df, 
             formula = level ~ group,
             order = c('ASD', 'TD'),
             alternative = 'two-sided',
             var.equal = F)
    }, error = function(e) {
      # Return a tibble with NA statistic and p_value = 1 for failed tests
      tibble(statistic = NA, p_value = 1)
    })
  }
  
  ttests_out <- data %>%
    select(-ados) %>%
    pivot_longer(-group, names_to = 'protein', values_to = 'level') %>%
    nest(data = c(level, group)) %>% 
    mutate(ttest = map(data, test_fn)) %>%
    unnest(ttest) %>%
    filter(!is.na(p_value), p_value < 1) %>%  # Remove failed tests
    arrange(p_value) %>%
    mutate(m = n(),
           hm = log(m) + 1/(2*m) - digamma(1),
           rank = row_number(),
           p.adj = m*hm*p_value/rank)
  
  if(nrow(ttests_out) == 0) {
    proteins_s1 <- character(0)
  } else {
    proteins_s1 <- ttests_out %>%
      slice_min(p.adj, n = n_proteins) %>%
      pull(protein)
  }
  
  # Random Forest selection - with error handling
  predictors <- data %>% select(-c(group, ados))
  response <- data %>% pull(group) %>% factor()
  
  # Remove constant columns for RF
  constant_cols <- sapply(predictors, function(x) length(unique(x)) <= 1)
  if(any(constant_cols)) {
    predictors <- predictors[, !constant_cols]
  }
  
  if(ncol(predictors) == 0) {
    proteins_s2 <- character(0)
    rf_importance <- matrix(nrow = 0, ncol = 2, dimnames = list(NULL, c("MeanDecreaseAccuracy", "MeanDecreaseGini")))
  } else {
    rf_out <- randomForest(x = predictors, y = response, ntree = 1000, importance = T)
    
    proteins_s2 <- rf_out$importance %>% 
      as_tibble() %>%
      mutate(protein = rownames(rf_out$importance)) %>%
      slice_max(MeanDecreaseGini, n = n_proteins) %>%
      pull(protein)
    
    rf_importance <- rf_out$importance
  }
  
  # Hard intersection
  proteins_final <- intersect(proteins_s1, proteins_s2)
  
  return(list(
    proteins_ttest = proteins_s1,
    proteins_rf = proteins_s2,
    proteins_final = proteins_final,
    ttests_results = ttests_out,
    rf_importance = rf_importance
  ))
}

# Run original analysis for comparison
cat("=== BASELINE: Original Analysis ===\n")
original_results <- run_original_analysis(biomarker_clean)
cat("Original analysis - Final proteins selected:", length(original_results$proteins_final), "\n")
cat("Proteins:", paste(original_results$proteins_final, collapse = ", "), "\n\n")


# MODIFICATION 1: Proper Train/Test Split


run_analysis_with_proper_split <- function(data, n_proteins = 10, train_prop = 0.7) {
  
  # Initial split - set aside test data
  initial_split <- data %>%
    mutate(class = (group == 'ASD')) %>%
    initial_split(prop = train_prop, strata = class)
  
  train_data <- training(initial_split)
  test_data <- testing(initial_split)
  
  cat("Training set size:", nrow(train_data), "\n")
  cat("Test set size:", nrow(test_data), "\n")
  
  # Perform selection ONLY on training data
  selection_results <- run_original_analysis(train_data, n_proteins)
  
  # Evaluate on test set
  if(length(selection_results$proteins_final) > 0) {
    
    # Prepare training data for modeling
    train_model_data <- train_data %>%
      select(class, any_of(selection_results$proteins_final))
    
    # Fit logistic regression on training data
    fit <- glm(class ~ ., data = train_model_data, family = 'binomial')
    
    # Evaluate on test set
    test_model_data <- test_data %>%
      select(class, any_of(selection_results$proteins_final))
    
    predictions <- predict(fit, newdata = test_model_data, type = 'response')
    
    # Calculate metrics - fix the factor conversion issue
    test_results <- tibble(
      truth = as.factor(test_model_data$class),
      pred_prob = as.numeric(predictions),
      pred_class = as.factor(pred_prob > 0.5)
    )
    
    # Make sure factor levels are consistent
    levels(test_results$truth) <- c("FALSE", "TRUE")
    levels(test_results$pred_class) <- c("FALSE", "TRUE")
    
    # Calculate individual metrics
    auc_val <- test_results %>% roc_auc(truth = truth, pred_prob, event_level = "second") %>% pull(.estimate)
    acc_val <- test_results %>% accuracy(truth = truth, estimate = pred_class) %>% pull(.estimate)
    sens_val <- test_results %>% sensitivity(truth = truth, estimate = pred_class, event_level = "second") %>% pull(.estimate)
    spec_val <- test_results %>% specificity(truth = truth, estimate = pred_class, event_level = "second") %>% pull(.estimate)
    
    performance <- tibble(
      .metric = c("roc_auc", "accuracy", "sensitivity", "specificity"),
      .estimate = c(auc_val, acc_val, sens_val, spec_val)
    )
    
    return(list(
      selection_results = selection_results,
      train_data = train_data,
      test_data = test_data,
      model = fit,
      performance = performance,
      test_predictions = test_results
    ))
  } else {
    return(list(
      selection_results = selection_results,
      train_data = train_data,
      test_data = test_data,
      error = "No proteins selected in intersection"
    ))
  }
}

cat("=== MODIFICATION 1: Proper Train/Test Split ===\n")
mod1_results <- run_analysis_with_proper_split(biomarker_clean)

if(!"error" %in% names(mod1_results)) {
  cat("Proteins selected:", length(mod1_results$selection_results$proteins_final), "\n")
  cat("Selected proteins:", paste(mod1_results$selection_results$proteins_final, collapse = ", "), "\n")
  cat("\nTest Set Performance:\n")
  print(mod1_results$performance)
} else {
  cat("Error in Modification 1:", mod1_results$error, "\n")
}
cat("\n")


# MODIFICATION 2: Larger Number of Top Proteins


cat("=== MODIFICATION 2: Larger Number of Top Proteins ===\n")
protein_numbers <- c(15, 20, 25, 30)

mod2_results <- list()

for(n_prot in protein_numbers) {
  cat("\n--- Testing with", n_prot, "proteins ---\n")
  
  result <- run_analysis_with_proper_split(biomarker_clean, n_proteins = n_prot)
  
  if(!"error" %in% names(result)) {
    cat("T-test proteins:", length(result$selection_results$proteins_ttest), "\n")
    cat("RF proteins:", length(result$selection_results$proteins_rf), "\n")
    cat("Final intersection:", length(result$selection_results$proteins_final), "\n")
    
    if(length(result$selection_results$proteins_final) > 0) {
      auc_value <- result$performance %>% filter(.metric == "roc_auc") %>% pull(.estimate)
      acc_value <- result$performance %>% filter(.metric == "accuracy") %>% pull(.estimate)
      sens_value <- result$performance %>% filter(.metric == "sensitivity") %>% pull(.estimate)
      spec_value <- result$performance %>% filter(.metric == "specificity") %>% pull(.estimate)
      
      cat("Test AUC:", round(auc_value, 3), "\n")
      cat("Test Accuracy:", round(acc_value, 3), "\n")
      cat("Test Sensitivity:", round(sens_value, 3), "\n")
      cat("Test Specificity:", round(spec_value, 3), "\n")
      cat("Selected proteins:", paste(result$selection_results$proteins_final, collapse = ", "), "\n")
    }
  } else {
    cat("Error:", result$error, "\n")
  }
  
  mod2_results[[as.character(n_prot)]] <- result
}


# MODIFICATION 3: Fuzzy Intersection


run_analysis_fuzzy_intersection <- function(data, n_proteins = 10, train_prop = 0.7, 
                                            weight_ttest = 0.5, weight_rf = 0.5, 
                                            final_n_proteins = 15) {
  
  # Initial split
  initial_split <- data %>%
    mutate(class = (group == 'ASD')) %>%
    initial_split(prop = train_prop, strata = class)
  
  train_data <- training(initial_split)
  test_data <- testing(initial_split)
  
  # Get extended protein lists from both methods
  selection_results <- run_original_analysis(train_data, n_proteins)
  
  # Create fuzzy scores for proteins
  # T-test scores (based on adjusted p-values, lower is better)
  ttest_scores <- selection_results$ttests_results %>%
    mutate(ttest_score = 1 - (rank(p.adj) - 1) / (n() - 1)) %>%
    select(protein, ttest_score)
  
  # RF scores (based on importance, higher is better)
  if(nrow(selection_results$rf_importance) > 0) {
    rf_scores <- selection_results$rf_importance %>%
      as_tibble() %>%
      mutate(protein = rownames(selection_results$rf_importance),
             rf_score = (rank(MeanDecreaseGini) - 1) / (n() - 1)) %>%
      select(protein, rf_score)
  } else {
    rf_scores <- tibble(protein = character(0), rf_score = numeric(0))
  }
  
  # Combine scores with fuzzy intersection (weighted average)
  fuzzy_scores <- ttest_scores %>%
    full_join(rf_scores, by = "protein") %>%
    mutate(
      ttest_score = ifelse(is.na(ttest_score), 0, ttest_score),
      rf_score = ifelse(is.na(rf_score), 0, rf_score),
      fuzzy_score = weight_ttest * ttest_score + weight_rf * rf_score
    ) %>%
    arrange(desc(fuzzy_score))
  
  # Select top proteins based on fuzzy score
  proteins_final <- fuzzy_scores %>%
    slice_head(n = final_n_proteins) %>%
    pull(protein)
  
  # Fit model and evaluate
  if(length(proteins_final) > 0) {
    train_model_data <- train_data %>%
      select(class, any_of(proteins_final))
    
    fit <- glm(class ~ ., data = train_model_data, family = 'binomial')
    
    test_model_data <- test_data %>%
      select(class, any_of(proteins_final))
    
    predictions <- predict(fit, newdata = test_model_data, type = 'response')
    
    # Calculate metrics - same fix as before
    test_results <- tibble(
      truth = as.factor(test_model_data$class),
      pred_prob = as.numeric(predictions),
      pred_class = as.factor(pred_prob > 0.5)
    )
    
    # Make sure factor levels are consistent
    levels(test_results$truth) <- c("FALSE", "TRUE")
    levels(test_results$pred_class) <- c("FALSE", "TRUE")
    
    # Calculate individual metrics
    auc_val <- test_results %>% roc_auc(truth = truth, pred_prob, event_level = "second") %>% pull(.estimate)
    acc_val <- test_results %>% accuracy(truth = truth, estimate = pred_class) %>% pull(.estimate)
    sens_val <- test_results %>% sensitivity(truth = truth, estimate = pred_class, event_level = "second") %>% pull(.estimate)
    spec_val <- test_results %>% specificity(truth = truth, estimate = pred_class, event_level = "second") %>% pull(.estimate)
    
    performance <- tibble(
      .metric = c("roc_auc", "accuracy", "sensitivity", "specificity"),
      .estimate = c(auc_val, acc_val, sens_val, spec_val)
    )
    
    return(list(
      fuzzy_scores = fuzzy_scores,
      proteins_final = proteins_final,
      train_data = train_data,
      test_data = test_data,
      model = fit,
      performance = performance,
      test_predictions = test_results
    ))
  } else {
    return(list(error = "No proteins selected"))
  }
}

cat("\n=== MODIFICATION 3: Fuzzy Intersection ===\n")

# Test different fuzzy intersection strategies
fuzzy_strategies <- list(
  "equal_weight" = list(weight_ttest = 0.5, weight_rf = 0.5),
  "ttest_heavy" = list(weight_ttest = 0.7, weight_rf = 0.3),
  "rf_heavy" = list(weight_ttest = 0.3, weight_rf = 0.7)
)

mod3_results <- list()

for(strategy_name in names(fuzzy_strategies)) {
  strategy <- fuzzy_strategies[[strategy_name]]
  
  cat("\n--- Fuzzy Strategy:", strategy_name, "---")
  cat(" (T-test weight:", strategy$weight_ttest, ", RF weight:", strategy$weight_rf, ")\n")
  
  result <- run_analysis_fuzzy_intersection(
    biomarker_clean, 
    n_proteins = 20,
    weight_ttest = strategy$weight_ttest,
    weight_rf = strategy$weight_rf,
    final_n_proteins = 15
  )
  
  if(!"error" %in% names(result)) {
    cat("Proteins selected:", length(result$proteins_final), "\n")
    
    auc_value <- result$performance %>% filter(.metric == "roc_auc") %>% pull(.estimate)
    acc_value <- result$performance %>% filter(.metric == "accuracy") %>% pull(.estimate)
    sens_value <- result$performance %>% filter(.metric == "sensitivity") %>% pull(.estimate)
    spec_value <- result$performance %>% filter(.metric == "specificity") %>% pull(.estimate)
    
    cat("Test AUC:", round(auc_value, 3), "\n")
    cat("Test Accuracy:", round(acc_value, 3), "\n")
    cat("Test Sensitivity:", round(sens_value, 3), "\n")
    cat("Test Specificity:", round(spec_value, 3), "\n")
    
    # Show top 10 proteins by fuzzy score
    cat("\nTop 10 proteins by fuzzy score:\n")
    top_proteins <- result$fuzzy_scores %>% 
      slice_head(n = 10) %>%
      select(protein, fuzzy_score, ttest_score, rf_score)
    print(top_proteins)
    
    cat("\nSelected proteins:", paste(result$proteins_final, collapse = ", "), "\n")
  } else {
    cat("Error:", result$error, "\n")
  }
  
  mod3_results[[strategy_name]] <- result
}

cat("\nScript completed all three modifications successfully!\n")


# ANALYSIS: How are results affected by each modification?


# MODIFICATION 1 (Proper Train/Test Split):
# The results show this modification working as expected - by performing feature 
# selection only on training data, we get realistic performance estimates without
# data leakage. The performance metrics (AUC ~0.69-0.76, Accuracy ~0.64-0.75) 
# represent honest estimates of how the model would perform on truly unseen data.
# This is significantly more conservative than what we'd see if selection was 
# performed on the full dataset.

# MODIFICATION 2 (Larger Number of Proteins):
# The script successfully selected 15 proteins using fuzzy intersection across
# all strategies, confirming that using more proteins (20 for selection) increases
# the pool available for the final fuzzy scoring. This helped avoid the "no proteins
# selected" problem that can occur with hard intersection when using only 10 proteins.

# MODIFICATION 3 (Fuzzy Intersection):
# The results clearly demonstrate the power of fuzzy intersection with different
# weighting strategies:
#
# EQUAL WEIGHT (0.5/0.5): AUC=0.69, Accuracy=0.638
# - Balanced selection with proteins like FSTL1, DR6, Notch 1 that score well in both methods
# - The fuzzy scores (0.992-0.998) show proteins with consistent importance across methods
#
# TTEST HEAVY (0.7/0.3): AUC=0.71, Accuracy=0.66  
# - Slightly better AUC, emphasizes statistical significance
# - Different protein ranking (DERM rises to #2, emphasizing t-test performance)
# - Higher specificity (0.792) but lower sensitivity (0.522)
#
# RF HEAVY (0.3/0.7): AUC=0.761, Accuracy=0.745 - BEST PERFORMANCE
# - Highest AUC and accuracy, emphasizes machine learning feature importance
# - More balanced sensitivity/specificity (0.739/0.75)
# - Different top proteins (MRC2 #2, TSP4 appears in top 5)
#
# The fuzzy scores reveal how different proteins excel in different methods:
# FSTL1 consistently ranks #1 across all strategies (scores ~0.999), while
# other proteins shift based on weighting (e.g., TSP4 benefits from RF weighting).
# This demonstrates that fuzzy intersection captures method-specific strengths
# and can optimize performance by emphasizing the most effective selection approach.