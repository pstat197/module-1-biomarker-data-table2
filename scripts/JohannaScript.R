library(randomForest)
library(caret)
library(pROC)
library(dplyr)


load("PSTAT 197/module-1-biomarker-data-table2/data/biomarker-clean.RData")

set.seed(123)
train_index <- createDataPartition(biomarker_clean$Group, p = 0.7, list = FALSE)
train_data <- biomarker_clean[train_index, ]
test_data  <- biomarker_clean[-train_index, ]


predictors <- setdiff(names(train_data), c("Group", "ADOS"))

top_n <- 30 

rf_model <- randomForest(
  x = train_data[, predictors],
  y = as.factor(train_data$Group),
  ntree = 500,
  importance = TRUE
)
rf_imp <- importance(rf_model)
rf_ranking <- rownames(rf_imp)[order(rf_imp[, 1], decreasing = TRUE)]
top_rf <- rf_ranking[1:top_n]


logreg_scores <- sapply(predictors, function(var) {
  fit <- glm(as.formula(paste("Group ~", var)),
             data = train_data, family = binomial())
  summary(fit)$coefficients[2, 4]  
})
logreg_ranking <- names(sort(logreg_scores)) 
top_logreg <- logreg_ranking[1:top_n]


all_features <- unique(c(top_rf, top_logreg))
feature_scores <- data.frame(
  protein = all_features,
  rf_rank = match(all_features, rf_ranking),
  logreg_rank = match(all_features, logreg_ranking)
)
feature_scores[is.na(feature_scores)] <- 999
feature_scores$combined_score <- feature_scores$rf_rank + feature_scores$logreg_rank
fuzzy_features <- head(feature_scores[order(feature_scores$combined_score), "protein"], top_n)


final_model <- randomForest(
  x = train_data[, fuzzy_features],
  y = as.factor(train_data$Group),
  ntree = 500,
  importance = TRUE
)

pred_probs <- predict(final_model,
                      newdata = test_data[, fuzzy_features],
                      type = "prob")[, 2]

roc_obj <- roc(test_data$Group, pred_probs)
auc_value <- auc(roc_obj)

cat("\nTest-set AUROC:", round(auc_value, 3), "\n")

