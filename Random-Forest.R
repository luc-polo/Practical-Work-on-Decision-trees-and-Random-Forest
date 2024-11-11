## Problem II

# Load necessary libraries
library(caret)
library(randomForest)

# Load the CSV file
bank_data <- read.csv("C:/Users/polol/OneDrive/Documents/Cours/Mines (cours)/3A/Majeure Science des données/UP 2 Apprentissage statistique/Arbres de décision/TP1-Decision-Trees/7 bank_marketing.csv", sep=";")

# Convert categorical variables to factors
bank_data$marital <- as.factor(bank_data$marital)
bank_data$education <- as.factor(bank_data$education)
bank_data$default <- as.factor(bank_data$default)
bank_data$housing <- as.factor(bank_data$housing)
bank_data$loan <- as.factor(bank_data$loan)
bank_data$contact <- as.factor(bank_data$contact)
bank_data$poutcome <- as.factor(bank_data$poutcome)
bank_data$class <- as.factor(bank_data$class)

# Split data into training set (70%) and test set (30%)
index <- createDataPartition(bank_data$class, p=0.7, list=FALSE)
train_data <- bank_data[index, ]
test_data <- bank_data[-index, ]

# Build 2 random forest models

############################## Model 1
# Implementation 1 with randomForest: Basic model manually optimized with ntree and mtry.

# Initialize lists to store results
ntree_values <- c(50, 150, 250)  # Different ntree values to test
mtry_values <- c(2, 3, 4)         # Different mtry values to test (mtry generally taken as the square root of the number of variables, here 9)
results <- list()

# Define a validation set to test results of different trials (20% validation, 80% training)
fold_size <- nrow(train_data)
val_indices <- sample(1:nrow(train_data), size = 0.2 * nrow(train_data))
val_set = train_data[val_indices,]
train_set = train_data[-val_indices,]

# Loop over ntree and mtry values
for (ntree in ntree_values) {
  for (mtry in mtry_values) {
    # Build random forest model
    rf_model <- randomForest(class ~ ., data=train_set, ntree=ntree, mtry=mtry)
    
    # Predictions on validation set to estimate performance
    rf_pred <- predict(rf_model, newdata=val_set)
    
    # Calculate confusion matrix
    conf_matrix <- confusionMatrix(rf_pred, val_set$class)
    
    # Retrieve accuracy
    accuracy <- conf_matrix$overall["Accuracy"]
    
    # Store results (ntree, mtry, and accuracy)
    results[[paste("ntree =", ntree, "mtry =", mtry)]] <- accuracy
  }
}

# Display results
print(results)

# According to results, the best compromise is "ntree = 100 mtry = 2"

# Train and predict using the optimal model on the validation set
rf_optimal_model <- randomForest(class ~ ., data=train_data, ntree=100, mtry=2)

# Predictions on the test set to estimate performance
rf_optimal_pred <- predict(rf_optimal_model, newdata=test_data)

# Calculate confusion matrix
conf_matrix <- confusionMatrix(rf_optimal_pred, test_data$class)

# Initialize a list to store OOB errors for each combination
oob_errors <- list()

# Loop over ntree and mtry values
for (ntree in ntree_values) {
  oob_per_mtry <- c()  # Temporary list for OOB errors per mtry value
  
  for (mtry in mtry_values) {
    # Build random forest model
    rf_model <- randomForest(class ~ ., data=train_data, ntree=ntree, mtry=mtry)
    
    # Extract average OOB error
    oob_error <- rf_model$err.rate[ntree, "OOB"]
    
    # Store OOB error for this ntree and mtry combination
    oob_per_mtry <- c(oob_per_mtry, oob_error)
  }
  
  # Store OOB errors for the corresponding ntree
  oob_errors[[paste("ntree", ntree)]] <- oob_per_mtry
}

# Plot OOB error evolution
plot(mtry_values, oob_errors[[1]], type="l", col="blue", lwd=2, ylim=c(0, max(unlist(oob_errors))),
     xlab="mtry", ylab="OOB Error", main="OOB Error vs. mtry for different ntree values")
lines(mtry_values, oob_errors[[2]], col="green", lwd=2)
lines(mtry_values, oob_errors[[3]], col="red", lwd=2)

# Add legend
legend("bottomright", legend=c("ntree = 100", "ntree = 300", "ntree = 500"),
       col=c("blue", "green", "red"), lwd=2)

# The decline in accuracy with increasing ntree and mtry may indicate potential overfitting, 
# as more trees or a high number of variables per node make the model too specific to the training data.
# This reduces tree diversity and the model's ability to generalize.

############################## Model 2############################################
# Implementation 2 with caret: Model automatically optimized with cross-validation.

train_data$class <- as.factor(train_data$class)

# Test different fold values for cross-validation
fold_values <- c(5,10)  # Choose the number of folds
results2 <- list()  # Store accuracy results

# Initialize variables to store the best model and its performance
best_model <- NULL
best_accuracy <- 0
best_folds <- NULL

for (folds in fold_values) {
  # Define parameters for cross-validation with different numbers of folds
  train_control <- trainControl(method="cv", number=folds)
  
  # Build the model with cross-validation
  rf_caret_model <- train(class ~ ., data=train_data, method="rf", trControl=train_control, tuneLength=5)
  
  # Retrieve performance results directly from the validated model
  accuracy <- max(rf_caret_model$results$Accuracy)
  
  results2[[paste("folds =", folds)]] <- accuracy
  
  # Save the best model and its accuracy
  if (accuracy > best_accuracy) {
    best_accuracy <- accuracy
    best_model <- rf_caret_model
    best_folds <- folds
  }
}

# Display accuracy results for each fold value
print(results2)
print(best_accuracy)
print(best_model)

# Make predictions on the test set with the best model
rf_caret_pred <- predict(best_model, newdata=test_data)

# Calculate confusion matrix
conf_matrix_caret <- confusionMatrix(rf_caret_pred, test_data$class)

# 10-fold validation gives a similar accuracy to 5-fold. This suggests that 10-fold may not be the best trade-off between bias and variance for your dataset.
# 5-fold validation also provides good accuracy and may be a good choice if reducing computation time is a priority.

# Display confusion matrices of both models
print(conf_matrix)      # Model 1: randomForest
print(conf_matrix_caret)   # Model 2: caret

library(ggplot2)

# For the randomForest model
conf_data_rf <- as.data.frame(conf_matrix$table)

ggplot(conf_data_rf, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Confusion Matrix - randomForest", x = "Actual Class", y = "Predicted Class") +
  theme_minimal() +
  geom_text(aes(label = Freq), color = "black", size = 5)

# For the caret model
conf_data_caret <- as.data.frame(conf_matrix_caret$table)

ggplot(conf_data_caret, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Confusion Matrix - caret", x = "Actual Class", y = "Predicted Class") +
  theme_minimal() +
  geom_text(aes(label = Freq), color = "black", size = 5)

library(pROC)

# Predictions with probabilities for Model 1 (randomForest)
rf_probs <- predict(rf_model, test_data, type="prob")[,2]
roc_rf <- roc(test_data$class, rf_probs)
auc_rf <- auc(roc_rf)

# Predictions with probabilities for Model 2 (caret)
rf_caret_probs <- predict(rf_caret_model, test_data, type="prob")[,2]
roc_caret <- roc(test_data$class, rf_caret_probs)
auc_caret <- auc(roc_caret)

# Plot ROC curves
plot(roc_rf, col="blue", main="ROC Curve - randomForest vs caret")
plot(roc_caret, col="red", add=TRUE)

# Compare AUCs
print(paste("AUC for Model 1 (randomForest):", auc_rf))
print(paste("AUC for Model 2 (caret):", auc_caret))

# For Model 1: randomForest
precision_rf <- conf_matrix$byClass["Pos Pred Value"]
recall_rf <- conf_matrix$byClass["Sensitivity"]

# For Model 2: caret
precision_caret <- conf_matrix_caret$byClass["Pos Pred Value"]
recall_caret <- conf_matrix

