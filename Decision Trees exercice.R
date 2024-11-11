
###############################################################################
    #Problem II : Decision trees, an application on real data :
###############################################################################


##### QUESTION 1: #####

csv_file_path = "C:/Users/polol/OneDrive/Documents/Cours/Mines (cours)/3A/Majeure Science des données/UP 2 Apprentissage statistique/Arbres de décision/TP1-Decision-Trees/7 bank_marketing.csv"
# Upload CSV file
data <- read.table(csv_file_path, header = TRUE, sep = ";", stringsAsFactors = FALSE)

# Variable types
variable_types <- sapply(data, class)
print("Variable types:")
print(variable_types)
#Result:
#age:       "integer"
#marital:   "character"
#education: "character"
#default:   "character"
#balance:   "integer"
#housing:   "character"
#loan:      "character"
#contact:   "character"
#poutcome:  "character"
#class:     "character"

# Commentaries:
# All the variables are strings, except "age" and "balance" which are integers.


# Missing values
missing_values <- colSums(is.na(data))
print("Missing values for each variable:")
print(missing_values)

# Commentaries:
# There are no missing values for any of the variables.

# Distributions of observations for different variables
numeric_vars <- names(data)[sapply(data, is.numeric)]
par(mfrow = c(2, 3)) 
for (var in numeric_vars) {
  hist(data[[var]], main = paste("Distribution of", var), xlab = var, col = "lightblue", border = "black")
}

# Create frequency plots for each categorical variable
categorical_vars <- names(data)[sapply(data, is.character)]
par(mfrow = c(2, 4))
plot_list <- list()
for (var in categorical_vars) {
  freq_table <- table(data[[var]])
  print(freq_table)
  freq_df <- as.data.frame(freq_table)
  
  barplot(freq_df$Freq, names.arg = freq_df$Var1, 
          main = paste("Frequency of", var), 
          xlab = var, ylab = "Frequency", 
          col = "lightgreen", 
          las = 2) 
}



# Boxplots
par(mfrow = c(1, 3))  
for (var in numeric_vars) {
  boxplot(data[[var]], main = paste("Boxplot of", var), ylab = var, col = "lightblue")
}


##### QUESTION 2: #####



###Step 1: Data Encoding

library(caret)
#Initializing the encoding model
dummy_model <- dummyVars(class ~ ., data = data)
# Applying the encoding model
encoded_data <- predict(dummy_model, newdata = data)
# Converting to dataframe
encoded_data <- as.data.frame(encoded_data)
# Adding the target variable
encoded_data$class <- data$class  

###Step 2: Splitting the data
# Define the proportion for train and test sets
set.seed(123)  # To ensure reproducibility
train_ratio <- 0.7  
# Calculate the size of the training set
train_size <- floor(train_ratio * nrow(data))
# Create a random sample of indices for the training set
train_indices <- sample(seq_len(nrow(data)), size = train_size)
# Create the test set and train set
test_set = encoded_data[-train_indices,]
train_set = encoded_data[train_indices,]

print(class(data))
###Step 3: Hyperparameter optimization (cross-validation) and decision tree construction
library(rpart)

# Define the hyperparameter values to test
cp_test_values <- seq(0, 0.06, by = 0.02)  # Test different values for cp (useful for pruning)
minsplit_test_values <- seq(20, 30, by = 10)  # Test different values for minsplit
minbucket_test_values <- seq(10, 30, by = 10) # Test different values for minbucket
maxdepth_test_values <- seq(4, 10, by = 3)  # Test different values for maxdepth

# Optimize the decision tree parameters using "manual" cross-validation
cv_splits = 3 #number of cross-validation splits

# Function that returns the best hyperparameters for our decision tree
optimize_parameters <- function(train_set, cv_splits, cp_test_values, minsplit_test_values, minbucket_test_values, maxdepth_test_values) {
  #Table to store results
  results <- data.frame(cp = numeric(), minsplit = numeric(), minbucket = numeric(), maxdepth = numeric(), accuracy = numeric())
  
  # Variable definitions to keep track of the progress of the cross-validation
  total_nb_of_tests = cv_splits * length(cp_test_values) * length(minsplit_test_values) * length(minbucket_test_values) * length(maxdepth_test_values)
  nb_of_tests_made = 0
  
  
  for (cp in cp_test_values) {
    for (minsplit in minsplit_test_values) {
      for (minbucket in minbucket_test_values) {
        for (maxdepth in maxdepth_test_values) {
          
          # Calculate the size of the folds
          fold_size <- floor(nrow(train_set) / cv_splits)
          # Initialize a vector to store fold model accuracy scores
          fold_accuracies_list <- numeric()
          
          for (fold in 1:cv_splits){
            # Real-time tracking of the loop progress
            nb_of_tests_made = nb_of_tests_made + 1
            
            #Define the indices of the training set to be used as the validation set for this fold
            test_indices <- ((fold - 1) * fold_size + 1):(fold * fold_size)
            if (fold == cv_splits) {
              test_indices <- ((fold - 1) * fold_size + 1):nrow(train_set)
            }
            
            # Create the training and validation sets
            val_set <- train_set[test_indices, ]
            fold_train_set <- train_set[-test_indices, ]
            
            
            # Build the decision tree
            decision_tree <- rpart(class ~ .,
                                   data = fold_train_set,
                                   method = "class", 
                                   control = rpart.control(cp = cp,     # Complexity parameter
                                                           minsplit = minsplit, # Minimum number of observations required to split a node
                                                           minbucket = minbucket, # Minimum number of observations in each leaf
                                                           maxdepth = maxdepth)) #Maximum depth
            
            # Predict on the validation fold
            predictions <- predict(decision_tree, newdata = val_set, type = "class")
            # Calculate the accuracy of the model on this fold
            accuracy <- sum(predictions == val_set$class) / length(val_set$class)
            fold_accuracies_list <- c(fold_accuracies_list, accuracy)
            
            # Real-time progress tracking
            cat(nb_of_tests_made,"/",total_nb_of_tests," tests performed \n")
          }
          mean_accuracy <- mean(fold_accuracies_list)
          
          results <- rbind(results, data.frame(cp = cp, minsplit = minsplit, minbucket = minbucket, maxdepth = maxdepth, accuracy = mean_accuracy))
          
        }
      }
    }
  }
  
  cat("The results found are:\n")
  print(results)
  
  # Find the best parameter combination:
  best_hyper_param <- results[which.max(results$accuracy), ]
  print(best_hyper_param)
  # Return the best hyperparameters
  return(list(cp = best_hyper_param$cp, 
              minsplit = best_hyper_param$minsplit, 
              minbucket = best_hyper_param$minbucket, 
              maxdepth = best_hyper_param$maxdepth))
}

#Execute the function
list_best_param = optimize_parameters(train_set, cv_splits, cp_test_values, minsplit_test_values, minbucket_test_values, maxdepth_test_values)



###Step 3: Build and train the model
# Define sample weights to balance the importance of the classes that are unequally present in the dataset


optimised_decision_tree <- rpart(class ~ .,
                                 data = train_set,
                                 method = "class", 
                                 control = rpart.control(cp = list_best_param$cp,     # Best complexity parameter 
                                                         minsplit = list_best_param$minsplit, # Best minimum number of observations required to split a node
                                                         minbucket = list_best_param$minbucket, # Best minimum number of observations in each leaf
                                                         maxdepth = list_best_param$maxdepth)) # Best maximum depth

# Visualizing the tree
# Load the library to visualize the tree
library(rpart.plot)
# Visualize the decision tree
par(mfrow = c(1, 1)) 
rpart.plot(optimised_decision_tree)



##### QUESTION 3: #####


## Predict classes on the test set
predictions_test <- predict(optimised_decision_tree, newdata = test_set, type = "class")

# Calculate the accuracy (or other performance metric)
accuracy <- sum(predictions_test == test_set$class) / nrow(test_set)
cat("Accuracy on the test set:", accuracy, "\n")

### Calculate the confusion matrix
table(predictions_test, test_set$class)

# predictions_test   no  yes
#               no  1691  297
#               yes 121  251

### ROC curve
# Convert classes to binary (1 for 'yes', 0 for 'no')
test_set$class_binary <- ifelse(test_set$class == "yes", 1, 0)
# Predict probabilities for the positive class
probabilities <- predict(optimised_decision_tree, test_set, type = "prob")[, "yes"]
# Compute the ROC curve
roc_curve <- roc(test_set$class_binary, probabilities)
# Plot the ROC curve
library(pROC)
plot(roc_curve, main = "ROC Curve")
# Calculate and print the AUC (Area Under the Curve)
auc_value <- auc(roc_curve)
cat("AUC (Area Under the Curve):", auc_value, "\n")


