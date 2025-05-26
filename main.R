# Step 1: Install & Load All Required Libraries
# List of needed packages
libraries <- c("readr", "ggplot2", "corrplot", "dplyr", "caret", 
               "randomForest", "rpart", "rpart.plot", "pROC", 
               "reshape2", "writexl")

# Install missing ones and load all
for (lib in libraries) {
  if (!require(lib, character.only = TRUE)) install.packages(lib)
  library(lib, character.only = TRUE)
}

# Step 2: Load the Dataset
# Load the dataset from local file path
churn_data <- read.csv("C:/Users/Mohan/Documents/Dataset.csv")

# Show a quick look at the data
head(churn_data)

# Step 3: Data
# Summary statistics
summary(churn_data)

# Structure of the dataset
str(churn_data)

# Count missing values
cat("Total missing values:", sum(is.na(churn_data)), "\n")

# Step 4: Data Cleaning & Preparation

# Convert categorical columns to factors
churn_data$Churn <- as.factor(churn_data$Churn)
churn_data$International.plan <- as.factor(churn_data$International.plan)
churn_data$Voice.mail.plan <- as.factor(churn_data$Voice.mail.plan)

# Drop columns that don’t help in modeling
clean_data <- churn_data %>% select(-State, -Area.code)
# Step 5: Exploratory Data Analysis (EDA)
#  Churn Count Plot
ggplot(clean_data, aes(x = Churn)) + 
  geom_bar(fill = c("skyblue", "tomato")) +
  ggtitle("Churn Distribution") +
  theme_minimal()
#  Histograms of Numeric Columns
numeric_features <- sapply(clean_data, is.numeric)
reshaped_data <- melt(clean_data[, numeric_features])

ggplot(reshaped_data, aes(x = value)) + 
  facet_wrap(~ variable, scales = "free", ncol = 3) +
  geom_histogram(fill = "#4682B4", color = "black", alpha = 0.8) +
  theme_minimal() +
  labs(title = "Histogram of Numeric Features")

# Correlation Heatmap
correlations <- cor(clean_data[, numeric_features])
corrplot(correlations, method = "color", type = "upper", tl.cex = 0.8)

# Step 6: Split the Dataset
set.seed(123)

# 70% for training, 30% for testing
split_indices <- createDataPartition(clean_data$Churn, p = 0.7, list = FALSE)
training_set <- clean_data[split_indices, ]
testing_set <- clean_data[-split_indices, ]
# Step 7: Build a Decision Tree Model
decision_tree <- rpart(Churn ~ ., data = training_set, method = "class")

rpart.plot(decision_tree, extra = 104, fallen.leaves = TRUE, 
           main = "Decision Tree for Churn Prediction")
# Step 8: Build a Random Forest Model
forest_model <- randomForest(Churn ~ ., data = training_set, 
                             ntree = 100, importance = TRUE)

# View model summary
print(forest_model)

# Plot most important features
varImpPlot(forest_model, main = "Feature Importance (Random Forest)")
# Step 9: Make Predictions on the Test Set
# Get probabilities for the positive class
rf_probabilities <- predict(forest_model, testing_set, type = "prob")[, 2]
tree_probabilities <- predict(decision_tree, testing_set, type = "prob")[, 2]

# Get predicted labels
rf_predictions <- predict(forest_model, testing_set, type = "response")
tree_predictions <- predict(decision_tree, testing_set, type = "class")

 # Step 10: ROC Curve & Model Performance
# ROC Curves
rf_roc_curve <- roc(testing_set$Churn, rf_probabilities)
tree_roc_curve <- roc(testing_set$Churn, tree_probabilities)

# Plot both
plot(rf_roc_curve, col = "navy", lwd = 2, main = "ROC Curve Comparison")
lines(tree_roc_curve, col = "darkred", lwd = 2)
legend("bottomright", legend = c("Random Forest", "Decision Tree"), 
       col = c("navy", "darkred"), lwd = 2)

# Print AUC values
cat("Random Forest AUC:", auc(rf_roc_curve), "\n")
cat("Decision Tree AUC:", auc(tree_roc_curve), "\n")


# Step 11: Export Prediction Results
# Collect everything into a final table
final_results <- data.frame(
  Actual = testing_set$Churn,
  RF_Predicted = rf_predictions,
  RF_Prob = rf_probabilities,
  Tree_Predicted = tree_predictions,
  Tree_Prob = tree_probabilities
)

# Save to CSV and Excel
write.csv(final_results, "Churn_Predictions.csv", row.names = FALSE)
write_xlsx(final_results, "Churn_Predictions.xlsx")


