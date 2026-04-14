import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn import preprocessing

# Load the dataset 
dataset = pd.read_csv('Crop_recommendation.csv')

# Data Preprocessing => convert the categorical labels to numerical values
le = preprocessing.LabelEncoder()
dataset['label'] = le.fit_transform(dataset['label'])

 
X = dataset.drop(columns='label')# features
y = dataset['label'] #labels

# Initialize the decision tree 
decision_tree = DecisionTreeClassifier()

# 5-fold cross-validation
cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#track metrics in each fold
accuracies = []
precisions = []
recalls = []

fold_num = 1
for train_idx, test_idx in cross_validation.split(X, y): # split the dataset into training and testing sets
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx] # training and testing features
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx] # training and testing labels

    decision_tree.fit(X_train, y_train) # Train the model

    y_pred = decision_tree.predict(X_test) # Make predictions
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred) 
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
# Print metrics for each fold
    print(f"--- Fold {fold_num} ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    fold_num += 1

# Overall average metrics across all folds
print(f"\nOverall Average Accuracy: {sum(accuracies)/5:.4f}")
print(f"Overall Average Precision: {sum(precisions)/5:.4f}")
print(f"Overall Average Recall: {sum(recalls)/5:.4f}")




# Visualize the Decision Tree 
plt.figure(figsize=(100, 50))
plot_tree(
    decision_tree,
    filled=True,
    fontsize=6
)
plt.savefig("decision_tree_visualization.png", dpi=300, bbox_inches='tight')
plt.title("Decision Tree Visualization", fontsize=20)
plt.show()

# User input prediction
def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    # Create a DataFrame for user input
    user_data = pd.DataFrame([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]], columns=X.columns)
    prediction = decision_tree.predict(user_data)
    crop = le.inverse_transform(prediction)
    return crop[0]
# User input for crop recommendation
print("\n=== Crop Recommendation System ===")
try:
    nitrogen = float(input("Enter Nitrogen content: "))
    phosphorus = float(input("Enter Phosphorus content: "))
    potassium = float(input("Enter Potassium content: "))
    temperature = float(input("Enter Temperature: "))
    humidity = float(input("Enter Humidity: "))
    soil = float(input("Enter Soil pH: "))
    rainfall = float(input("Enter Rainfall: "))

    # Predict the crop based on user input
    recommended_crop = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, soil, rainfall)
    print(f"\n>>> Recommended Crop: {recommended_crop} <<<")

except ValueError:
    # Handle invalid input
    print("Please enter valid numerical values!")
