# Model Development
# # Patient Diagnosis Prediction
# In this lab, we will build a machine learning model to predict whether a patient
# has an epidemiological investigation disease based on some features.
# We will use the patient_diagnosis.csv dataset, which contains the following columns:

# Import the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('data/patient_diagnosis.csv')

# Display the first five rows of the dataset
print(df.head(5).to_string(), '\n')
# Display descriptive statistics
print(df.describe(include='all').to_string(), '\n')

# # Preprocess the dataset
# Handling missing values
df = df.dropna()
# Handling duplicate values
df = df.drop_duplicates()

# Check class balance
print(df['Outcome Variable'].value_counts().to_string(), '\n')

n = 1
plt.figure(figsize=(20, 10))
for i in df.drop('Disease', axis=1).columns:
    plt.subplot(3, 5, n)
    if df[i].dtype == 'object':
        sns.countplot(y=df[i])
    else:
        sns.kdeplot(df[i])
        plt.grid()
    n += 1
plt.tight_layout()
plt.title('Feature Distribution')
plt.show()

# # Feature selection and transformation
# Encode target and categorical variables.
from sklearn.preprocessing import LabelEncoder

mapping = {}

# Encode the categorical variables
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    mapping[col] = dict(zip(le.classes_, le.transform(le.classes_)))

# Display the mapping of the categorical variables
print(mapping, '\n')

# Display descriptive statistics of the dataset after preprocessing
print(df.describe(include='all').to_string(), '\n')

#############################################################################
############################## STEP 2 #######################################
# # # Prepare the data for modeling
# # Split the dataset into features and target variable
# X = df.drop('Outcome Variable', axis=1)
# y = df['Outcome Variable']

# # K-fold cross-validation with stratification
# from sklearn.model_selection import StratifiedKFold
# 
# # Instantiate the StratifiedKFold object
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
# 
# # Split the dataset into training and testing sets
# for train_index, test_index in skf.split(X, y):
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
# 
# # Display the shape of the training and testing sets
# print('X_train shape:', X_train.shape)
# print('X_test shape:', X_test.shape)
# print('y_train shape:', y_train.shape)
# print('y_test shape:', y_test.shape, '\n')

############################################################################
############################## STEP 3 ######################################
# # # Train Model using RandomForestClassifier
# from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier()
# 
# # # GridSearchCV
# from sklearn.model_selection import GridSearchCV
# 
# # Create the parameter grid
# param_grid_rf = {'n_estimators': [50, 100, 150, 200],
#                     'max_depth': [3, 5, 7, 9, 11],
#                     'min_samples_split': [2, 4, 6, 8, 10],
#                     'min_samples_leaf': [1, 2, 3, 4, 5]}
# 
# # Instantiate the GridSearchCV object
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, n_jobs=-1,
#                            verbose=2, scoring='accuracy', refit=True)
# 
# # Fit the GridSearchCV object
# grid_search.fit(X_train, y_train)
# # Get the best model
# best_model = grid_search.best_estimator_


#############################################################################
############################## STEP 4 #######################################
# # # Evaluate Model
# y_pred = best_model.predict(X_test)
# 
# # Calculate the classification report
# from sklearn.metrics import classification_report
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# 
# print('Random Forest Classification Report:')
# print(classification_report(y_test, y_pred), '\n')
# 
# # Calculate the accuracy
# accuracy = accuracy_score(y_test, y_pred)
# presicion = precision_score(y_test, y_pred, average='weighted')
# recall = recall_score(y_test, y_pred, average='weighted')
# f1 = f1_score(y_test, y_pred, average='weighted')
# print('Random Forest Accuracy: {0}, Precision: {1}, Recall: {2}, F1 Score: {3}'
#       .format(accuracy, presicion, recall, f1), '\n')
# 
# # Plot confusion matrix
# from sklearn.metrics import ConfusionMatrixDisplay
# ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
# plt.title('Random Forest Confusion Matrix')
# plt.show()


#############################################################################
############################## STEP 5 #######################################
# # # Save the model
# import joblib
# 
# # Create a directory named models if it does not exist
# import os
# if not os.path.exists('models'):
#     os.makedirs('models')
# 
# # Save the best model as a pickle file and name it with models_label
# joblib.dump(best_model, 'models/best_model.pkl')
# # Save the mapping as a pickle file
# joblib.dump(mapping, 'models/mapping.pkl')
# # Save the column names as a pickle file
# joblib.dump(X.columns, 'models/columns.pkl')
