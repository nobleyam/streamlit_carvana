import numpy as np
import pandas as pd

train_FS = pd.read_csv("train_FS.csv")



import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Define columns
numerical_columns = [
    "VehicleAge", "VehOdo", "MMRAcquisitionAuctionAveragePrice",
    "MMRAcquisitionAuctionCleanPrice", "MMRAcquisitionRetailAveragePrice",
    "MMRAcquisitonRetailCleanPrice", "MMRCurrentAuctionAveragePrice",
    "MMRCurrentAuctionCleanPrice", "MMRCurrentRetailAveragePrice",
    "MMRCurrentRetailCleanPrice", "VehBCost", "WarrantyCost"

]

categorical_columns = [
    "Auction", "Make", "Color", "Transmission", "WheelType",
    "Nationality", "Size", "TopThreeAmericanName", "PRIMEUNIT", "AUCGUART", "IsOnlineSale"
]

# Standardize numerical columns
scaler = StandardScaler()
train_FS[numerical_columns] = scaler.fit_transform(train_FS[numerical_columns])

from sklearn.preprocessing import OneHotEncoder

# Create a OneHotEncoder object
one_hot_encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

# Apply One-Hot Encoding to all nominal columns
one_hot_encoded = one_hot_encoder.fit_transform(train_FS[categorical_columns])

# Convert the result into a DataFrame
one_hot_encoded_FS = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(categorical_columns))

# Concatenate the one-hot encoded DataFrame with the original DataFrame
train_FS = pd.concat([train_FS.reset_index(drop=True), one_hot_encoded_FS.reset_index(drop=True)], axis=1)

# Optionally, drop the original nominal columns if they are no longer needed
train_FS.drop(columns=categorical_columns, inplace=True)

joblib.dump(one_hot_encoder, 'one_hot_encoder.pkl')

import pickle

with open("scaler3.pkl", "wb") as f:
    pickle.dump(scaler, f)
import pandas as pd
from sklearn.model_selection import train_test_split

# Assuming train_FS is your dataset and 'IsBadBuy' is your target variable

# Drop the target column ('IsBadBuy') and split the dataset
X = train_FS.drop(columns=['IsBadBuy'])  # Features
y = train_FS['IsBadBuy']  # Target variable

# Split the dataset (70% for training and 30% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Now you have X_train, X_test, y_train, y_test
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

model = LogisticRegression(penalty=None, C=1.0, fit_intercept=True, class_weight='balanced',l1_ratio=None)
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
y_prob_test = model.predict_proba(X_test)

cm_train = confusion_matrix(y_train, y_pred_train)
report_train = classification_report(y_train, y_pred_train)

cm_test = confusion_matrix(y_test, y_pred_test)
report_test = classification_report(y_test, y_pred_test)

print("Evaluation the Model on Training Set")
print(f"Confusion Matrix:\n{cm_train}")
print(f"Classification Report:\n{report_train}")
print("-"*80)
print("Evaluation the Model on Testing Set")
print(f"Confusion Matrix:\n{cm_test}")
print(f"Classification Report:\n{report_test}")

joblib.dump(model, "Logestic_Regression_Model1.pkl")