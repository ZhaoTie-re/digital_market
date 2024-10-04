# usr/bin/env python3
import argparse

parser = argparse.ArgumentParser(description='XGBClassifier implementation.')
parser.add_argument('--h5Path', type=str, help='Path to HDF5 output from channel.')
parser.add_argument('--threshold', type=float, default=0.5, help='Threshold value for binary classification.')

args = parser.parse_args()

# %%
import pandas as pd

file_path = args.h5Path

with pd.HDFStore(file_path) as store:
    X_train = store['X_train']
    X_test = store['X_test']
    y_train = store['y_train']
    y_test = store['y_test']

# %%
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

X = X_train.values.astype('float32')
y = y_train.values.ravel().astype('int32')

# Crate XGBoost model
def create_model(learning_rate=0.001, max_depth=3):
    return XGBClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        objective='binary:logistic',  # Use logistic regression for binary classification
    )

# Create a XGBClassifier
model = XGBClassifier(model=create_model)

# Define the hyperparameters to search (learning rate and max depth)
param_grid = {
    'learning_rate': [1e-3, 1e-2, 1e-1], 
    'max_depth': [3, 5, 7, 9, 11]
}

# Use KFold cross-validation with 5 splits (5-fold cross-validation)
kf = KFold(n_splits=5, shuffle=True, random_state=42) # random_state for reproducibility

# Create a GridSearchCV object to search for the best hyperparameters (F1 score)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, scoring='f1', n_jobs=-1)

# Fit the GridSearchCV object to the data
grid_result = grid.fit(X, y)

# %%
import pickle
from joblib import dump

# Get the best model
best_model = grid_result.best_estimator_

# Save the best model to a file using pickle
with open('best_xgb_classifier.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save the best model to a file using joblib
dump(best_model, 'best_xgb_classifier.joblib')

# %%
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Get the best model
best_model = grid_result.best_estimator_

# Plot the first 10 trees
with PdfPages('first_10_trees.pdf') as pdf:
    for i in range(10): 
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(20, 8), dpi=300, tight_layout=True)
        xgb.plot_tree(best_model, num_trees=i, ax=ax)
        pdf.savefig(fig) 
        plt.close(fig) 

# %%
# Get the scores (f1) for each parameter combination and convert them to a DataFrame
scores = grid_result.cv_results_['mean_test_score']
params = grid_result.cv_results_['params']

df_f1 = pd.DataFrame(params)
df_f1['score'] = scores

# Plot the scores for each max depth and learning rate
for learning_rate in df_f1['learning_rate'].unique():
    subset = df_f1[df_f1['learning_rate'] == learning_rate]
    plt.plot(subset['max_depth'], subset['score'], marker='o', label=f'learning_rate={learning_rate}')

plt.style.use('default')
plt.xlabel('max_depth')
plt.ylabel('f1')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='learning_rate')
plt.tight_layout()
plt.savefig("cv_f1_xgb_classifier.pdf", format='pdf')
plt.clf()
# plt.show()

# %%
grid_loss = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, scoring='neg_log_loss', n_jobs=-1)
grid_result_loss = grid_loss.fit(X, y)

losses = grid_result_loss.cv_results_['mean_test_score']
params = grid_result_loss.cv_results_['params']

df_loss = pd.DataFrame(params)
df_loss['loss'] = losses

# Plot the scores for each max depth and learning rate
for learning_rate in df_loss['learning_rate'].unique():
    subset = df_loss[df_loss['learning_rate'] == learning_rate]
    plt.plot(subset['max_depth'], subset['loss'], marker='o', label=f'learning_rate={learning_rate}')

plt.style.use('default')
plt.xlabel('max_depth')
plt.ylabel('neg_log_loss')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='learning_rate')
plt.tight_layout()
plt.savefig("cv_loss_xgb_classifier.pdf", format='pdf')
plt.clf()
# plt.show()

# %%
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

model_name = 'XGBClassifier'
best_params = grid_result.best_params_
best_score = grid_result.best_score_ # this is the f1 score

# Predict probabilities for the test set
y_pred_proba = grid_result.best_estimator_.predict_proba(X_test)

# Extract probabilities for the positive class (class 1)
y_pred_proba_pos = y_pred_proba[:, 1]

# Adjust threshold
threshold = args.threshold
y_pred = (y_pred_proba_pos >= threshold).astype(int)

# Compute the metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

results = {
    "model": model_name,
    "best_params": best_params,
    "best_cv_f1": best_score,
    "best_test_f1": f1,
    "best_test_accuracy": accuracy,
    "best_test_precision": precision,
    "best_test_recall": recall,
    "best_test_auc": auc
}

with open("best_results_xgb_classifier.json", "w") as f:
    json.dump(results, f)
    
# %%
from sklearn.metrics import confusion_matrix,  roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_pos)
roc_auc = roc_auc_score(y_test, y_pred)

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16}, cbar=True, ax=ax1)
ax1.set_title('Confusion Matrix', fontsize=20)
ax1.set_xlabel('Predicted', fontsize=16)
ax1.set_ylabel('Truth', fontsize=16)

# Plot ROC curve
ax2.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate', fontsize=16)
ax2.set_ylabel('True Positive Rate', fontsize=16)
ax2.set_title('Receiver Operating Characteristic', fontsize=20)
ax2.legend(loc="lower right")
plt.tight_layout()
plt.savefig("test_perform_xgb_classifier.pdf", format='pdf')
plt.clf()
# plt.show()