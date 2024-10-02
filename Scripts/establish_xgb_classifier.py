# usr/bin/env python3
import argparse

parser = argparse.ArgumentParser(description='XGBClassifier implementation.')
parser.add_argument('--h5Path', type=str, help='Path to HDF5 output from channel.')

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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Get the best model
best_model = grid.best_estimator_
roc_curves = []
aucs = []

# Get the fp, tp, and auc for each fold
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    best_model.fit(X_train, y_train)

    y_pred = best_model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_curves.append((fpr, tpr))
    aucs.append(auc(fpr, tpr))

# Compute the mean ROC curve and AUC
mean_fpr = np.linspace(0, 1, 100)
mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in roc_curves], axis=0)
std_tpr = np.std([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in roc_curves], axis=0)

# Compute the 95% confidence interval for the mean ROC curve
tprs_upper = np.minimum(mean_tpr + 1.96*std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - 1.96*std_tpr, 0)

# Plot the mean ROC curve with 95% confidence interval
plt.style.use('default')
plt.plot(mean_fpr, mean_tpr, color='darkorange', lw=2, label=r'Mean ROC (AUC = %0.2f)' % np.mean(aucs))
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='skyblue', alpha=.4, label=r'$\pm$ 1.96 std. dev. (95% CI)')
plt.vlines(mean_fpr, tprs_lower, tprs_upper, color='skyblue', alpha=.4)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('')
plt.grid(False)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='ROC statistics')
plt.tight_layout()
plt.savefig("roc_plot_xgb_classifier.pdf", format='pdf')
plt.clf()
# plt.show()

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
plt.savefig("test_f1_xgb_classifier.pdf", format='pdf')
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
plt.savefig("test_loss_xgb_classifier.pdf", format='pdf')
plt.clf()
# plt.show()

# %%
import json

model_name = 'XGBClassifier'
best_params = grid_result.best_params_
best_score = grid_result.best_score_ # this is the f1 score

results = {
    "model": model_name,
    "best_params": best_params,
    "best_f1": best_score
}

with open("best_results_xgb_classifier.json", "w") as f:
    json.dump(results, f)


