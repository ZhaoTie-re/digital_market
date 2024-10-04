# usr/bin/env python3
# %%
import argparse

parser = argparse.ArgumentParser(description='Multilayer Perceptron implementation.')
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
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, KFold
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE  # Import SMOTE

X = X_train.values.astype('float32') 
y = y_train.values.ravel().astype('int32')

# Apply SMOTE to handle class imbalance
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Compute class weights automatically based on sample distribution
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

# Create neural network models (multilayer perceptron)
def create_model(learning_rate=0.001):
    model = Sequential()
    model.add(Input(shape=(X.shape[1],)))  # Input layer
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))  # first hidden layer with more neurons
    model.add(Dropout(0.5))  # Dropout for regularization
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))  # second hidden layer
    model.add(Dropout(0.3))  # Dropout for regularization
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))  # third hidden layer
    model.add(Dropout(0.3))  # Dropout for regularization
    model.add(Dense(1, activation='sigmoid'))  # Output layer, use sigmoid activation for binary classification
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Create a KerasClassifier
model = KerasClassifier(model=create_model, epochs=100, verbose=0)

# Define the hyperparameters to search (batch size and learning rate)
param_grid = {
    'batch_size': [16, 32, 64],  # Expanded batch sizes
    'model__learning_rate': [1e-3, 1e-2, 1e-1]  # Expanded learning rate search
}

# use KFold cross-validation with 5 splits
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Callbacks for early stopping
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

# Create a GridSearchCV object to search for the best hyperparameters
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, scoring='f1', n_jobs=-1)  # n_jobs=-1 to use all CPU cores (parallelize)

# Fit model with class weights and resampled data from SMOTE
grid_result = grid.fit(X_res, y_res, callbacks=callbacks, class_weight=class_weights_dict)

# %%
import tensorflow as tf
# import netron

# Get the best model    
best_model = grid_result.best_estimator_.model()

# Save the best model in Keras format for later use
model_path = 'best_nn_mlp.keras'
model_path_h5 = 'best_nn_mlp.h5'
tf.keras.models.save_model(best_model, model_path)
tf.keras.models.save_model(best_model, model_path_h5)

# # Visualize the best model using Netron
# netron.start(model_path, browse=False)

# %%
# Get the scores (f1) for each parameter combination and convert them to a DataFrame
scores = grid_result.cv_results_['mean_test_score']
params = grid_result.cv_results_['params']

df_f1 = pd.DataFrame(params)
df_f1['score'] = scores

# Plot the scores for each batch size and learning rate
for learning_rate in df_f1['model__learning_rate'].unique():
    subset = df_f1[df_f1['model__learning_rate'] == learning_rate]
    plt.plot(subset['batch_size'], subset['score'], marker='o', label=f'learning_rate={learning_rate}')

plt.style.use('default')
plt.xlabel('batch_size')
plt.ylabel('f1')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='learning_rate')
plt.tight_layout()
plt.savefig("cv_f1_nn_mlp.pdf", format='pdf')
plt.clf()
# plt.show()

# %%
# Get the scores (neg_log_loss) for each parameter combination and convert them to a DataFrame
grid_loss = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, scoring='neg_log_loss', n_jobs=-1)
grid_result_loss = grid_loss.fit(X_res, y_res, callbacks=callbacks, class_weight=class_weights_dict)

losses = grid_result_loss.cv_results_['mean_test_score']
params = grid_result_loss.cv_results_['params']

df_loss = pd.DataFrame(params)
df_loss['loss'] = losses

# Plot the scores for each batch size and learning rate
for learning_rate in df_loss['model__learning_rate'].unique():
    subset = df_loss[df_loss['model__learning_rate'] == learning_rate]
    plt.plot(subset['batch_size'], subset['loss'], marker='o', label=f'learning_rate={learning_rate}')

plt.style.use('default')
plt.xlabel('batch_size')
plt.ylabel('neg_log_loss')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='learning_rate')
plt.tight_layout()
plt.savefig("cv_loss_nn_mlp.pdf", format='pdf')
plt.clf()
# plt.show()

# %%
# Save final performance to a json file
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

model_name = 'Multilayer Perceptron'
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
auc = roc_auc_score(y_test, y_pred_proba_pos)

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

with open("best_results_nn_mlp.json", "w") as f:
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
plt.savefig("test_perform_nn_mlp.pdf", format='pdf')
plt.clf()
# plt.show()