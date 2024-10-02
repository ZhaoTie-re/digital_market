# usr/bin/env python3
# %%
import argparse

parser = argparse.ArgumentParser(description='Multilayer Perceptron implementation.')
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
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, KFold
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import json

X = X_train.values.astype('float32') 
y = y_train.values.ravel().astype('int32')

# Create neural network models (multilayer perceptron)
def create_model(learning_rate=0.001):
    model = Sequential()
    model.add(Input(shape=(X.shape[1],))) # Input layer
    model.add(Dense(32, activation='relu'))  # first hidden layer 32 neurons
    model.add(Dense(16, activation='relu'))  # second hidden layer 16 neurons
    model.add(Dense(1, activation='sigmoid'))  # Output layer, use sigmoid activation for binary classification
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Create a KerasClassifier
model = KerasClassifier(model=create_model, epochs=50, verbose=0)

# Define the hyperparameters to search (batch size and learning rate only for this example, grid search)
param_grid = {
    'batch_size': [32, 64, 128], 
    'model__learning_rate': [1e-4, 1e-3, 1e-2] 
}

# use KFold cross-validation with 5 splits(5-fold cross-validation)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Create a GridSearchCV object to search for the best hyperparameters
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, scoring='accuracy', n_jobs=-1) # n_jobs=-1 to use all CPU cores (parallelize)
grid_result = grid.fit(X, y)

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
plt.title('Receiver Operating Characteristic')
plt.grid(False)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='ROC statistics')
plt.tight_layout()
plt.savefig("roc_plot_nn_mlp.pdf", format='pdf')
plt.clf()
# plt.show()

# %%
# Get the scores (accurancy) for each parameter combination and convert them to a DataFrame
scores = grid_result.cv_results_['mean_test_score']
params = grid_result.cv_results_['params']

df_accuancy = pd.DataFrame(params)
df_accuancy['score'] = scores

# Plot the scores for each batch size and learning rate
for learning_rate in df_accuancy['model__learning_rate'].unique():
    subset = df_accuancy[df_accuancy['model__learning_rate'] == learning_rate]
    plt.plot(subset['batch_size'], subset['score'], marker='o', label=f'learning_rate={learning_rate}')

plt.style.use('default')
plt.xlabel('batch_size')
plt.ylabel('accuracy')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='learning_rate')
plt.tight_layout()
plt.savefig("test_accurancy_nn_mlp.pdf", format='pdf')
plt.clf()
# plt.show()

# %%
# Get the scores (neg_log_loss) for each parameter combination and convert them to a DataFrame
grid_loss = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, scoring='neg_log_loss', n_jobs=-1)
grid_result_loss = grid_loss.fit(X, y)

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
plt.savefig("test_loss_nn_mlp.pdf", format='pdf')
plt.clf()
# plt.show()

# %%
# Save the best parameters and score to a json file
scores = grid_result.cv_results_['mean_test_score']
params = grid_result.cv_results_['params']

model_name = 'Multilayer Perceptron'
best_params = grid_result.best_params_
best_score = grid_result.best_score_ # this is the accuracy

results = {
    "model": model_name,
    "best_params": best_params,
    "best_accurancy": best_score
}

with open("best_results_nn_mlp.json", "w") as f:
    json.dump(results, f)