# usr/bin/env python3
# %%
import argparse

parser = argparse.ArgumentParser(description='Featur engineering.')
parser.add_argument('--h5Path', type=str, help='Path to HDF5 output from raw_data_check channel.')
parser.add_argument('--ignoreFeaturePath', type=str, help='Path to the file containing the features to be ignored.')

args = parser.parse_args()

# %%
import pandas as pd

file_path = args.h5Path
ignore_feature_path = args.ignoreFeaturePath

# Read the DataFrames from the HDF5 file
with pd.HDFStore(file_path) as store:
    predict_var = store['predict_var']
    target_var = store['target_var']
    
with open(ignore_feature_path, 'r') as file:
    ignore_feature = [line.strip() for line in file.readlines()]

# %% [markdown]
# # Feature engineering - normalization, transformation & one-hot encoding

# %%
from sklearn.preprocessing import StandardScaler

numerical_cols = predict_var.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
predict_var[numerical_cols] = scaler.fit_transform(predict_var[numerical_cols])

# %%
non_numerical_cols = predict_var.select_dtypes(include=['object']).columns
predict_var = pd.get_dummies(predict_var, columns=non_numerical_cols)

# %% [markdown]
# # Feature engineering - feature selection
# In this process, we use two methods, `Elastic Networks` as well as `Mutual Information` to weigh the predictive power of the predict_var with respect to the target_var.

# %%
# Elastic Net

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

elastic_net = ElasticNet()

param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1],
    'l1_ratio': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
}

grid_search = GridSearchCV(elastic_net, param_grid, cv=5)
grid_search.fit(predict_var, target_var)
best_params = grid_search.best_params_

elastic_net = ElasticNet(alpha=best_params['alpha'], l1_ratio=best_params['l1_ratio'])
elastic_net.fit(predict_var, target_var)

# %%
# Mutual Information

from sklearn.feature_selection import mutual_info_classif

mutual_info = mutual_info_classif(predict_var, target_var)

# %%
# Integration of two approaches

import matplotlib.pyplot as plt
import numpy as np

elastic_net_coef = elastic_net.coef_

df = pd.DataFrame({
    'mutual_info': mutual_info,
    'elastic_net_coef': elastic_net_coef
}, index=predict_var.columns)

df_sorted1 = df.reindex(df['mutual_info'].sort_values(ascending=False).index)
df_sorted2 = df.reindex(df['elastic_net_coef'].abs().sort_values(ascending=False).index)

plt.style.use('default')
fig, axs = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.3})

cax1 = axs[0].imshow(df_sorted1['mutual_info'].values.reshape(-1, 1), cmap='hot', interpolation='nearest', aspect=0.2)
axs[0].set_title('mutual_info')

cax2 = axs[1].imshow(df_sorted2['elastic_net_coef'].values.reshape(-1, 1), cmap='cool', interpolation='nearest', aspect=0.2)
axs[1].set_title('elastic_net_coef')

axs[0].set_yticks(np.arange(len(df_sorted1.index)))
axs[0].set_yticklabels(df_sorted1.index)
axs[0].set_xticks([]) 

axs[1].set_yticks(np.arange(len(df_sorted2.index)))
axs[1].set_yticklabels(df_sorted2.index)
axs[1].set_xticks([])  

fig.colorbar(cax1, ax=axs[0], shrink=0.5, location='right')
fig.colorbar(cax2, ax=axs[1], shrink=0.5, location='right')

plt.tight_layout()
plt.savefig("feature_importance.pdf", format='pdf')
# plt.show()


# %%
from sklearn.model_selection import train_test_split

predict_var = predict_var.drop(columns=[col for col in predict_var.columns if any((feature == col or col.startswith(feature + '_')) for feature in ignore_feature)])
target_var = pd.DataFrame(target_var)

X_train, X_test, y_train, y_test = train_test_split(predict_var, target_var, test_size=0.2, stratify=target_var, random_state=42)

# %%
with pd.HDFStore('feature_engineering.h5') as store:
    store['X_train'] = X_train
    store['X_test'] = X_test
    store['y_train'] = y_train
    store['y_test'] = y_test


