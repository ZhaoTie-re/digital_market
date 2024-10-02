# usr/bin/env python3
# %%
import argparse

parser = argparse.ArgumentParser(description='Raw data checking.')
parser.add_argument('--csvPath', type=str, help='Path to the raw data file in CSV format.')
parser.add_argument('--targetCol', type=str, help='Name of the target variable in the raw data.')

args = parser.parse_args()

# %%
import pandas as pd

data_src_path = args.csvPath
data_src = pd.read_csv(data_src_path, index_col=0)

target_var_name = args.targetCol

predict_var = data_src.drop([target_var_name], axis=1)
target_var = data_src[target_var_name]

# %% [markdown]
# # Data exploration and visualization for different feature types.
# 
# 1. For numeric variables, we use the KDE （Kernel Density Estimation） to evaluate the distribution of the data and the presence or absence of extreme values or outliers; and check that the distribution of the data is the same before and after the Z-score scaling to make sure that we are not losing the feature information.
# 
# 2. For categorical or hierarchical features, we need to verify that the categorization of each variable is balanced (i.e., whether there are extreme quantitative inequalities) before one-hot encoding using `pd.get_dummies`.

# %%
# For numeric features.

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler

numeric_columns = predict_var.select_dtypes(include=['int64', 'float64']).columns

data_types = predict_var.dtypes

pdf_pages = PdfPages('numeric_features_check.pdf')

plt.rcParams.update({'font.size': 14})

scaler = StandardScaler()
predict_scaled = predict_var.copy()
predict_scaled[numeric_columns] = scaler.fit_transform(predict_var[numeric_columns])

for column in numeric_columns:
    plt.style.use('default')
    fig, axs = plt.subplots(1, 2, figsize=(20, 5))

    sns.kdeplot(predict_var[column], color = "darkblue", 
                linewidth = 1.5, shade=True, label='Original', ax=axs[0])
    axs[0].set_title('Original Density Plot for ' + column)
    axs[0].set_xlabel(column)
    axs[0].set_ylabel('Density')
    axs[0].grid(True)
    axs[0].legend(loc='lower center')

    sns.kdeplot(predict_scaled[column], color = "red", 
                linewidth = 1.5, shade=True, label='Standardized', ax=axs[1])
    axs[1].set_title('Standardized Density Plot for ' + column)
    axs[1].set_xlabel(column)
    axs[1].set_ylabel('Density')
    axs[1].grid(True)
    axs[1].legend(loc='lower center')

    pdf_pages.savefig(fig)

    plt.close()

pdf_pages.close()

# %%
# For categorical or hierarchical features

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

pdf_pages = PdfPages('categorical_feature_check.pdf')

non_numeric_columns = predict_scaled.select_dtypes(include=['object']).columns

for column in non_numeric_columns:
    plt.style.use('default')
    counts = predict_scaled[column].value_counts()

    colors = plt.cm.Paired(range(len(counts)))
    explode = [0.1 if i == 0 else 0 for i in range(len(counts))]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', 
            startangle=90, shadow=True, explode=explode, colors=colors, textprops={'fontsize': 12})

    ax.set_title(f'Distribution of {column}', fontsize=16)

    ax.legend(counts.index, title=column, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    pdf_pages.savefig(fig, bbox_inches='tight')

    plt.close()

pdf_pages.close()


# %%
with pd.HDFStore('raw_data_check.h5') as store:
    store['predict_var'] = predict_var
    store['target_var'] = target_var


