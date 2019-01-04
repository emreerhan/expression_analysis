
# coding: utf-8

# In[6]:


_random_seed_ = 42


# In[22]:


import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt


# In[142]:


expression_minmax_df = pd.read_csv('../data/processed/expression_log_tpm_minmax.tsv', sep='\t', index_col=0)


# In[143]:


expression_minmax_df = expression_minmax_df.set_index('pog_id')


# In[144]:


drugs_df = pd.read_csv('../data/processed/drugs_filtered.tsv', sep='\t', index_col=0)


# In[145]:


drugs_selected_df = drugs_df[['pog_id', 'drug_name', 'days_on_tx_since_biopsy', 'cancer_cohort']]


# # Prepare features and labels

# ## Join drugs and expression tables

# In[146]:


drugs_expression_df = drugs_selected_df.join(expression_minmax_df, on='pog_id', how='inner')


# In[147]:


drugs_expression_df = drugs_expression_df.drop_duplicates()


# Number of drug types and their names

# ## Select cancer type and drug
# decide based on notebook 0

# In[148]:


drugs_expression_sel_df = drugs_expression_df[(drugs_expression_df['cancer_cohort'] == 'BRCA') & (drugs_expression_df['drug_name'] == 'GEMCITABINE')]


# In[102]:


# drugs_expression_sel_df = drugs_expression_df[(drugs_expression_df['drug_name'] == 'GEMCITABINE')]


# ## Set features (X) and labels (y)

# In[149]:


X = drugs_expression_sel_df.loc[:, expression_minmax_df.columns]


# In[150]:


y = drugs_expression_sel_df.loc[:, 'days_on_tx_since_biopsy']


# ## Power transform on y

# In[151]:


from sklearn.preprocessing import PowerTransformer


# In[152]:


power_transformer = PowerTransformer(method='box-cox', standardize=True)


# In[153]:


y_trans = power_transformer.fit_transform(y.values.reshape(-1, 1))[:, 0]


# ## Naive feature selection: Variance threshold

# In[108]:


from sklearn.feature_selection import VarianceThreshold


# In[124]:


variance_selector = VarianceThreshold(threshold=0.01)


# In[125]:


X_selected = variance_selector.fit_transform(X)


# In[126]:


X_selected.shape


# In[127]:


selected_columns = X.columns[variance_selector.get_support()]


# In[128]:


X_selected_df = pd.DataFrame(data=X_selected, columns=selected_columns, index=X.index)


# In[129]:


from sklearn.model_selection import train_test_split


# In[130]:


X_train, X_test, y_train, y_test = train_test_split(X_selected_df, y_trans, test_size=0.25, random_state=_random_seed_)


# # Recursive Feature Selection

# In[154]:


from sklearn.feature_selection import RFE


# In[155]:


from sklearn.svm import LinearSVR


# In[162]:


num_features = math.floor(len(X_selected_df.columns)/20)


# In[163]:


num_features


# In[164]:


svr = LinearSVR(max_iter=10000)


# In[165]:


selector = RFE(svr, num_features, step=50, verbose=2)


# In[ ]:


selector = selector.fit(X_train, y_train)


# In[ ]:


selector.score(X_test, y_test)

