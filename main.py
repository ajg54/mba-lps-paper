"""
Main Python script
Analysing the data sets from the paper 'The dominance, prestige, and leadership account of social power motives'
Original paper authored by Suessenbach et al.
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestRegressor

"""
Settings
"""
input_folder = "input_data"
pd.set_option('display.width', 1000)
pd.options.display.max_columns = 20
dpl_list = ['dominance', 'prestige', 'leadership']
dpl4 = [feature+"4" for feature in dpl_list]
dpl6 = [feature+"6" for feature in dpl_list]
test_proportion = 0.25
random_seed = 42

"""
Load Data
"""
us_data = pd.read_csv(os.path.join(input_folder, 'sample4.csv'), index_col=0)
uk_data = pd.read_csv(os.path.join(input_folder, 'sample5.csv'), index_col=0)
# print(list(us_data))
# print(list(uk_data))
print(uk_data.describe())

"""
Study 7
"""
uk_train_X, uk_test_X, uk_train_y, uk_test_y = train_test_split(uk_data[dpl6],
                                                                uk_data['employment_rank'],
                                                                test_size=test_proportion, random_state=random_seed)
lasso_model = LassoCV()
lasso_model.fit(uk_train_X, uk_train_y)
print(lasso_model.score(uk_test_X, uk_test_y))
print(lasso_model.coef_)

"""
Study 6
"""
uk_data['donated'] = (uk_data['percentage'] > 0).astype('int')
print(uk_data['donated'].describe())
charity_train_X, charity_test_X, charity_train_y, charity_test_y = train_test_split(uk_data[dpl6], uk_data['donated'],
                                                                                    test_size=test_proportion,
                                                                                    random_state=random_seed)
logit_model = LogisticRegressionCV(penalty='l1', solver='liblinear', cv=5)
logit_model.fit(charity_train_X, charity_train_y)
print(logit_model.score(charity_test_X, charity_test_y))
print(logit_model.coef_)
(expanded_train_X, expanded_test_X, 
 expanded_train_y, expanded_test_y) = train_test_split(uk_data[dpl6], uk_data['percentage'],
                                                       test_size=test_proportion, random_state=random_seed)

expanded_model = RandomForestRegressor()
expanded_model.fit(expanded_train_X, expanded_train_y)
print(expanded_model.score(expanded_test_X, expanded_test_y))
print(expanded_model.feature_importances_)
