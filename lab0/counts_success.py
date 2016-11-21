import pandas as pd
import numpy as np


def counts_successes(X, X_test, predictors, target):
    X_new = X.copy()
    X_test_new = X_test.copy()

    # Подсчет counts для каждого признака
    for col_name in predictors:
        tmp = X_new[col_name].value_counts().to_frame().reset_index()
        tmp.columns = [col_name, col_name + '_counts']

        X_new = pd.merge(X_new, tmp, on=col_name)

        X_test_new = pd.merge(X_test_new, tmp, how='left', on=col_name)
        X_test_new = X_test_new.replace(np.nan, 0)
        X_test_new[col_name + '_counts'] = X_test_new[col_name + '_counts'].astype(int)

    # Подсчет successes для каждого признака
    for col_name in predictors:
        tmp = X_new[X_new[target] == 1][
            col_name].value_counts().to_frame().reset_index()
        tmp.columns = [col_name, col_name + '_successes']

        X_new = pd.merge(X_new, tmp, how = 'left', on=col_name)
        X_new = X_new.replace(np.nan, 0)
        X_new[col_name + '_successes'] = X_new[col_name + '_successes'].astype(int)

        X_test_new = pd.merge(X_test_new, tmp, how='left', on=col_name)
        X_test_new = X_test_new.replace(np.nan, 0)
        X_test_new[col_name + '_successes'] = X_test_new[col_name + '_successes'].astype(int)

    # Сглаженное отношение двух предыдущих величин
    for col_name in predictors:
        tmp1 = X_new[col_name + '_successes']
        tmp2 = X_new[col_name + '_counts']
        tmp = ((tmp1 + 1) / (tmp2 + 2)).to_frame()
        tmp.columns = [col_name + '_smooth']
        X_new = pd.concat([X_new, tmp], axis=1, join='inner')

        tmp1 = X_test_new[col_name + '_successes']
        tmp2 = X_test_new[col_name + '_counts']
        tmp = ((tmp1 + 1) / (tmp2 + 2)).to_frame()
        tmp.columns = [col_name + '_smooth']
        X_test_new = pd.concat([X_test_new, tmp], axis=1, join='inner')

    return X_new, X_test_new
