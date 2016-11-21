import numpy as np


class Dist (object):

    def __init__(self, X):
        l = float(X.shape[0])
        p_2 = []
        f = []

        for col_name in X.columns:
            p_ = []
            p_2_ = []
            f_ = []
            key_ = []

            for key, value in X.groupby(col_name).groups.items():
                f_j = len(value)
                p_.append(f_j / l)
                p_2_.append(f_j * (f_j - 1) / l / (l - 1))
                f_.append(np.log(f_j + 1))
                key_.append(key)

            idxs  = np.argsort(p_)
            p_2_ = np.array(p_2_)[idxs]
            key_ = np.array(key_)[idxs]
            p_2.append(np.column_stack((np.cumsum(p_2_), key_)))
            f.append(np.array(f_))

        self.p_2 = p_2[:]
        self.f = f[:]

    def ind_matches(self, x, y):
        return np.count_nonzero(x - y)

    def smooth_ind_matches(self, x, y):
        sum_ = 0.0
        l = x.size
        idxs = np.nonzero(x == y)[0]
        p_2 = self.p_2
        for idx in idxs:
            vals = p_2[idx]
            sum_ += vals[np.nonzero(vals[:, 1] == x[idx])][0][0]
        return (l - idxs.size) + sum_

    def log_ind_matches(self, x, y):
        sum_ = 0.0
        idxs = np.nonzero(x - y)[0]
        f = self.f
        for idx in idxs:
            vals = f[idx]
            size = vals.size
            x_val = x[idx]
            y_val = y[idx]
            if size > x_val and size > y_val:
                sum_ += vals[x_val] * vals[y_val]
        return sum_