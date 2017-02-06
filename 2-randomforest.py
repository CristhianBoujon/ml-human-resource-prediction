from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np


df = pd.read_csv("HR_comma_sep.csv")

df_x = df.drop("left", axis = 1)
df_y = df["left"]


# Converts "sale" as category type and integer value
df_x["sales"] = df_x["sales"].astype("category")
df_x["sales"] = df_x["sales"].cat.codes

# Converts "salary" as category type and integer value
df_x["salary"] = df_x["salary"].astype("category")
df_x["salary"] = df_x["salary"].cat.codes

X = df_x.values
y = df_y.values

kfold = StratifiedKFold(n_splits = 5, shuffle=True)

cv_scores, train_scores = np.array([]), np.array([])

for train_idx, cv_idx in kfold.split(X, y): 

    X_train, y_train = X[train_idx], y[train_idx]
    X_cv, y_cv = X[cv_idx], y[cv_idx]

    model = RandomForestClassifier(max_depth = 20)

    model.fit(X_train, y_train)

    cv_score = model.score(X_cv, y_cv)

    train_scores = np.append(train_scores, model.score(X_train, y_train))

    cv_scores = np.append(cv_scores, cv_score)

print train_scores.mean(), train_scores.std()
print cv_scores.mean(), cv_scores.std()
