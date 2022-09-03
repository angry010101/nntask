from math import sqrt
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv("internship_train.csv")

target = "target"
ypoints = data[target]
del data[target]
xpoints = data


X_train, X_test, y_train, y_test = train_test_split(
    xpoints, ypoints, test_size=0.10, random_state=42, shuffle=True)

linreg = GradientBoostingRegressor()
linreg.fit(X_train, y_train)

ytrain_pred = linreg.predict(X_train)
y_pred = linreg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)

X_hidden_test = pd.read_csv("internship_hidden_test.csv")
y_hidden_pred = linreg.predict(X_hidden_test)
pd.DataFrame(y_hidden_pred).to_csv("predictions.csv")
print('RMSE: %f' % rmse)

