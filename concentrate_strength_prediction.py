import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor 

df = pd.read_csv("E:\\aniketh\\data set\\concrete_data.csv")

df.isna().sum()

df.info()

df.describe()

X = df[1:]
y = df['Strength']

sns.set(style="white") 
plt.figure(figsize = (20 , 20))
for variable in range(9):
    plt.subplot(4, 3, variable + 1)
    sns.kdeplot(df[list(X)[variable]], shade = True, color="green")
plt.show()

df['target_binary'] = np.where( df.Strength > 46, 'optimal', 'suboptimal')

sns.set(style="white") 
plt.figure(figsize = (20 , 60))
for variable in range(8):
    plt.subplot(15,3 , variable + 1)
    sns.boxplot(x = df['target_binary'], y =df[list(X)[variable]],  palette='Set1' )
plt.show()

df['target_cat'] = pd.cut(df.Strength,
                     bins=[0, 23, 34, 46, 82],
                     labels=["Good", "Better", "Great", "Perfect"])

sns.set(style="white") 
plt.figure(figsize = (20 , 60))
for variable in range(9):
    plt.subplot(15,3 , variable + 1)
    sns.boxplot(x = df['target_cat'], y = df[list(X)[variable]],  palette='Set1' )
plt.show()

X =  df.drop(columns=['Strength', 'target_cat', 'target_binary'])
y =  df['Strength']

colormap = plt.cm.Blues
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap=colormap, annot=True, linewidths=0.2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True, random_state = 43)

# Feature Scaling 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test =sc.transform(X_test)

algorithms = {"KNN": KNeighborsRegressor(),
          "Linear Regression": LinearRegression(), 
          "Ridge Regression": Ridge(),
          "Random Forest": RandomForestRegressor(),
          "SVR RBF": SVR(),
          "Linear SVR": LinearSVR(),
          "Decision Tree":DecisionTreeRegressor(),
          "Adaboost" :AdaBoostRegressor(),
          "Gradient Boosting":GradientBoostingRegressor(),
          "Neural Network": MLPRegressor(max_iter=10000) ,
          "XGBRegressor" : XGBRegressor()}



def train_and_test(algorithms, X_train,y_train,X_test,y_test):
    model_scores = {}
    for name, model in algorithms.items():
        model.fit(X_train, y_train)
        print(name + " R2: {:.2f}".format(model.score(X_test, y_test)))


model_scores = train_and_test(algorithms, X_train, y_train, X_test, y_test)

estimator = XGBRegressor()
estimator.fit(X_train, y_train)
print("R2: {:.2f}".format(estimator.score(X_test, y_test)))

param_grid       =      {"learning_rate": (0.05, 0.10, 0.15, 0.2),
                         "max_depth": [5, 6, 8],
                         "min_child_weight": [ 5, 7, 9, 11],
                         "gamma":[ 0.0, 0.1, 0.2, 0.25],
                         "colsample_bytree":[ 0.3, 0.4, 0.5, 0.7],
                          "n_estimators": [1000]}


optimized_estimator =  GridSearchCV(estimator, param_grid)
optimized_estimator.fit(X_train, y_train)


optimized_estimator.best_params_
for i, j in optimized_estimator.best_params_.items():
    print("\nBest " + str(i) + " parameter: " +  str(j))


print("XGBoost R2 after hyperparametric Tuning: {:.2f}".format(optimized_estimator.score(X_test, y_test)))
 