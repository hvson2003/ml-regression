import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, KFold
from math import sqrt

data = pd.read_csv('water-consumption.csv')

# Chia tập huấn luyện và tập kiểm tra
dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle=True)

X_train = dt_Train.iloc[:, 2:11]
y_train = dt_Train.iloc[:, 11]
X_test = dt_Test.iloc[:, 2:11]
y_test = dt_Test.iloc[:, 11]


# print("Tập X train: ")
# print(X_train)
# print("Tập X test: ")
# print(X_test)
# print("Tập y train: ")
# print(y_train)
# print("Tập y test: ");
# print(y_test)

# Hàm để đánh giá mô hình và in kết quả
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    nse = 1 - (sum((y_pred - y_test) ** 2) / sum((y_test - y_test.mean()) ** 2))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    return nse, r2, mae, rmse

# Linear Regression
lr1 = LinearRegression()
lr1.fit(X_train, y_train)
nse_lr1, r2_lr1, mae_lr1, rmse_lr1 = evaluate_model(lr1, X_test, y_test)

# LASSO Regression
lasso1 = Lasso(alpha=1.0)
lasso1.fit(X_train, y_train)
nse_lasso1, r2_lasso1, mae_lasso1, rmse_lasso1 = evaluate_model(lasso1, X_test, y_test)

# Ridge Regression
ridge1 = Ridge(alpha=1.0)
ridge1.fit(X_train, y_train)
nse_ridge1, r2_ridge1, mae_ridge1, rmse_ridge1 = evaluate_model(ridge1, X_test, y_test)

# K-fold Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

r2_scores, mae_scores, rmse_scores, nse_scores = [], [], [], []

for train_index, test_index in kf.split(X_train):
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    lr = LinearRegression()
    lr.fit(X_train_fold, y_train_fold)
    nse, r2, mae, rmse = evaluate_model(lr, X_test_fold, y_test_fold)

    r2_scores.append(r2)
    mae_scores.append(mae)
    rmse_scores.append(rmse)
    nse_scores.append(nse)

# Trung bình các chỉ số đánh giá từ K-fold Cross-validation
r2_kfold = sum(r2_scores) / len(r2_scores)
mae_kfold = sum(mae_scores) / len(mae_scores)
rmse_kfold = sum(rmse_scores) / len(rmse_scores)
nse_kfold = sum(nse_scores) / len(nse_scores)

# In kết quả so sánh
print("Linear Regression:")
print(f'R2: {r2_lr1:.10f}, MAE: {mae_lr1:.10f}, RMSE: {rmse_lr1:.10f}, NSE: {nse_lr1:.10f}')

print("\nLASSO Regression:")
print(f'R2: {r2_lasso1:.10f}, MAE: {mae_lasso1:.10f}, RMSE: {rmse_lasso1:.10f}, NSE: {nse_lasso1:.10f}')

print("\nRidge Regression:")
print(f'R2: {r2_ridge1:.10f}, MAE: {mae_ridge1:.10f}, RMSE: {rmse_ridge1:.10f}, NSE: {nse_ridge1:.10f}')

print("\nK-fold Cross-validation:")
print(f'R2: {r2_kfold:.10f}, MAE: {mae_kfold:.10f}, RMSE: {rmse_kfold:.10f}, NSE: {nse_kfold:.10f}')
