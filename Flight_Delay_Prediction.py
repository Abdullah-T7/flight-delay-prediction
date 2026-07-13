import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from xgboost import XGBClassifier


data = pd.read_csv('data/Jan_2019_ontime.csv')

data.drop(data[data['CANCELLED'] == 1].index, inplace=True, errors='ignore')

data.drop([
    'Unnamed: 21',
    'TAIL_NUM',
    'ARR_DEL15',
    'ARR_TIME',
    'DEP_TIME',
    'DIVERTED',
    'CANCELLED',
    'OP_CARRIER',
    'OP_CARRIER_AIRLINE_ID',
    'ORIGIN_AIRPORT_SEQ_ID',
    'ORIGIN_AIRPORT_ID',
    'DEST_AIRPORT_ID',
    'DEST_AIRPORT_SEQ_ID',
    'OP_CARRIER_FL_NUM'
], axis=1, inplace=True, errors='ignore')


y = data['DEP_DEL15']
X = data.drop('DEP_DEL15', axis=1)


categorical_col = [col_name for col_name in X.columns if X[col_name].dtypes == 'str']
numerical_col = X.select_dtypes(include=['int64', 'float64']).columns.tolist()


X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0,
    stratify=y
)


categorical_transformer = Pipeline(
    steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('one', OneHotEncoder(handle_unknown='ignore'))
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_col),
        ('num', 'passthrough', numerical_col)
    ]
)


rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=0,
    class_weight='balanced'
)


rf_pipeline = Pipeline(
    steps=[
        ('pre', preprocessor),
        ('model', rf_model)
    ]
)


rf_pipeline.fit(X_train, y_train)

rf_predicts = rf_pipeline.predict(X_valid)


negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale = negative / positive


X_train_pre = preprocessor.fit_transform(X_train)
X_valid_pre = preprocessor.transform(X_valid)


xgb_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    scale_pos_weight=scale,
    early_stopping_rounds=10,
    random_state=0
)


xgb_model.fit(
    X_train_pre,
    y_train,
    eval_set=[(X_valid_pre, y_valid)],
    verbose=False
)


xgb_predicts = xgb_model.predict(X_valid_pre)


print("Random Forest")
print("Accuracy:", accuracy_score(y_valid, rf_predicts))
print(classification_report(y_valid, rf_predicts))
print(confusion_matrix(y_valid, rf_predicts))


print("XGBoost")
print("Accuracy:", accuracy_score(y_valid, xgb_predicts))
print(classification_report(y_valid, xgb_predicts))
print(confusion_matrix(y_valid, xgb_predicts))


fig, axes = plt.subplots(1, 2, figsize=(12, 5))


ConfusionMatrixDisplay.from_predictions(
    y_valid,
    rf_predicts,
    display_labels=['Not Delayed', 'Delayed'],
    cmap='Greens',
    ax=axes[0]
)

axes[0].set_title('Random Forest')


ConfusionMatrixDisplay.from_predictions(
    y_valid,
    xgb_predicts,
    display_labels=['Not Delayed', 'Delayed'],
    cmap='Blues',
    ax=axes[1]
)

axes[1].set_title('XGBoost')


plt.tight_layout()
plt.show()


actual = y_valid.value_counts().sort_index()

rf_result = pd.Series(rf_predicts).value_counts().sort_index()

xgb_result = pd.Series(xgb_predicts).value_counts().sort_index()


labels = ['Not Delayed', 'Delayed']

x = np.arange(len(labels))

width = 0.25


plt.figure(figsize=(9, 5))

plt.bar(x - width, actual, width, label='Actual')
plt.bar(x, rf_result, width, label='Random Forest')
plt.bar(x + width, xgb_result, width, label='XGBoost')

plt.xticks(x, labels)
plt.ylabel('Number of Flights')
plt.title('Actual vs Predicted')
plt.legend()

plt.show()


difference = rf_predicts != xgb_predicts

plt.figure(figsize=(6, 4))

plt.bar(
    ['Same Prediction', 'Different Prediction'],
    [
        len(difference) - difference.sum(),
        difference.sum()
    ],
    color=['gray', 'red']
)

plt.ylabel('Number of Flights')
plt.title('Prediction Difference')

plt.show()