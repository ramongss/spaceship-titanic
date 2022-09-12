# %%
# import libraries
import pandas as pd

from sklearn import model_selection
from sklearn import ensemble
from sklearn import pipeline
from sklearn import metrics

from feature_engine import encoding

import time

def preprocess_data(path):
    # load train data
    df = pd.read_csv(path)

    # drop 'Name' column
    df = df.drop(['Name'], axis=1)

    # fill na
    df = df.fillna(df.mode().iloc[0])

    # change logical to int
    df['CryoSleep'] = df['CryoSleep'].astype(int)
    df['VIP'] = df['VIP'].astype(int)

    if 'Transported' in df.columns.tolist():
        df['Transported'] = df['Transported'].astype(int)

    # split cabin column
    df['Cabin_deck'], df['Cabin_num'], df['Cabin_side'] = df['Cabin'].str.split('/', 3).str
    df['Cabin_num'] = df['Cabin_num'].astype(int)

    return(df)

# %%
# load train data
df_train = preprocess_data('./data/train.csv')

# select features and target
features = df_train.drop(['PassengerId', 'Transported', 'Cabin'], axis=1).columns.to_list()
target = 'Transported'

# split train and test sets
X_train, X_validation, y_train, y_validation = model_selection.train_test_split(df_train[features],
                                                                                df_train[target],
                                                                                random_state=0,
                                                                                test_size=0.2)

# %%
# select categorical and numerical features
cat_features = X_train.dtypes[X_train.dtypes=='object'].index.tolist()
num_features = list(set(X_train.columns) - set(cat_features))

# %%
# create model pipeline
onehot = encoding.OneHotEncoder(drop_last=True, variables=cat_features)

clf = ensemble.RandomForestClassifier(n_estimators=200,
                                      min_samples_leaf=20,
                                      n_jobs=-1,
                                      random_state=0)

params = {"n_estimators":[100, 150, 200,250],
          "min_samples_leaf": [5, 10, 20, 25]}

grid_search = model_selection.GridSearchCV(clf,
                                           params,
                                           n_jobs=1,
                                           cv=4,
                                           scoring='roc_auc',
                                           verbose=-1,
                                           refit=True)

pipe_rf = pipeline.Pipeline(steps = [("One Hot", onehot),
                                     ("Model", grid_search)])

# %%
# train the model
pipe_rf.fit(X_train, y_train)

# %%
# test the model
y_validation_pred = pipe_rf.predict(X_validation)
y_validation_prob = pipe_rf.predict_proba(X_validation)

acc_validation = round(100*metrics.accuracy_score(y_validation, y_validation_pred),2)
roc_validation = metrics.roc_auc_score(y_validation, y_validation_prob[:,1] )

print("Baseline: ", round((1-y_validation.mean())*100,2))
print("acc_validation:", acc_validation)
print("roc_validation:", roc_validation)

# %%
# load test dataset
X_test = preprocess_data('./data/test.csv')

# predict the test
y_test_pred = pipe_rf.predict(X_test[features])
y_test_prob = pipe_rf.predict_proba(X_test[features])

# %%
# create the submission file
df_submit = pd.DataFrame({'PassengerId': X_test['PassengerId'],
                          'Transported': y_test_pred.astype(bool)})
now = time.strftime('%Y-%m-%d_%H:%M:%S')
df_submit.to_csv(f'./results/{now}_submit_file.csv', index=False)