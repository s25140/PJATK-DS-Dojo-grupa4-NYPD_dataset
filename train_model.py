import joblib
import lightgbm as lgbm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


def train_model():
    # It's important when loading file to use lib that was used to save file
    X_train = joblib.load('data/X_train.pkl')
    X_test = joblib.load('data/X_test.pkl')
    y_train = joblib.load('data/y_train.pkl')
    y_test = joblib.load('data/y_test.pkl')

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


    model = lgbm.LGBMClassifier(n_estimators=56, num_leaves=31, learning_rate=0.1, reg_alpha=0, reg_lambda=0, min_gain_to_split=0, min_child_weight=0.001, subsample=1, colsample_bytree=0.7)
    model.fit(X_train, y_train)

    joblib.dump(model, 'models/lgbm_model.pkl')


    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')


    print('Accuracy:', accuracy)
    print('F1 Score:', f1)

