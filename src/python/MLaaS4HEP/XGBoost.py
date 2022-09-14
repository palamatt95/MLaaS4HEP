import xgboost as xgb

def model():
    params = {
        'objective':'binary:logistic',
        'max_depth': 9,
        'subsample': 0.9,
        'alpha': 10,
        'learning_rate': 0.0001,
        'n_estimators': 3000
    }
    xgb_model = xgb.XGBClassifier(**params)
    return xgb_model
