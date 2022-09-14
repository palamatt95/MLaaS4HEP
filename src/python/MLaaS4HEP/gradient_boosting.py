from sklearn.ensemble import GradientBoostingClassifier

def model():
    gbc = GradientBoostingClassifier(n_estimators=50, max_depth=10,min_samples_leaf=200,max_features=10,verbose=1)
    return gbc
