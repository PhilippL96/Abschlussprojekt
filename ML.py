import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import mean_squared_error
import numpy as np

def preprocess_data(df):
    X = df['reviews']
    y = df['ratings']
    
    vectorizer = CountVectorizer()
    X_transformed = vectorizer.fit_transform(X)
    
    return X_transformed, y, vectorizer

def get_best_model(X_transformed, y):
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

    param_grid = {
        'alpha': [0.1, 1, 10]
    }
    
    model = MultinomialNB()
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    return best_model, rmse