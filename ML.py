import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, mean_squared_error
import streamlit as st
import numpy as np

def get_best_model(df):
    X = df['reviews']
    y = df['ratings']

    # Daten in Trainings- und Test-Sets aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pipeline mit CountVectorizer und einem ML-Modell (Multinomial Naive Bayes) erstellen
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])

    # Parameter für GridSearchCV festlegen
    param_grid = {
        'vectorizer__max_features': [500, 1000, 1500],
        'vectorizer__ngram_range': [(1, 1), (1, 2)],
        'classifier__alpha': [0.1, 1, 10]
    }

    # GridSearchCV mit der Pipeline und den Parametern ausführen
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Bestes Modell auswählen und bewerten
    best_model = grid_search.best_estimator_
    vect = best_model.named_steps['vectorizer']


    y_pred = best_model.predict(X_test)

    # Mean Squared Error (MSE) berechnen
    mse = mean_squared_error(y_test, y_pred)
    # Root Mean Squared Error (RMSE) berechnen
    rmse = np.sqrt(mse)

    # Ergebnisse ausgeben
    st.markdown(f"Mittlere Abweichung des Models: {rmse}")

    return best_model, vect