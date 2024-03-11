import sklearn
import numpy as np
import pandas as pd
import pickle
from typing import Tuple
import numpy as np
from sklearn.pipeline import Pipeline

with open("D:\Applied ML\ASSINGMENT_3\Best_models\logistic_regression_best_model.pkl", 'rb') as file:
    best_model = pickle.load(file)



def score(text:str, model:Pipeline, threshold=0.5) -> Tuple[bool , float]:
   
    # Vectorize the text
    vect_text = model.named_steps['tfidf'].transform([text])
    
    # Get the propensity score for the positive class
    propensity = model.named_steps['clf'].predict_proba(vect_text)[:, 1][0]
    prediction = (propensity >= threshold)
    
    return bool(prediction), propensity