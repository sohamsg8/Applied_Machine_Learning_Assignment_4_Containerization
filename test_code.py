import unittest
import pickle
import pandas as pd
import random
import numpy as np
import requests
import subprocess
import time
from score import *

class TestScoreFunction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open("D:\Applied ML\ASSINGMENT_3\Best_models\logistic_regression_best_model.pkl", 'rb') as model_file:
            cls.loaded_model = pickle.load(model_file)
        cls.test_df = pd.read_csv(r"D:\Applied ML\ASSINGMENT_3\Data\test.csv")

    def test_smoke_test(self):
        r = random.randint(0, len(self.test_df) - 1)
        text = self.test_df.iat[r, 0]
        result = score(text, self.loaded_model, 0.5)
        self.assertIsNotNone(result)

    def test_format_test(self):
        r = random.randint(0, len(self.test_df) - 1)
        text = self.test_df.iat[r, 0]
        prediction, propensity = score(text, self.loaded_model, 0.5)
        self.assertIsInstance(prediction, (bool, np.bool_))
        self.assertIsInstance(propensity, float)

    def test_prediction_values(self):
        r = random.randint(0, len(self.test_df) - 1)
        text = self.test_df.iat[r, 0]
        prediction, _ = score(text, self.loaded_model, 0.5)
        self.assertIn(prediction, [True, False])

    def test_propensity_range(self):
        r = random.randint(0, len(self.test_df) - 1)
        text = self.test_df.iat[r, 0]
        _, propensity = score(text, self.loaded_model, 0.5)
        self.assertTrue(0 <= propensity <= 1)

    def test_threshold_zero(self):
        r = random.randint(0, len(self.test_df) - 1)
        text = self.test_df.iat[r, 0]
        prediction, _ = score(text, self.loaded_model, 0.0)
        self.assertTrue(prediction)

    def test_threshold_one(self):
        r = random.randint(0, len(self.test_df) - 1)
        text = self.test_df.iat[r, 0]
        prediction, _ = score(text, self.loaded_model, 1.0)
        self.assertFalse(prediction)

    def test_obvious_spam_input(self):
        text = self.test_df.iat[2, 0]
        prediction, _ = score(text, self.loaded_model, 0.5)
        self.assertTrue(prediction)

    def test_obvious_non_spam_input(self):
        text = self.test_df.iat[3, 0]
        prediction, _ = score(text, self.loaded_model, 0.5)
        self.assertFalse(prediction)


class TestFlaskIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flask_process = subprocess.Popen(['python', 'app.py']) 
        time.sleep(10)
        cls.test_df = pd.read_csv(r"D:\Applied ML\ASSINGMENT_3\Data\train.csv")      
        


    def test_flask(self):
        r = random.randint(0, len(self.test_df) - 1)
        text = self.test_df.iat[r, 0]
        data = {'text': text}
        response = requests.post('http://127.0.0.1:5000/score', json=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('prediction', response.json())
        self.assertIn('propensity', response.json())
        
    @classmethod
    def tearDownClass(cls):
        cls.flask_process.terminate()

if __name__ == '__main__':
    unittest.main()
