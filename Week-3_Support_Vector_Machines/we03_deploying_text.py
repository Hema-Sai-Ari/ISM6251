import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import pickle


mowers_model = pickle.load(open("C:/Users/arihe/Box/SEM-2/DATA SCIENCE PRORAMMING/SIR PYTHON FIES/class-3(More on regression and SVM)/we03_svm.pkl", "rb"))

print("\n*****************************************************")
print("* The lawn mower Prediction Model *")
print("*****************************************************\n")
Income = float(input("Enter the income of the customer: "))
Lot_Size =float(input("Enter the  customer's Lot Size : "))
df = pd.DataFrame({'Income': [Income], 'Lot_Size' : [Lot_Size]})
print(df)
result = mowers_model.predict(df)

ownership = ('Not_owner', 'owner')
probability = mowers_model.predict_proba(df)


print(f"\n The Lawn Mower Prediction Model indicates that the probability of ownership is {probability[0][1]:.4f} and predicted that the customer is {ownership[result[0]]}, of the lawn Mower")