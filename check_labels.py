import pickle
import os

try:
    lb = pickle.load(open('label_transform.pkl', 'rb'))
    print("Classes found in label binarizer:")
    for i, label in enumerate(lb.classes_):
        print(f"{i}: {label}")
except Exception as e:
    print(f"Error loading label_transform.pkl: {e}")
