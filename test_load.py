import pickle
import os
import sys

print("Starting load test...")
try:
    if os.path.exists('cnn_model.pkl'):
        print(f"File exists. Size: {os.path.getsize('cnn_model.pkl')}")
        with open('cnn_model.pkl', 'rb') as f:
            print("Opening file...")
            model = pickle.load(f)
            print("Model loaded.")
    else:
        print("File not found.")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("Done.")
