import pickle


with open('x.pkl', 'rb') as f:
    x = pickle.load(f)
print(x)