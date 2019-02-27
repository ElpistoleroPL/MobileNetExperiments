import os
import sys
import pickle

print(sys.argv[1])
dict = None
with open(sys.argv[1], 'rb') as f:
        dict  = pickle.load(f)

print("Val acc: %f" %(max(dict[dict.keys()[0]]['val_acc'])))
print("Acc: %f" %(max(dict[dict.keys()[0]]['acc'])))
