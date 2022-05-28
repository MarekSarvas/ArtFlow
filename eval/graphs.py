import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure

normal = None
styled = None

with open('normal.json') as json_file:
    normal = json.load(json_file)

with open('styled.json') as json_file:
    styled = json.load(json_file)

with open('wikiart.json') as json_file:
    wikiarted = json.load(json_file)


X = list(styled.keys())

styled_acc = [styled[net]['epoch_acc'] for net in X]
normal_acc = [normal[net]['epoch_acc'] for net in X]
wikiarted_acc = [wikiarted[net]['epoch_acc'] for net in X]


X_axis = np.arange(len(X))
figure(figsize=(15, 5), dpi=80)
plt.bar(X_axis - 0.2, styled_acc, 0.4, label='styled')
plt.bar(X_axis, wikiarted_acc, 0.4, label='wikiart200')
#plt.bar(X_axis + 0.2, normal_acc, 0.4, label='normal')

plt.xticks(X_axis, X)
plt.xlabel("Nets")
plt.ylabel("Percentage")
plt.title("Accuracy ")
plt.legend()
plt.savefig('results.png')
