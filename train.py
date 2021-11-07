#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import torch
import torch.nn as nn
import librosa 
import numpy as np
import pylab
import os
import librosa.display
from tqdm.notebook import tqdm
import warnings
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn.functional as F
import torch.optim as optim
from audio2numpy import open_audio
from scipy.signal import hilbert
import signal_envelope as se
warnings.filterwarnings("ignore")

#Read Dataset
with open('metadata.json') as f:
  data = json.load(f)
  
#Smooth values
def movmean(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

#Feature extraction
areas  = []
asymptomatic = []
covid19      = []
verified     = []

for key in tqdm(data):
  if 'filename' not in key:
    continue
  path = key['filename']
  name = path.replace(".mp3", "")
  try:
    signal, sampling_rate = open_audio(os.path.join("raw",path))
  except:
    print(0)
    continue

  if 'asymptomatic' in key:
    asymptomatic.append(int(key['asymptomatic']))
  else:
    asymptomatic.append(2)

  if 'covid19' in key:  
    covid19.append(int(key['covid19']))
  else:
    covid19.append(2)

  if 'verified' in key:   
    verified.append(int(key['verified']))
  else:
    verified.append(2)

  correct = None
  if len(signal.shape) == 1:
    correct = signal
  else:
    if np.sum(np.var(signal[:,1])) > np.sum(np.var(signal[:,0])):
      correct = signal[:,1]
    else:
      correct = signal[:,0]

  correct = movmean(correct,3)
  area = np.sum(np.abs(hilbert(correct)))
  areas.append(area)
  
areas, asymptomatic, covid19, verified = np.array(areas), np.array(asymptomatic), np.array(covid19), np.array(verified)   

#Use LR to find threshold
def train(X, Y):
  X = np.array(X).reshape(-1,1)
  Y = np.array(Y).reshape(-1)

  from sklearn.linear_model import LogisticRegression
  neigh = LogisticRegression(max_iter = 1000)
  neigh.fit(X, Y)

  from sklearn.metrics import classification_report,accuracy_score,auc,roc_curve,cohen_kappa_score
  print(classification_report(Y, neigh.predict(X)))
  print(accuracy_score(Y, neigh.predict(X)))
  fpr, tpr, thresholds = roc_curve(Y, neigh.predict(X))
  print(auc(fpr,tpr))
  print(cohen_kappa_score(Y,neigh.predict(X)))
  
#Experiments
from sklearn.metrics import classification_report,accuracy_score,auc,roc_curve,cohen_kappa_score
from sklearn.linear_model import LogisticRegression
  
mask = np.logical_and(covid19 !=2,verified !=2)
mask = np.logical_and(mask, covid19 == verified )
X, Y = areas[mask], verified[mask]
X = np.array(X).reshape(-1,1)
Y = np.array(Y).reshape(-1)
neigh = LogisticRegression(max_iter = 1000)
neigh.fit(X, Y)
accuracy_score(Y, neigh.predict(X))
(0.5 - neigh.intercept_) / neigh.coef_


print(classification_report(Y, X <= 2000))
print(accuracy_score(Y, X <= 2000))
fpr, tpr, thresholds = roc_curve(Y, X <= 2000)
print(auc(fpr,tpr))
print(cohen_kappa_score(Y,X <= 2000))
print("---")

print(classification_report(Y, X <= 3000))
print(accuracy_score(Y, X <= 3000))
fpr, tpr, thresholds = roc_curve(Y, X <= 3000)
print(auc(fpr,tpr))
print(cohen_kappa_score(Y,X <= 3000))
print("---")

print(classification_report(Y, X <= 4000))
print(accuracy_score(Y, X <= 4000))
fpr, tpr, thresholds = roc_curve(Y, X <= 4000)
print(auc(fpr,tpr))
print(cohen_kappa_score(Y,X <= 4000))
print("---")


[0.91,0.96,0.93,0.86,0.93]
[0.97,0.92,0.94,0.88,0.94]
[0.99,0.86,0.92,0.84,0.93]

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

fig = plt.figure(figsize=(3,4))
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
plt.bar(np.array(range(5)) * 0.3,[0.91,0.96,0.93,0.86,0.93],width = 0.2, color = ['green', 'darkorange', 'purple', 'deeppink','darkred'])
plt.ylim([0.7,1.05])
plt.yticks([0.7,0.8,0.9])

colors = ['green', 'darkorange', 'purple', 'deeppink','darkred']
names  = ['Sensitivity','Specificity','Accuracy','Kappa coefficient','Area under ROC curve']
cmap = dict(zip(names, colors))
patches = [Patch(color=v, label=k) for k, v in cmap.items()]
plt.legend(handles=patches, bbox_to_anchor=(1.04, 0.5), loc='center left', borderaxespad=0, ncol = 5)

for index, value in enumerate([0.91,0.96,0.93,0.86,0.93]):
    plt.text(index * 0.3 - 0.1, value + 0.01, str(value))
plt.tight_layout()


[0.91,0.96,0.93,0.86,0.93]
[0.97,0.92,0.94,0.88,0.94]
[0.99,0.86,0.92,0.84,0.93]

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

fig = plt.figure(figsize=(3,4))
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
plt.bar(np.array(range(5)) * 0.3,[0.97,0.92,0.94,0.88,0.94],width = 0.2, color = ['green', 'darkorange', 'purple', 'deeppink','darkred'])
plt.ylim([0.7,1.05])
plt.yticks([10],color='w')

colors = ['green', 'darkorange', 'purple', 'deeppink','darkred']
names  = ['Sensitivity','Specificity','Accuracy','Kappa coefficient','Area under ROC curve']
cmap = dict(zip(names, colors))
patches = [Patch(color=v, label=k) for k, v in cmap.items()]
plt.legend(handles=patches, bbox_to_anchor=(1.04, 0.5), loc='center left', borderaxespad=0, ncol = 5)

for index, value in enumerate([0.97,0.92,0.94,0.88,0.94]):
    plt.text(index * 0.3 - 0.1, value + 0.01, str(value))
plt.tight_layout()

[0.91,0.96,0.93,0.86,0.93]
[0.97,0.92,0.94,0.88,0.94]
[0.99,0.86,0.92,0.84,0.93]

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

fig = plt.figure(figsize=(3,4))
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)

plt.bar(np.array(range(5)) * 0.3,[0.99,0.86,0.92,0.84,0.93],width = 0.2, color = ['green', 'darkorange', 'purple', 'deeppink','darkred'])
plt.ylim([0.7,1.05])
plt.yticks([10],color='w')

colors = ['green', 'darkorange', 'purple', 'deeppink','darkred']
names  = ['Sensitivity','Specificity','Accuracy','Kappa coefficient','Area under ROC curve']
cmap = dict(zip(names, colors))
patches = [Patch(color=v, label=k) for k, v in cmap.items()]
plt.legend(handles=patches, bbox_to_anchor=(1.04, 0.5), loc='center left', borderaxespad=0, ncol = 5)

for index, value in enumerate([0.99,0.86,0.92,0.84,0.93]):
    plt.text(index * 0.3 - 0.1, value + 0.01, str(value))
plt.tight_layout()

X, Y = areas[covid19 != 2], covid19[covid19 !=2]
X, Y = np.array(X), np.array(Y)
print(np.mean(X[Y == 1]), np.mean(X[Y == 0]),np.std(X[Y == 1]), np.std(X[Y==0]))

X, Y = areas[verified != 2], verified[verified !=2]
X, Y = np.array(X), np.array(Y)
print(np.mean(X[Y == 1]), np.mean(X[Y == 0]),np.std(X[Y == 1]), np.std(X[Y == 0]))

mask = np.logical_and(covid19 !=2,verified !=2)
mask = np.logical_and(mask, covid19 == verified )

X, Y = areas[mask], verified[mask]
X, Y = np.array(X), np.array(Y)
print(np.mean(X[Y == 1]), np.mean(X[Y == 0]),np.std(X[Y == 1]), np.std(X[Y == 0]))


mask = np.logical_and(covid19 !=2,verified !=2)
mask = np.logical_and(mask, covid19 == verified )
mask = np.logical_and(mask,  np.logical_or(asymptomatic == 1, verified == 0))

X, Y = areas[mask], verified[mask]
X, Y = np.array(X), np.array(Y)
print(np.mean(X[Y == 1]), np.mean(X[Y == 0]),np.std(X[Y == 1]), np.std(X[Y == 0]))


mask = np.logical_and(covid19 !=2,verified !=2)
mask = np.logical_and(mask, covid19 == verified )
mask = np.logical_and(mask,  np.logical_or(asymptomatic == 0, verified == 0))

X, Y = areas[mask], verified[mask]
X, Y = np.array(X), np.array(Y)
print(np.mean(X[Y == 1]), np.mean(X[Y == 0]),np.std(X[Y == 1]), np.std(X[Y == 0]))


import pandas
import seaborn as sns
import matplotlib.pyplot as plt
df = pandas.DataFrame([[4058.6317263222168,988.0895589280572,988.0895589280572,943.0214298039596,1028.4490775466522]+[8987.92609708205,10746.046660534295,12655.51678493258,12655.51678493258,12655.51678493258],
              ['Positive'] * 5 + ['Negative'] * 5,
              ['Verbal','Verified','Matched','MatchedAsymp','MatchedSymp','Verbal','Verified','Matched','MatchedAsymp','MatchedSymp']]).transpose()
df.columns = ['y','t','x']


fig = plt.figure(figsize=(10,3))
sns.set_context('paper')
b = sns.barplot(x = 'x', y = 'y', hue = 't',data = df, palette=["C19", "C1"])
b.tick_params(labelsize=12)
plt.axes().set_yscale('log',basey=10)
plt.axes().set_ylim([300,33000])
plt.minorticks_off()
plt.legend(fontsize = 12, ncol = 2)

fig = plt.figure(figsize=(4,3))

mask = np.logical_and(covid19 !=2,verified !=2)
mask = np.logical_and(mask, covid19 == verified )
mask = np.logical_and(mask,  np.logical_or(asymptomatic == 1, verified == 0))

X, Y = areas[mask], verified[mask]
X, Y = np.array(X), np.array(Y)

ay = X[Y == 1]

mask = np.logical_and(covid19 !=2,verified !=2)
mask = np.logical_and(mask, covid19 == verified )
mask = np.logical_and(mask,  np.logical_or(asymptomatic == 0, verified == 0))

X, Y = areas[mask], verified[mask]
X, Y = np.array(X), np.array(Y)

sy = X[Y == 1]

#ax = fig.add_axes(['MatchedAsymp','MatchedSymp'],width = 10,heigh= 10)
bplot = plt.boxplot([ay,sy], widths = 0.8,notch=True,vert=True, patch_artist=True, labels = ['MatchedAsymp','MatchedSymp'],
                    flierprops={'marker': 'o', 'markersize': 1})

colors = ['lightgreen', 'lightyellow']
for patch, color in zip(bplot['boxes'], colors):
      patch.set_facecolor(color)

plt.ylim([0,3000])
plt.yticks([500,1500,2500],fontsize =12)
plt.xticks(fontsize =12)

import scipy
scipy.stats.ttest_ind(ay,sy)

X, Y = areas[covid19 != 2], covid19[covid19 !=2]
X, Y = np.array(X), np.array(Y)
print(scipy.stats.ttest_ind(X[Y==1], X[Y==0]))

X, Y = areas[verified != 2], verified[verified !=2]
X, Y = np.array(X), np.array(Y)
print(scipy.stats.ttest_ind(X[Y==1], X[Y==0]))

mask = np.logical_and(covid19 !=2,verified !=2)
mask = np.logical_and(mask, covid19 == verified )

X, Y = areas[mask], verified[mask]
X, Y = np.array(X), np.array(Y)
print(scipy.stats.ttest_ind(X[Y==1], X[Y==0]))

mask = np.logical_and(covid19 !=2,verified !=2)
mask = np.logical_and(mask, covid19 == verified )
mask = np.logical_and(mask,  np.logical_or(asymptomatic == 1, verified == 0))

X, Y = areas[mask], verified[mask]
X, Y = np.array(X), np.array(Y)
print(scipy.stats.ttest_ind(X[Y==1], X[Y==0]))


mask = np.logical_and(covid19 !=2,verified !=2)
mask = np.logical_and(mask, covid19 == verified )
mask = np.logical_and(mask,  np.logical_or(asymptomatic == 0, verified == 0))

X, Y = areas[mask], verified[mask]
X, Y = np.array(X), np.array(Y)
print(scipy.stats.ttest_ind(X[Y==1], X[Y==0]))

#verbal
train(areas[covid19 != 2], covid19[covid19 !=2])

#verified
train(areas[verified != 2], verified[verified !=2])

#matched
mask = np.logical_and(covid19 !=2,verified !=2)
mask = np.logical_and(mask, covid19 == verified )
train(areas[mask], verified[mask])
sum(mask)

#matched asy
mask = np.logical_and(covid19 !=2,verified !=2)
mask = np.logical_and(mask, covid19 == verified )
mask = np.logical_and(mask,  np.logical_or(asymptomatic == 1, verified == 0))
train(areas[mask], verified[mask])

#matched sy
mask = np.logical_and(covid19 !=2,verified !=2)
mask = np.logical_and(mask, covid19 == verified )
mask = np.logical_and(mask,  np.logical_or(asymptomatic == 0, verified == 0))
train(areas[mask], verified[mask])
