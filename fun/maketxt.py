import pandas as pd
import yaml
import os
import numpy as np
import csv

listA = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]

for i in range(len(listA)):
    f = open("filename.csv", 'a', newline='')
    writer = csv.writer(f)
    writer.writerow(listA[i])
    f.close()
def ImagePath(cfg):
    samplepath=cfg['metadata']
    data = pd.read_excel(samplepath)
    for i in range(len(data[cfg['sampleinfo']].values)):
        minpath=data[cfg['lableinfo']].values[i]
        