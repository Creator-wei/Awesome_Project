import glob
from tqdm import tqdm
import os
import csv

commands = ['yes', 'no', 'up', 'down', 'left', 'right','one','two','three','four','five','six','seven','eight','nine','zero']

file_path = './DataSet.csv'


csvFile = open(file_path,"w")
writer = csv.writer(csvFile)
writer.writerow(["file_name","Label"])

for i in tqdm(range(len(commands))):
    path = "./data/%s/*.wav" % commands[i]
    file = glob.glob(path)
    for j in range(500):
        try:
            name = file[j].replace("\\","/")
            writer.writerow([name,str(i)])
        except:
            exit()
