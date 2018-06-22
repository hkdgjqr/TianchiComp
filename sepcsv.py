
import csv
import pandas as pd 
import numpy as np

print "start seperate csv file"

idx=0
datalist=[]
LINELIMIT=500000;
csvindex=1;
'''
with open("./tianchi_fresh_comp_train_user_trun.csv",'rb') as f:
	for line in f:
		if idx< 50:
			a=line.strip().split(',')
			datalist.append(a)
			idx +=1
		else:

			df=pd.DataFrame(datalist)

			df.to_csv('./tianchi_fresh_comp_train_user_' + 'trun' + '.csv',header=False,index=False)
			break

'''


with open("./tianchi_fresh_comp_train_user.csv",'rb') as f:
	for line in f:
		a=line.strip().split(',')
		datalist.append(a)
			
		if idx< LINELIMIT:
			idx +=1
		else:
			df=pd.DataFrame(datalist)

			df.to_csv('./tianchi_fresh_comp_train_user_' + str(csvindex) + '.csv',header=False,index=False)
			csvindex +=1
			del datalist
			datalist=[]
			idx=0
	df=pd.DataFrame(datalist)
	df.to_csv('./tianchi_fresh_comp_train_user_' + str(csvindex) + '.csv',header=False,index=False)







'''
user = pd.read_csv("./tianchi_fresh_comp_train_user.csv")


dur = len(user)/10  
ifrom = 0  
idx = 0  
while ifrom < len(user):  
    ito = ifrom + dur  
    data = user[ifrom:ito]  
    print("from ", ifrom, "to ", (ito-1), "total", len(data))  
    data.to_csv('./tianchi_fresh_comp_train_user_' + str(idx) + '.csv', index=False)  
    ifrom = ito  
idx += 1
'''