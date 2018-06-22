import csv
import pandas as pd

title=[['browse','collect','market','buy']]
ptitle=pd.DataFrame(title)

ptitle.to_csv("trainning_data_X_all.csv",index=False, header=False)


for index in range(1,6):

	df=pd.read_csv("trainning_data_X_"+str(index)+".csv")
	df.to_csv("trainning_data_X_all.csv",index=False, header=False, mode='a+')

