import csv
import time
import datetime
import math
import pandas as pd 
import threadpool

# f=open("./tianchi_fresh_comp_train_user.csv")
# context=f.readlines()

import numpy as np

d1=datetime.datetime.strptime("2014-11-18 0","%Y-%m-%d %H")

# ii=0
# gg=10

def do_conv(index):
	train_day29=[]
	offline_candidate_day30=[]
	online_candidate_day31=[]
	ui_dict=[{} for i in range(4)]
	ui_buy={}

	path="./tianchi_fresh_comp_train_user_"+str(index)+".csv"
	with open(path,'rb') as context:
		for line in context:
			line=line.replace("\r\n","")
			line=line.replace("\n","")
			array=line.split(',')

			if array[0]=='user_id':
				continue
			d2=datetime.datetime.strptime(array[-1],"%Y-%m-%d %H")
			day= int((d2-d1).days)
			# print ['day=',day]
			# print "day= %d" % day
			uid=(array[0],array[1],day+1)
			if day >= 24 and day <=28:
				train_day29.append(uid)
			if day >= 25 and day <=29 :
				offline_candidate_day30.append(uid)
			if day >= 26 and day <=30:
				online_candidate_day31.append(uid)
			if day == 31:
				print "crazy thing happen"
			uid2=(array[0],array[1],day)
			actiontype= int(array[2])-1
			if uid in ui_dict[actiontype]:
				ui_dict[actiontype][uid] +=1
			else:
				ui_dict[actiontype][uid] =1

			if array[2]=='4':
				ui_buy[uid] =1
	print "we are done here in file:"+str(index)
	print len(ui_dict[0])

	train_day29=list(set(train_day29))
	offline_candidate_day30=list(set(offline_candidate_day30))
	online_candidate_day31=list(set(online_candidate_day31))
	print "training item number : %d \t" % len(train_day29)
	

	SevenDay_dict_29=[{} for i in range(4)]
	for uid in train_day29:
		last_uid=(uid[0],uid[1],uid[2]-1)
		for i in range(4):
			buyPair=(last_uid[0],last_uid[1])

			if last_uid in ui_dict[i]:
				if buyPair in SevenDay_dict_29[i]:
					SevenDay_dict_29[i][buyPair] += ui_dict[i][last_uid]
				else:
					SevenDay_dict_29[i][buyPair] = ui_dict[i][last_uid]
			else:
				if buyPair in SevenDay_dict_29[i]:
					pass
				else:
					SevenDay_dict_29[i][buyPair] = 0

	SevenDay_dict_30=[{} for i in range(4)]
	for uid in offline_candidate_day30:
		last_uid=(uid[0],uid[1],uid[2]-1)
		for i in range(4):
			buyPair=(uid[0],uid[1])

			if last_uid in ui_dict[i]:
				if buyPair in SevenDay_dict_30[i]:
					SevenDay_dict_30[i][buyPair] += ui_dict[i][last_uid]
				else:
					SevenDay_dict_30[i][buyPair] = ui_dict[i][last_uid]
			else:
				if buyPair in SevenDay_dict_30[i]:
					pass
				else:
					SevenDay_dict_30[i][buyPair] = 0

	SevenDay_dict_31=[{} for i in range(4)]
	for uid in online_candidate_day31:
		last_uid=(uid[0],uid[1],uid[2]-1)
		for i in range(4):
			buyPair=(uid[0],uid[1])

			if last_uid in ui_dict[i]:
				if buyPair in SevenDay_dict_31[i]:
					SevenDay_dict_31[i][buyPair] += ui_dict[i][last_uid]
				else:
					SevenDay_dict_31[i][buyPair] = ui_dict[i][last_uid]
			else:
				if buyPair in SevenDay_dict_31[i]:
					pass
				else:
					SevenDay_dict_31[i][buyPair] = 0

	del train_day29,offline_candidate_day30,online_candidate_day31,ui_dict
	'''
	# get train X ,y 
	X=np.zeros((len(SevenDay_dict_29[0]),4))
	y=np.zeros((len(SevenDay_dict_29[0]),))
	print "size----"
	print np.shape(X)
	print np.shape(y)


	id=0
	for pair in SevenDay_dict_29[0]:
		uid=(pair[0],pair[1],int(29))
		for i in range(4):
			X[id][i]=math.log1p(SevenDay_dict_29[i][pair])
		y[id] = 1 if uid in ui_buy else 0
		id += 1


	pX=np.zeros((len(SevenDay_dict_30[0]),4))

	id=0
	for pair in SevenDay_dict_30[0]:
		uid=(pair[0],pair[1],int(30))
		for i in range(4):
			pX[id][i]=math.log1p(SevenDay_dict_30[i][pair])
		id += 1


	pXtest=np.zeros((len(SevenDay_dict_31[0]),4))

	id=0
	for pair in SevenDay_dict_31[0]:
		uid=(pair[0],pair[1],int(31))
		for i in range(4):
			pXtest[id][i]=math.log1p(SevenDay_dict_31[i][pair])
		id += 1

	print "X= ", X, '\n\n','y= ', y
	print '------------------------\n\n'
	print 'train number= ', len(y), 'positive number= ', sum(y), '\n'
	'''
	# traindata_X=pd.DataFrame(X)
	# traindata_X.to_csv("trainning_data_X_"+str(index)+".csv",header=False,index=False)

	# traindata_Y=pd.DataFrame(y)
	# traindata_Y.to_csv("trainning_data_y_"+str(index)+".csv",header=False,index=False)

	# traindata_pX=pd.DataFrame(pX)
	# traindata_pX.to_csv("trainning_data_pX_"+str(index)+".csv",header=False,index=False)

	# traindata_pXtest=pd.DataFrame(pXtest)
	# traindata_pXtest.to_csv("trainning_data_pXtest_"+str(index)+".csv",header=False,index=False)

	SevenDay_dict_29_dump=pd.DataFrame(SevenDay_dict_29[0].keys())
	SevenDay_dict_29_dump.to_csv("SevenDay_dict_29_dump_"+str(index)+".csv",header=False,index=False)

	SevenDay_dict_30_dump=pd.DataFrame(SevenDay_dict_30[0].keys())
	SevenDay_dict_30_dump.to_csv("SevenDay_dict_30_dump_"+str(index)+".csv",header=False,index=False)

	SevenDay_dict_31_dump=pd.DataFrame(SevenDay_dict_31[0].keys())
	SevenDay_dict_31_dump.to_csv("SevenDay_dict_31_dump_"+str(index)+".csv",header=False,index=False)

	print "thread "+str(index)+" is done !"



def do_merge(totalsepnum):
	


	# ptitle=pd.DataFrame(title)
	# ptitle.to_csv("trainning_data_y_all.csv",index=False, header=False)

	# title=[['browse','collect','market','buy']]
	# ptitle=pd.DataFrame(title)

	# ptitle.to_csv("trainning_data_X_all.csv",index=False, header=False)
	# ptitle.to_csv("trainning_data_pX_all.csv",index=False, header=False)
	# ptitle.to_csv("trainning_data_pXtest_all.csv",index=False, header=False)

	# df=pd.read_csv("trainning_data_y_"+str(1)+".csv")
	# df.to_csv("trainning_data_y_all.csv",index=False, mode='a+')
	df1=pd.read_csv("trainning_data_X_"+str(1)+".csv")
	df1.to_csv("trainning_data_X_all.csv",index=False)

	for index in range(2,totalsepnum+1):
		df=pd.read_csv("trainning_data_y_"+str(index)+".csv")
		df.to_csv("trainning_data_y_all.csv",index=False, mode='a+')

		df1=pd.read_csv("trainning_data_X_"+str(index)+".csv")
		df1.to_csv("trainning_data_X_all.csv",index=False, mode='a+')

		df2=pd.read_csv("trainning_data_pX_"+str(index)+".csv")
		df2.to_csv("trainning_data_pX_all.csv",index=False, mode='a+')

		df3=pd.read_csv("trainning_data_pXtest_"+str(index)+".csv")
		df3.to_csv("trainning_data_pXtest_all.csv",index=False, mode='a+')

def do_merge_dump(totalsepnum):

	df=pd.read_csv("SevenDay_dict_29_dump_"+str(1)+".csv")
	df.to_csv("SevenDay_dict_29_dump_all.csv",index=False, mode='a+')
	for index in range(2,totalsepnum+1):
		df=pd.read_csv("SevenDay_dict_29_dump_"+str(index)+".csv")
		df.to_csv("SevenDay_dict_29_dump_all.csv",index=False, mode='a+')



def predict():
	X=pd.read_csv('trainning_data_X_all.csv')
	X=np.array(X)
	

if __name__ == '__main__':

	# totalsepnum=47
	# param_list = range(1, totalsepnum+1)
	# pool = threadpool.ThreadPool(4)
	# requests = threadpool.makeRequests(do_conv, param_list)
	# [pool.putRequest(req) for req in requests]
	# pool.wait()
	# print "all thread is done"
	# predict()

	do_merge(2)
	# do_merge_dump(2)



		# ii=ii+1
		# if (ii>gg):
		# 	break



# print "training item number : %d \t" % len(train_day29)

# import math


'''
ii=0
#for feature, for this demo, sum of 4 operations
ui_dict=[{} for i in range(4)]
ui_buy={}

with open("./tianchi_fresh_comp_train_user.csv",'rb') as context:
	for line in context:
		line =line.replace("\r\n","")
		array=line.split(',')
		if array[0]=="user_id":
			continue
		d2=datetime.datetime.strptime(array[-1],"%Y-%m-%d %H")
		day= int((d2-d1).days)
		uid= (array[0],array[1],day)
		actiontype= int(array[2])-1
		if uid in ui_dict[actiontype]:
			ui_dict[actiontype][uid] +=1
		else:
			ui_dict[actiontype][uid] =1

		if array[2]=='4':
			ui_buy[uid] =1
		# ii=ii+1
		# if (ii>gg):
		# 	break
print "we are done here"
print len(ui_dict[0])


#for label

# for line in context:
# 	line =line.replace("\r\n","")
# 	array=line.split(',')
# 	if array[0]=="user_id":
# 		continue
# 	d2=datetime.datetime.strptime(array[-1],"%Y-%m-%d %H")
# 	uid= (array[0],array[1],int((d2-d1).days))

# 	if array[2]=='4':
# 		ui_buy[uid] =1
	# ii=ii+1
	# if (ii>3*gg):
		# break

wf=open('offline_groundtruth.txt','w')
wf.write('user_id,item_id\n')
for key in ui_buy:
	if key[2]==30:
		wf.write('%s,%s \n'%(key[0],key[1]))




SevenDay_dict_29=[{} for i in range(4)]
id=0
for uid in train_day29:
	last_uid=(uid[0],uid[1],uid[2]-1)
	for i in range(4):
		buyPair=(last_uid[0],last_uid[1])

		if last_uid in ui_dict[i]:
			if buyPair in SevenDay_dict_29[i]:
				SevenDay_dict_29[i][buyPair] += ui_dict[i][last_uid]
			else:
				SevenDay_dict_29[i][buyPair] = ui_dict[i][last_uid]
		else:
			if buyPair in SevenDay_dict_29[i]:
				pass
			else:
				SevenDay_dict_29[i][buyPair] = 0


SevenDay_dict_30=[{} for i in range(4)]
id=0
for uid in offline_candidate_day30:
	last_uid=(uid[0],uid[1],uid[2]-1)
	for i in range(4):
		buyPair=(uid[0],uid[1])

		if last_uid in ui_dict[i]:
			if buyPair in SevenDay_dict_30[i]:
				SevenDay_dict_30[i][buyPair] += ui_dict[i][last_uid]
			else:
				SevenDay_dict_30[i][buyPair] = ui_dict[i][last_uid]
		else:
			if buyPair in SevenDay_dict_30[i]:
				pass
			else:
				SevenDay_dict_30[i][buyPair] = 0

SevenDay_dict_31=[{} for i in range(4)]
id=0
for uid in online_candidate_day31:
	last_uid=(uid[0],uid[1],uid[2]-1)
	for i in range(4):
		buyPair=(uid[0],uid[1])

		if last_uid in ui_dict[i]:
			if buyPair in SevenDay_dict_31[i]:
				SevenDay_dict_31[i][buyPair] += ui_dict[i][last_uid]
			else:
				SevenDay_dict_31[i][buyPair] = ui_dict[i][last_uid]
		else:
			if buyPair in SevenDay_dict_31[i]:
				pass
			else:
				SevenDay_dict_31[i][buyPair] = 0
# print "0:/n"
# print SevenDay_dict_29[0]
# print "1:/n"
# print SevenDay_dict_29[1]
# print "2:/n"
# print SevenDay_dict_29[2]
# print "3:/n"
# print SevenDay_dict_29[3]
# print train_day29
# print buyPair
# buyPair= ('10001082','282816229')
# print SevenDay_dict_29[0][buyPair]
# print SevenDay_dict_29[1][buyPair]
# print SevenDay_dict_29[2][buyPair]
# print SevenDay_dict_29[3][buyPair]


# get train X ,y 
X=np.zeros((len(SevenDay_dict_29[0]),4))
y=np.zeros((len(SevenDay_dict_29[0]),))
print "size----"
print np.shape(X)
print np.shape(y)


id=0
for pair in SevenDay_dict_29[0]:
	uid=(pair[0],pair[1],int(29))
	for i in range(4):
		X[id][i]=math.log1p(SevenDay_dict_29[i][pair])
	y[id] = 1 if uid in ui_buy else 0
	id += 1


pX=np.zeros((len(SevenDay_dict_30[0]),4))

id=0
for pair in SevenDay_dict_30[0]:
	uid=(pair[0],pair[1],int(30))
	for i in range(4):
		pX[id][i]=math.log1p(SevenDay_dict_30[i][pair])
	id += 1


pXtest=np.zeros((len(SevenDay_dict_31[0]),4))

id=0
for pair in SevenDay_dict_31[0]:
	uid=(pair[0],pair[1],int(31))
	for i in range(4):
		pXtest[id][i]=math.log1p(SevenDay_dict_31[i][pair])
	id += 1

print ['len31=',len(SevenDay_dict_31[0])]
# id=0
# for uid in train_day29:
# 	last_uid=(uid[0],uid[1],uid[2]-1)
# 	for i in range(4):
# 		X[id][i]=math.log1p(ui_dict[i][last_uid]  if last_uid in ui_dict[i] else 0 )
# 	y[id] = 1 if uid in ui_buy else 0
# 	id += 1

# log1p(x)=log(1+x)

print "X= ", X, '\n\n','y= ', y
print '------------------------\n\n'
print 'train number= ', len(y), 'positive number= ', sum(y), '\n'


# rate=1.0*sum(y)/len(y)
# print "rate= %f" % rate
# #get predict pX for offline_candidate_day30
# pX=np.zeros((len(offline_candidate_day30),4))
# id=0
# for uid in offline_candidate_day30:
# 	last_uid=(uid[0],uid[1],uid[2]-1)
# 	for i in range(4):
# 		pX[id][i]=math.log1p(ui_dict[i][last_uid] if  last_uid in ui_dict[i] else 0)
# 	id +=1

# #get predict pX for online_candidate_day31
# pXtest=np.zeros((len(online_candidate_day31),4))
# id=0
# for uid in online_candidate_day31:
# 	last_uid=(uid[0],uid[1],uid[2]-1)
# 	for i in range(4):
# 		pXtest[id][i]=math.log1p(ui_dict[i][last_uid] if  last_uid in ui_dict[i] else 0)
# 	id +=1







###### training ##############

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X,y)

### evaluate########################
py=model.predict_proba(pX)
npy=[]

for a in py:
	npy.append(a[1])
py=npy



#combine
lx=zip(SevenDay_dict_30[0],py)
print '------------------------'
lx=sorted(lx,key=lambda x:x[1],reverse=True)

# outnum=int(len(lx)*rate)
# print "outnum=%d" %outnum

wf=open('ans.csv','w')
wf.write('user_id,item_id\n')
for i in range(2734):
	item=lx[i]
	wf.write('%s,%s \n'%(item[0][0],item[0][1]))


#---------------------------||||||-----------------------------------------------

### evaluate########################
pytest=model.predict_proba(pXtest)

# print " original pytest="
# print pytest


npytest=[]
for a in pytest:
	npytest.append(a[1])
pytest=npytest

# print "pytest="
# print pytest
#combine
lxt=zip(SevenDay_dict_30[0],pytest)
print '------------------------'
lxt=sorted(lxt,key=lambda x:x[1],reverse=True)


# print "lxt="
# print lxt

# outnum=int(len(lxt)*rate)
# print "outnum=%d" %outnum

writeCSV=csv.writer(open("tianchi_mobile_recommendation_predict.csv",'w'))
writeCSV.writerow(['user_id','item_id'])

for i in range(27340):
	itemt=lxt[i]
	writeCSV.writerow([itemt[0][0],itemt[0][1]])
'''