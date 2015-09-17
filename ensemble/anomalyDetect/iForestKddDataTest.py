from IsolationForest import iForest
import numpy as np
import sys

def readKddData(dataPath="filename"):
	dataList = list()
	labels = list()
	categorySet = set([1,2,3,41])
	for line in open(dataPath, 'r'):
		segs = line.split(",")
		tmpList = [float(value) for idx,value in enumerate(segs) if idx not in categorySet]
		dataList.append(tmpList)
		labels.append(segs[-1].replace("\.", ""))
	return np.array(dataList), labels

if __name__=="__main__":
	train_data,train_labels = readKddData(sys.argv[1])
	i_forest = iForest()
	i_forest.fit(train_data)
	test_data, test_labels = readKddData(sys.argv[2])
	for idx in range(len(test_data)):
		print i_forest.anomalyScore(test_data[idx]),test_labels[idx]
