import math
import numpy as np
import random

def _h(i):
	return np.log(i) + 0.5772156649

def _c(n):
	if n > 2:
		return 2*_h(n-1) - 2*(float(n-1)/n)
	if n == 2:
		return 1
	else:
		return 0

def _anomaly_score(dict_scores, n_samples):
	score = np.array(dict_scores).mean()
	score = -score/_c(n_samples)
	return 2**score

class exNode(object):
	def __init__(self, size = 0):
		self.size = size

class inNode(object):
	def __init__(self, splitAttIdx=-1, splitValue=-1, leftNode=None, rightNode=None):
		self.splitAttIdx = splitAttIdx
		self.splitValue = splitValue
		self.leftNode = leftNode
		self.rightNode = rightNode

def iTree_split(dataSet, attr):
	idx = int(random.choice(attr))
	max_value = max(dataSet[:, idx])
	min_value = min(dataSet[:, idx])
	threshold = int(random.uniform(min_value, max_value))
	dataSetR = filterData(dataSet, idx, threshold, ">")
	dataSetL = filterData(dataSet, idx, threshold, "<")
	return dataSetR, dataSetL, idx, threshold

def filterData(dataSet, idx, threshold, flag):
	dataSetP = []
	if flag == '<':
		for v in dataSet:
			if v[idx] < threshold:
				dataSetP.append(v)
	if flag == '>':
		for v in dataSet:
			if v[idx] >= threshold:
				dataSetP.append(v)
	return np.array(dataSetP)

class iTree(object):
	def __init__(self, dataSet, e, max_depth):
		if e >= max_depth or len(dataSet) <= 1:
			exNode_t = exNode(len(dataSet))
			self.node = exNode_t
		else:
			dataSetR, dataSetL, idx, threshold = iTree_split(dataSet, range(len(dataSet[0])))
			node_r = iTree(dataSetR, e+1, max_depth)
			node_l = iTree(dataSetL, e+1, max_depth)
			inNode_t = inNode(idx, threshold, node_l, node_r)
			self.node = inNode_t

def calcPathLength(tree, X):
	if isinstance(tree.node, exNode):
		return 1
	else:
		attIdx = tree.node.splitAttIdx
		value = tree.node.splitValue
		if X[attIdx] < value:
			return calcPathLength(tree.node.leftNode, X) + 1
		else:
			return calcPathLength(tree.node.rightNode, X) + 1
				

class iForest(object):
	def __init__(self, n_estimators=100, sample_size=256, max_depth = 100):
		self.n_estimators = n_estimators
		self.sample_size = sample_size
		self.max_depth = max_depth

	def fit(self, X):
		n_samples, n_features = X.shape
		self.trees = [iTree(np.array(X[np.random.choice(n_samples, self.sample_size, replace=False)]), 0, max_depth=self.max_depth) 
			for i in range(self.n_estimators)]

	def calcPaths(self, X):
		return [calcPathLength(t, X) for t in self.trees]

	def anomalyScore(self, X):
		paths = [calcPathLength(t, X) for t in self.trees]
		return _anomaly_score(paths, self.sample_size)	

if __name__=="__main__":
	data = [[1,1,1,1],[2,2,2,2],[2,2,2,2],[3,3,3,3]]
	data = np.array(data)
	i_forest = iForest(5, 2, 4)
	i_forest.fit(data)
	print i_forest.anomalyScore(np.array([0,0,0,0]))
