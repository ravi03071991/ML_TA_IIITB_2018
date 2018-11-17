import pandas as pd
import numpy as np
import numpy.random as rand
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys


seq2 = pd.Series(np.arange(2))

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')
	print(cm)
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

def plotConfusionMatrix(test_labels,predicted_labels):
	cnf_matrix = confusion_matrix(test_labels, predicted_labels)
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=seq2, normalize=True,title='Normalized confusion matrix')
	plt.show()



class Vaccine(object):
	seq2 = pd.Series(np.arange(2))

	"""docstring for Vaccine"""
	def __init__(self, trainFile, testFile):
		self.trainFile = trainFile
		self.testFile = testFile
		self.__lr = LogisticRegression()
		self.__dtree = DecisionTreeClassifier()
		self.__rforest = RandomForestClassifier()
		self.__svm = SVC(kernel='rbf')
		self.train_data = None
		self.train_labels = None
		self.test_data = None
		self.test_labels = None
		self.predicted_labels = None

	def trainingData(self):
		df = pd.read_csv(self.trainFile)
		df = df.drop(columns='Unnamed: 0')
		df = df.dropna()
		self.train_labels = df['label']
		self.train_data = df.drop(columns='label')

	def testingData(self):
		df = pd.read_csv(self.testFile)
		df = df.drop(columns='Unnamed: 0')
		df = df.dropna()
		self.test_labels = df['label']
		self.test_data = df.drop(columns='label')

	def data(self):
		self.trainingData()
		self.testingData()

	def trainLogisticRegression(self):
		self.__lr.fit(self.train_data,self.train_labels)	
	
	def testLogisticRegression(self):
		self.predicted_labels =  self.__lr.predict(self.test_data)
		print "Logistic Regression score " + str(self.__lr.score(self.test_data, self.test_labels))

	def trainDecesionTree(self):
		self.__dtree.fit(self.train_data,self.train_labels)

	def testDecesionTree(self):
		self.predicted_labels = self.__dtree.predict(self.test_data)
		print "Decesion Tree Score " + str(self.__dtree.score(self.test_data,self.test_labels))
	
	def trainRandomForrest(self):
		self.__rforest.fit(self.train_data,self.train_labels)

	def testRandomForrest(self):
		self.predicted_labels = self.__rforest.predict(self.test_data)
		print "Random Forest Score " + str(self.__rforest.score(self.test_data,self.test_labels))

	def trainSVM(self):
		self.__svm.fit(self.train_data,self.train_labels)

	def testSVM(self):
		self.predicted_labels = self.__svm.predict(self.test_data)
		print "SVM score " + str(self.__svm.score(self.test_data,self.test_labels))
 
if __name__ == "__main__":
	train_data_name = sys.argv[1]
	test_data_name = sys.argv[2]
	model = Vaccine(train_data_name,test_data_name)
	model.data()
	# model.trainLogisticRegression()
	# model.testLogisticRegression()
	# plotConfusionMatrix(model.test_labels,model.predicted_labels)
	
	# model.trainDecesionTree()
	# model.testDecesionTree()

	# model.trainRandomForrest()
	# model.testRandomForrest()

	model.trainSVM()
	model.testSVM()

	# plotConfusionMatrix(model.test_labels,model.predicted_labels)