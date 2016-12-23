import numpy as np
from os import path

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout

# This is abstract class. You need to implement yours.
class AbstractModelBuilder(object):

	def __init__(self, weights_path = None):
		self.weights_path = weights_path

	def getModel(self):
		weights_path = self.weights_path
		model = self.buildModel()

		if weights_path and path.isfile(weights_path):
			try:
				model.load_weights(weights_path)
			except Exception, e:
				print e

		return model

	# You need to override this method.
	def buildModel(self):
		raise NotImplementedError("You need to implement your own model.")
