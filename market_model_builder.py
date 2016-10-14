from deeplearning_assistant.model_builder import AbstractModelBuilder

class MarketPolicyGradientModelBuilder(AbstractModelBuilder):

	def buildModel(self):
		from keras.models import Model
		from keras.layers import merge, Convolution2D, MaxPooling2D, Input, Dense, Flatten, Dropout, Reshape, TimeDistributed, BatchNormalization, Merge, merge
		from keras.layers.advanced_activations import LeakyReLU

		B = Input(shape = (3,))
		b = Dense(5, activation = "relu")(B)

		inputs = [B]
		merges = [b]

		for i in xrange(1):
			S = Input(shape=[2, 60, 1])
			inputs.append(S)

			h = Convolution2D(2048, 3, 1, border_mode = 'valid')(S)
			h = LeakyReLU(0.001)(h)
			h = Convolution2D(2048, 5, 1, border_mode = 'valid')(S)
			h = LeakyReLU(0.001)(h)
			h = Convolution2D(2048, 10, 1, border_mode = 'valid')(S)
			h = LeakyReLU(0.001)(h)
			h = Convolution2D(2048, 20, 1, border_mode = 'valid')(S)
			h = LeakyReLU(0.001)(h)
			h = Convolution2D(2048, 40, 1, border_mode = 'valid')(S)
			h = LeakyReLU(0.001)(h)

			h = Flatten()(h)
			h = Dense(512)(h)
			h = LeakyReLU(0.001)(h)
			merges.append(h)

			h = Convolution2D(2048, 60, 1, border_mode = 'valid')(S)
			h = LeakyReLU(0.001)(h)

			h = Flatten()(h)
			h = Dense(512)(h)
			h = LeakyReLU(0.001)(h)
			merges.append(h)

		m = merge(merges, mode = 'concat', concat_axis = 1)
		m = Dense(1024)(m)
		m = LeakyReLU(0.001)(m)
		m = Dense(512)(m)
		m = LeakyReLU(0.001)(m)
		m = Dense(256)(m)
		m = LeakyReLU(0.001)(m)
		V = Dense(2, activation = 'softmax')(m)
		model = Model(input = inputs, output = V)

		return model

class MarketModelBuilder(AbstractModelBuilder):
	
	def buildModel(self):
		from keras.models import Model
		from keras.layers import merge, Convolution2D, MaxPooling2D, Input, Dense, Flatten, Dropout, Reshape, TimeDistributed, BatchNormalization, Merge, merge
		from keras.layers.advanced_activations import LeakyReLU

		dr_rate = 0.0

		B = Input(shape = (3,))
		b = Dense(5, activation = "relu")(B)

		inputs = [B]
		merges = [b]

		for i in xrange(1):
			S = Input(shape=[2, 60, 1])
			inputs.append(S)

			h = Convolution2D(64, 3, 1, border_mode = 'valid')(S)
			h = LeakyReLU(0.001)(h)
			h = Convolution2D(128, 5, 1, border_mode = 'valid')(S)
			h = LeakyReLU(0.001)(h)
			h = Convolution2D(256, 10, 1, border_mode = 'valid')(S)
			h = LeakyReLU(0.001)(h)
			h = Convolution2D(512, 20, 1, border_mode = 'valid')(S)
			h = LeakyReLU(0.001)(h)
			h = Convolution2D(1024, 40, 1, border_mode = 'valid')(S)
			h = LeakyReLU(0.001)(h)

			h = Flatten()(h)
			h = Dense(2048)(h)
			h = LeakyReLU(0.001)(h)
			h = Dropout(dr_rate)(h)
			merges.append(h)

			h = Convolution2D(2048, 60, 1, border_mode = 'valid')(S)
			h = LeakyReLU(0.001)(h)

			h = Flatten()(h)
			h = Dense(4096)(h)
			h = LeakyReLU(0.001)(h)
			h = Dropout(dr_rate)(h)
			merges.append(h)

		m = merge(merges, mode = 'concat', concat_axis = 1)
		m = Dense(1024)(m)
		m = LeakyReLU(0.001)(m)
		m = Dropout(dr_rate)(m)
		m = Dense(512)(m)
		m = LeakyReLU(0.001)(m)
		m = Dropout(dr_rate)(m)
		m = Dense(256)(m)
		m = LeakyReLU(0.001)(m)
		m = Dropout(dr_rate)(m)
		V = Dense(2, activation = 'linear', init = 'zero')(m)
		model = Model(input = inputs, output = V)

		return model
