import os
import numpy as np

from market_env import MarketEnv
from market_model_builder import MarketPolicyGradientModelBuilder

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class PolicyGradient:

	def __init__(self, env, discount = 0.99, model_filename = None, history_filename = None):
		self.env = env
		self.discount = discount
		self.model_filename = model_filename
		self.history_filename = history_filename

		from keras.optimizers import SGD
		self.model = MarketPolicyGradientModelBuilder(modelFilename).getModel()
		sgd = SGD(lr = 0.1, decay = 1e-6, momentum = 0.9, nesterov = True)
		self.model.compile(loss='mse', optimizer='rmsprop')

	def discount_rewards(self, r):
		discounted_r = np.zeros_like(r)
		running_add = 0
		r = r.flatten()

		for t in reversed(xrange(0, r.size)):
			if r[t] != 0:
				running_add = 0

			running_add = running_add * self.discount + r[t]
			discounted_r[t] = running_add

		return discounted_r

	def train(self, max_episode = 1000000, max_path_length = 200, verbose = 0):
		env = self.env
		model = self.model
		avg_reward_sum = 0.

		for e in xrange(max_episode):
			env.reset()
			observation = env.reset()
			game_over = False
			reward_sum = 0

			inputs = []
			outputs = []
			predicteds = []
			rewards = []

			while not game_over:
				aprob = model.predict(observation)[0]
				inputs.append(observation)
				predicteds.append(aprob)
				
				if aprob.shape[0] > 1:
					action = np.random.choice(self.env.action_space.n, 1, p = aprob / np.sum(aprob))[0]

					y = np.zeros([self.env.action_space.n])
					y[action] = 1.

					outputs.append(y)
				else:
					action = 0 if np.random.uniform() < aprob else 1

					y = [float(action)]
					outputs.append(y)

				observation, reward, game_over, info = self.env.step(action)
				reward_sum += float(reward)

				rewards.append(float(reward))

				if verbose > 0:
					if env.actions[action] == "LONG" or env.actions[action] == "SHORT":
						color = bcolors.FAIL if env.actions[action] == "LONG" else bcolors.OKBLUE
						print "%s:\t%s\t%.2f\t%.2f\t" % (info["dt"], color + env.actions[action] + bcolors.ENDC, reward_sum, info["cum"]) + ("\t".join(["%s:%.2f" % (l, i) for l, i in zip(env.actions, aprob.tolist())]))

			avg_reward_sum = avg_reward_sum * 0.99 + reward_sum * 0.01
			toPrint = "%d\t%s\t%s\t%.2f\t%.2f" % (e, info["code"], (bcolors.FAIL if reward_sum >= 0 else bcolors.OKBLUE) + ("%.2f" % reward_sum) + bcolors.ENDC, info["cum"], avg_reward_sum)
			print toPrint
			if self.history_filename != None:
				os.system("echo %s >> %s" % (toPrint, self.history_filename))


			dim = len(inputs[0])
			inputs_ = [[] for i in xrange(dim)]
			for obs in inputs:
				for i, block in enumerate(obs):
					inputs_[i].append(block[0])
			inputs_ = [np.array(inputs_[i]) for i in xrange(dim)]

			outputs_ = np.vstack(outputs)
			predicteds_ = np.vstack(predicteds)
			rewards_ = np.vstack(rewards)

			discounted_rewards_ = self.discount_rewards(rewards_)
			#discounted_rewards_ -= np.mean(discounted_rewards_)
			discounted_rewards_ /= np.std(discounted_rewards_)

			#outputs_ *= discounted_rewards_
			for i, r in enumerate(zip(rewards, discounted_rewards_)):
				reward, discounted_reward = r

				if verbose > 1:
					print outputs_[i],
				
				#outputs_[i] = 0.5 + (2 * outputs_[i] - 1) * discounted_reward
				if discounted_reward < 0:
					outputs_[i] = 1 - outputs_[i]
					outputs_[i] = outputs_[i] / sum(outputs_[i])
				outputs_[i] = np.minimum(1, np.maximum(0, predicteds_[i] + (outputs_[i] - predicteds_[i]) * abs(discounted_reward)))

				if verbose > 1:
					print predicteds_[i], outputs_[i], reward, discounted_reward

			model.fit(inputs_, outputs_, nb_epoch = 1, verbose = 0, shuffle = True)
			model.save_weights(self.model_filename)

if __name__ == "__main__":
	import sys
	import codecs

	codeListFilename = sys.argv[1]
	modelFilename = sys.argv[2] if len(sys.argv) > 2 else None
	historyFilename = sys.argv[3] if len(sys.argv) > 3 else None

	codeMap = {}
	f = codecs.open(codeListFilename, "r", "utf-8")

	for line in f:
		if line.strip() != "":
			tokens = line.strip().split(",") if not "\t" in line else line.strip().split("\t")
			codeMap[tokens[0]] = tokens[1]

	f.close()

	env = MarketEnv(dir_path = "./data/", target_codes = codeMap.keys(), input_codes = [], start_date = "2010-08-25", end_date = "2015-08-25", sudden_death = -1.0)

	pg = PolicyGradient(env, discount = 0.9, model_filename = modelFilename, history_filename = historyFilename)
	pg.train(verbose = 1)
