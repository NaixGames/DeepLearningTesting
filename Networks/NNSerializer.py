from FFNN import FFNN
import torch
import sys 

sys.path.append("../Utils")
import Functions

class NNSerializer:
	def __init__(self, neural_network: FFNN):
		self.neural_network = neural_network
	
	def save_params(self, path: str) -> None:
		torch.save(self.neural_network.layers_weights, path + "weights.pt")
		torch.save(self.neural_network.layers_bias, path + "bias.pt")

	def save_training_data(self, path: str) -> None:
		torch.save(self.neural_network.bn_gamma, path + "BN_Gamma.pt")
		torch.save(self.neural_network.bn_beta, path + "BN_Beta.pt")
		torch.save(self.neural_network.bn_running_mean, path + "BN_Running_Mean.pt")
		torch.save(self.neural_network.bn_running_var, path + "BN_Running_Var.pt")

	def load_params(self, path: str) -> None:          
		torch.serialization.add_safe_globals([torch.nn.modules.container.ParameterList])
					
		loaded_weights = torch.load(path + "weights.pt")
		loaded_bias =  torch.load(path + "bias.pt")

		#check dimensions align before loading
		assert(len(loaded_weights) == len(self.neural_network.layers_weights))
		assert(len(loaded_bias) == len(self.neural_network.layers_bias))

		self.neural_network.layers_weights = loaded_weights
		self.neural_network.layers_bias = loaded_bias

	def load_training_data(self, path: str) -> None:

		loaded_bn_gamma = torch.load(path + "BN_Gamma.pt")
		loaded_bn_beta = torch.load(path + "BN_Beta.pt")
		loaded_running_mean = torch.load(path + "BN_Running_Mean.pt")
		loaded_running_var = torch.load(path + "BN_Running_Var.pt")

		assert(len(loaded_bn_gamma) == len(self.neural_network.bn_gamma))
		assert(len(loaded_bn_beta) == len(self.neural_network.bn_beta))
		assert(len(loaded_running_mean) == len(self.neural_network.bn_running_mean))
		assert(len(loaded_running_var) == len(self.neural_network.bn_running_var))

		self.neural_network.bn_gamma = loaded_bn_gamma
		self.neural_network.bn_beta = loaded_bn_beta
		self.neural_network.bn_running_mean = loaded_running_mean
		self.neural_network.bn_running_var = loaded_running_var



