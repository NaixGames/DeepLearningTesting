from FFNN import FFNN
import torch
import sys 

sys.path.append("../Utils")
import Functions

class NNSerializer:
	def __init__(self, neural_network: FFNN):
		self.neural_network = neural_network
	
	def save_weights(self, path: str) -> None:
		torch.save(self.neural_network.layers_weights, path + "weights.pt")
		torch.save(self.neural_network.layers_bias, path + "bias.pt")

	def load_weights(self, path: str) -> None:          
		torch.serialization.add_safe_globals([torch.nn.modules.container.ParameterList])
					
		loaded_weights = torch.load(path + "weights.pt")
		loaded_bias =  torch.load(path + "bias.pt")

		#check dimensions align before loading
		assert(len(loaded_weights) == len(self.neural_network.layers_weights))
		assert(len(loaded_bias) == len(self.neural_network.layers_bias))

		self.neural_network.layers_weights = loaded_weights
		self.neural_network.layers_bias = loaded_bias



