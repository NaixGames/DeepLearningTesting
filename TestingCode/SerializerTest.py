import torch
import sys 

sys.path.append("../Networks")
from FFNN import FFNN
from NNSerializer import NNSerializer
sys.path.append("../Utils")
import Functions

if __name__ == "__main__":
	sample_size = 100
	possible_classes = 5
	features_size = 5
	inner_layers = [100, 200]
	function_array = [Functions.sigmoid, Functions.tanh]

	#Optional parameter for some functions
	beta_swish = 0.5
	alpha_celu = 0.75

	assert(len(inner_layers) == len(function_array))

	#We process the functions
	derivatives_array = []
	for i in range(0,len(function_array)):
		derivatives_array.append(Functions.differentiate(function_array[i]))

	for i in range(0, len(function_array)):
		func = function_array[i]
		#if we need an extra parameter, we "project" the function and the derivative
		#Note we do this after we got the derivatives, mainly because, if not, checking the lambda would give an incorrect check,
		#since the lambda results for "different definitions" is not the same
		if (func == Functions.swish):
			function_array[i] = lambda x : Functions.swish(x, beta_swish)
			derivatives_array[i] = lambda x : Functions.swish_grad(x, beta_swish)
		if (func == Functions.celu):
			function_array[i] = lambda x : Functions.celu(x, alpha_celu)
			derivatives_array[i] = lambda x : Functions.celu_grad(x, alpha_celu)

	net = FFNN(features_size, inner_layers, function_array, derivatives_array, possible_classes)
	print(net.summarize())


	serializer = NNSerializer(net)
	serializer.save_weights("SerializerTest/")
	serializer.load_weights("SerializerTest/")
	print(net.summarize())