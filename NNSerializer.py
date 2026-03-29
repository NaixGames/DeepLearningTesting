from FFNN import FFNN


class NNSerializer:
    def __init__(self, neural_network: FFNN):
        self.neural_network = neural_network
	
    def save_weights(self, path: str) -> None:
        #Need to iterate through params and save them.
        #Note I will not save the activations functiosn themselves, but then it needs some consitency
        pass

    def load_weights(self, path: str) -> None:
        #Need to iterate through params and try to fetch the weights from some serialize file
        pass