# Prototype
class Hidden_States_Network:
    weights = []
    features = []
    bias = []
    position = None
    max_range = None
    def forward():
        if position >= max_range
            return
        features[position+=1] = self.feature[position] * self.weights[position] + self.bias[position]
        forward()
    def backpropgation(output: int, learning_rate: int):
        for i in range(max_range):
            delta_feature = self.features[i+=1] - output
            delta_weights = self.weights[i] - output
            delta_bias = self.bias[i] - output
            gradient_feature = delta_feature/output
            gradient_weights = delta_weights/output
            gradient_bias = delta_bias/output
            self.features[i] = self.features[i] - learning_rate * gradient_feature
            self.weights[i] = self.weights[i] - learning_rate * gradient_weights
            self.bias[i] = self.bias[i] - learning_rate * gradient_bias
def reflection_function(vector: [], network_output: []):
    gradient_1 = sum(vector)/len(vector)
    gradient_2 = sum(network_output)/len(network_output)
    delta = (-gradient_1 + -gradient_2)
    output = delta/sum(network_output)
    return output

    
