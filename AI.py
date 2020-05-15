import numpy as np

class FullyConnectedLayer:

    def __init__(self, n_input, layer_size, activation_function = 'linear'):
        """Creates a fully connected layer
        Arguments:
            n_input {integer} -- Number of input neurons
            layer_size {integer} -- Number of neurons in layer
        Keyword Arguments:
            activation_function {str} -- Activation function of the layer (default: {'linear'})
        """
        self.weights =  np.random.normal(0, 1/np.sqrt(n_input), (layer_size,n_input))
        self.thresholds = np.zeros((layer_size,))
        self.activation_function = activation_function
        self.shape = (layer_size, n_input)

    def set_weights(self, w):
        """ Manually set the weights of the network
        Arguments:
            w {numpy.array} -- Weight matrix
        """
        if np.shape(w) == np.shape(self.weights):
            self.weights = w
        else:
            print('Warning: Weight matrix dimensions not compliant, dimensions of the layer are',  np.shape(self.weights))

    def set_thresholds(self, t):
        """ Manually set the thresholds of the network
        Arguments:
            t {numpy.array} -- Threshold vector
        """

        if np.shape(t) == np.shape(self.thresholds):
            self.thresholds = t
        else:
            print('Warning: Threshold dimensions not compliant, dimensions of the thresholds are',  np.shape(self.thresholds))

    def feed_forward(self, layer_input):
        """Feed forward through the layer
        Arguments:
            layer_input {numpy.array} -- Data to be passed through the layer
        Returns:
            np.array -- Output from layer with activation function
        """
        self.b = - self.thresholds + self.weights @ layer_input

        # Select activation function
        if self.activation_function.lower() == 'linear':
            return self.linear()
        elif self.activation_function.lower() == 'relu':
            return self.relu()
        elif self.activation_function.lower() == 'heaviside':
            return self.heaviside()
        elif self.activation_function.lower() == 'signum':
            return self.signum()
        elif self.activation_function.lower() == 'sigmoid':
            return self.sigmoid()
        elif self.activation_function.lower() == 'tanh':
            return self.tanh()
        elif self.activation_function.lower() == 'softmax':
            return self.softmax()
        else:
            print(self.activation_function, 'is not an avalible activation function')


    def output_error(self, train_label):
        """Compute the output_error, or the loss of the layer
        Arguments:
            train_label {numpy.array} -- Training label to use for computing the error
        Returns:
            numpy.array -- Output error of the layer
        """
        if self.activation_function == '':
            print('Error: Layer output has not yet been computed. Try running FeedForwardLayer.feed_forward first.')
            return None
        else:
            self.propagation_error = (train_label - self.output) * self.dg
            return self.propagation_error

    ############################
    ### Activation functions ###
    ############################

    def linear(self):
        # Returns layer output without activation function
        self.activation_function = 'linear'
        self.dg = np.ones(np.size(self.thresholds))
        self.output = self.b
        return self.output

    def relu(self):
        # Rectified linear unit function
        self.activation_function = 'relu'
        self.output = np.maximum(0,self.b)
        self.dg = np.sign(self.output)
        return self.output

    def heaviside(self):
        # Heavyside step function
        self.activation_function = 'heaviside'
        self.output = np.heaviside(self.b,0)
        self.dg = np.zeros(np.size(self.thresholds))
        return self.output

    def signum(self):
        self.activation_function = 'signum'
        self.output = np.sign(self.b)
        self.dg = np.zeros(np.size(self.thresholds))
        return self.output

    def sigmoid(self):
        self.activation_function = 'sigmoid'
        self.output = 1 / (1 + np.exp(-self.b))
        self.dg = np.exp(-self.b)/(1 + np.exp(-self.b))**2
        return self.output

    def tanh(self):
        self.activation_function = 'tanh'
        self.output = np.tanh(self.b)
        self.dg = 1 - self.output**2
        return self.output

    # Not tested yet
    def softmax(self, alpha = 1):
        self.activation_function = 'softmax'
        self.output = np.exp(alpha*self.b)/np.sum(np.exp(alpha*self.b))
        self.dg = 1
        return self.output

##########################################
### Network class
##########################################

class NeuralNetwork:

    def __init__(self, layer_list):
        """Create a neural network
        Arguments:
            layer_list {list(opynn.Layer)} -- Any type of opynn layer in order first to last in network
        """
        self.layers = layer_list
        self.network_depth = len(layer_list)
        self.n_inputs = layer_list[0].shape[0]
        self.n_outputs = layer_list[-1].shape[1]
        self.v = [np.zeros((self.layers[0].shape[0]))]
        self.error = []
        for i in range(self.network_depth):
            self.v.append(np.zeros((self.layers[i].shape[1])))
            self.error.append(np.zeros((self.layers[i].shape[1])))


    def output(self, data):
        """Feed forward through the network and return the output of the last layer
        Arguments:
            data {numpy.array} -- Input data to the network
        Returns:
            numpy.array -- Network output
        """
        self.v[0] = data.copy()
        for i in range(self.network_depth):
            self.v[i+1] = self.layers[i].feed_forward(self.v[i])
        return self.v[-1]

    def feed_forward(self, data):
        """Feed forward through the network and return all layer outputs
        Arguments:
            data {numpy.array} -- Input data to the network
        Returns:
            list(numpy.array) -- Outputs of all the layers in the network
        """
        self.v[0] = data.copy()
        for i in range(self.network_depth):
            self.v[i+1] = self.layers[i].feed_forward(self.v[i])
        return self.v


class GeneticOptimizer:
    def __init__(self, size, population_size, evaluator, n_generations,
                tournament_selection_probability, mutation_probability):
        self.size = size
        self.population_size = population_size
        self.evaluator = evaluator
        self.n_generations = n_generations
        self.tournament_selection_probability = tournament_selection_probability
        self.mutation_probability = mutation_probability
        self.population = None

    def optimize(self):
        # Initialize population
        if self.population == None:
            self.init_population()
            pop1 = self.population

        fitness = np.zeros(self.population_size)
        for i in range(self.n_generations):
            # Evaluate population
            for j in range(self.population_size):
                fitness[j] = self.evaluator(self.population[j, :])
            self.top_individual = self.population[np.argmax(fitness), :]
                
            # Selection (tournament style)
            selected = np.random.randint(self.population_size, size=(self.population_size, 2))
            selected = np.sort(selected, axis=1)
            r = (np.random.rand(self.population_size) < self.tournament_selection_probability).astype(int)
            selected = selected[np.arange(self.population_size), r]
            tmp_population = self.population[selected, :]

            # Crossover
            for j in np.arange(self.population_size, step=2):
                g1 = tmp_population[j, :]
                g2 = tmp_population[j + 1, :]
                cross_point = np.random.randint(self.size)
                g1[:cross_point] = g2[:cross_point]
                g2[cross_point:] = g1[cross_point:]
                tmp_population[j, :] = g1
                tmp_population[j + 1, :] = g2

            # Mutation
            r = np.random.rand(self.population_size, self.size) < self.mutation_probability
            tmp_population[r] = tmp_population[r] + np.random.normal(0, 1, size=tmp_population[r].shape)

            # Put in top dog again
            tmp_population[0, :] = self.top_individual
            self.population = tmp_population.copy()
            print('Generation:', i, ' - Max fitness:', fitness.max())
            
    def init_population(self):
        self.population = np.random.normal(0, 1, (self.population_size, self.size))