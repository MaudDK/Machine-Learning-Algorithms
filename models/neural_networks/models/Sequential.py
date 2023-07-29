class Sequential:
    def __init__(self, name):
        self.name = name
        self.layers = []
        self.outputs = []
    
    def add(self, layer):
        self.layers.append(layer)
    
    def assemble(self):
        pass

    def summary(self):
        print(f'{self.name} Summary:')
        print('_________________________________________________________________')
        print(f"{'Type:':<13}{'Params:':>13}")
        print('_________________________________________________________________')
        for i, layer in enumerate(self.layers):
            print(f'{i}_{layer.type:<13}{layer.params:>13}')
        print('_________________________________________________________________')

    def predict(self, X):
        a = X
        for layer in self.layers:
            a = layer.forward(a)
            self.outputs.append(a)
        
        return a

    def evaluate(self):
        pass