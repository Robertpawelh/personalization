class ModelParameters(): # helper class to store model parameters
    def __init__(self,
            n_days_in: int,
            hidden_dim: int,
            num_layers: int,
            n_days_out: int,
            n_specs: 10,
            dropout: float = 0.05,
            learning_rate: float = 0.0005,
            train_size: float = 0.8):
        self.n_days_in = n_days_in
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_days_out = n_days_out
        self.n_specs = n_specs
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.train_size = train_size
