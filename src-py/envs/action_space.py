class ActionSpace:
    def __init__(self, low_bound, high_bound):
        """_summary_

        Args:
            low_bound (np array): lower bound of action vector, shape: [action_dim,]
            high_bound (_type_): higher bound of action vector, shape: [action_dim,]
        """
        self.low = low_bound
        self.high = high_bound