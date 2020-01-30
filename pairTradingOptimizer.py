class MLEngine:
    def __init__(self, data):
        self.raw_data = data
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.X_test = None
        self.y_test = None


        self.model = {"SNN": None,
                     "LSTM": None,
                     "GRU":None}
        
        
    def get_signal(self):
        
        
    # model
    def build_RL(self):


            
    def build_KarmaFilter(self):
        

        
    # train:
    def train(self, model, epochs=100):

        
    # predict
    def predict(self, model, X):
    
    # evaluate:
    def score(self, model, X, y_true):
    
    def result_evaluation(self, model):
        #plot
        
