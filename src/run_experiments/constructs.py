from copy import deepcopy

class ModelWrapper:
    def train(self, train_X, train_y, test_X, test_y):
        pass
    
    def predict(self, X):
        pass
    
    def predict_proba(self, X):
        pass

"""
All experiments should be self contained: can be replicated from their given
hyper parameters + data
"""
class Experiment:
    
    @property
    def model(self) -> ModelWrapper:
        pass

"""
Experiment artifacts 
"""
class ExperimentArtifacts:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        self.data = {}
        self.metrics = {}
        self.model = None
        
    @classmethod
    def load(self, path: str) -> "ExperimentArtifacts":
        pass
    
    def set_model(self, model: ModelWrapper): # saves the trained model
        self.model = deepcopy(model)
    
    def append_data(self, item: dict):
        self.data[item["name"]] = item["data"]
    
    def append_metric(self, item: dict):
        self.metrics[item["name"]] = item["data"]
        
    def print_summary(self):
        print("Experiment Artifacts Summary:")
        print("Metrics:")
        for name in self.metrics:
            print(f" - {name}: {self.metrics[name]}")
            
    def save(self, path: str):
        pass
    
