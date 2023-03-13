import yaml

class DatasetFeature:
    def __init__(self, feature_name, dic):
        self.name = feature_name
        self.type = dic["type"]
    
    def __repr__(self):
        return "DatasetFeature: " + str(self.__dict__)

class Feature:
    def __init__(self, feature_name, dic):
        self.hash = False
        if "hash" in dic:
            self.hash = dic["hash"]
            if self.hash:
                assert "vocab_size" in dic
                self.vocab_size = dic["vocab_size"]
        self.embedding_dim = dic["embedding_dim"]
        self.name = feature_name
        self.belongs_to = dic["belongs_to"]

    def __repr__(self):
        return "Feature: " + str(self.__dict__)
        
class Model:
    def __init__(self, dic):
        self.learning_rate = dic["learning_rate"]
        self.features = []
        for feature_name, feature_dic in dic["features"].items():
            self.features.append(Feature(feature_name, feature_dic))

    def __repr__(self):
        return "Model: " + str(self.__dict__)

class Config:
    def __init__(self, path):
        dic = yaml.safe_load(open(path, 'r'))
        self.epochs = dic["epochs"]
        self.compression = dic["compression"]
        self.global_batch_size = dic["global_batch_size"]
        self.label = dic["label"]
        self.shuffle_buffer = dic["shuffle_buffer"]
        self.train_path = dic["train_path"]
        self.validate_path = dic["validate_path"]
        self.save_model_path = dic["save_model_path"]
        self.train_rows = dic["train_rows"]
        self.trainval_rows = dic["trainval_rows"]
        self.eval_rows = dic["eval_rows"]
        self.model = Model(dic["model"])
        self.dataset_features = []
        self.cycle_length = dic["cycle_length"]
        for feature_name, feature_dic in dic["dataset_features"].items():
            self.dataset_features.append(DatasetFeature(feature_name, feature_dic))
       
    def __repr__(self):
        return "Config: " + str(self.__dict__)
