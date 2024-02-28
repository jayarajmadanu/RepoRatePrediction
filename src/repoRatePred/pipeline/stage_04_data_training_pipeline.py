from src.repoRatePred.config.configuration import ConfigurationManager
from src.repoRatePred.logger import logger
from src.repoRatePred.entity.config_entity import DataTrainingConfig
from src.repoRatePred.components.data_trainer import DataTrainer

class DataTrainingPipeline:
    def __init__(self, config: DataTrainingConfig=None):
        if config == None :
            config = ConfigurationManager().get_data_training_config()
        self.data_training_config = config
        
    def main(self):
        data_trainer = DataTrainer(config= self.data_training_config)
        data_trainer.train()
        return ""

if __name__ == '__main__':
    try:
        obj = DataTrainingPipeline()
        obj.main()
    except Exception as e:
        raise e
        
        
    