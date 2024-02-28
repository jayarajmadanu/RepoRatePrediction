from src.repoRatePred.pipeline.stage_01_data_ingestion_pipeline import DataIngestionPipeline
from src.repoRatePred.pipeline.stage_02_data_validation_pipeline import DataValidationPipeline
from src.repoRatePred.pipeline.stage_03_data_transformation_pipeline import DataTransformationPipeline
from src.repoRatePred.pipeline.stage_04_data_training_pipeline import DataTrainingPipeline
from src.repoRatePred.logger import logger
from src.repoRatePred.config.configuration import ConfigurationManager


config = ConfigurationManager()

STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion_pipeline = DataIngestionPipeline(data_ingestion_config)
    data_ingestion_pipeline.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Validation stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_validation_config = config.get_data_validation_config()
    data_validation_pipeline = DataValidationPipeline(data_validation_config)
    data_validation_pipeline.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Transformation stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_transformation_config = config.get_data_transformation_config()
    data_transformation_pipeline = DataTransformationPipeline(data_transformation_config)
    data_transformation_pipeline.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Training stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_training_config = config.get_data_training_config()
    data_training_pipeline = DataTrainingPipeline(data_training_config)
    model_training_best_params = data_training_pipeline.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e