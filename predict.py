from src.repoRatePred.pipeline.predict_pipeline import PredictPipeline
from src.repoRatePred.logger import logger
data = {
    'GoldPrice': 3034,
    'BankRate': 7.1,
    'CrudeOilRate': 90,
    'UsdInr': 90,
    'PreviousRepoRate': 7
}

predict = PredictPipeline()
res = predict.predict(data)

logger.info(f'prediction of data {data} is RepoRate: {res}')