from utils.config import Config

from .train_SP_pipeline import Train_SP_Pipeline
from .test_SP_pipeline_base import Test_SP_Pipeline_base
from .benchmark_SP_pipeline import Benchmark_SP_Pipeline

def get_pipeline(config: Config):
    pipelines = {
        'SP_train': Train_SP_Pipeline,
        'SP_test': Test_SP_Pipeline_base,
        'SP_benchmark': Benchmark_SP_Pipeline
    }

    return pipelines[config.pipeline.name](config)
