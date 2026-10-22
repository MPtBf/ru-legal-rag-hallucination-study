# Нулевой шот: LLM-классы и пайплайн (вынесены из корневого llms.py).
from experiments.zero_shot.llms import BaseLLM, DeepSeekModel, GigaChatModel, log_error
from experiments.zero_shot.pipeline import ZeroShotPipeline

__all__ = [
    "BaseLLM",
    "DeepSeekModel",
    "GigaChatModel",
    "ZeroShotPipeline",
    "log_error",
]
