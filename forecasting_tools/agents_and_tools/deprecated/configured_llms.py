from forecasting_tools.ai_models.deprecated_model_classes.gpt4o import Gpt4o
from forecasting_tools.ai_models.deprecated_model_classes.gpt4ovision import (
    Gpt4oVision,
    Gpt4VisionInput,
)

default_llms = {
    "basic": "openrouter/anthropic/claude-sonnet-4.5",
    "advanced": "openrouter/anthropic/claude-sonnet-4.5",
}


class BasicLlm(Gpt4o):
    MODEL_NAME = "openrouter/anthropic/claude-sonnet-4.5"


class AdvancedLlm(Gpt4o):
    MODEL_NAME = "openrouter/anthropic/claude-sonnet-4.5"


class VisionLlm(Gpt4oVision):
    pass


class VisionData(Gpt4VisionInput):
    pass
