from benchmark import run_evaluation
from common import CONFIG, EvaluationConfig

__all__ = ["CONFIG", "EvaluationConfig", "run_evaluation"]


if __name__ == "__main__":
    run_evaluation(CONFIG)
