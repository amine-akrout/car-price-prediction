"""
Deploy the training flow with a schedule.
"""

from datetime import timedelta

from model_training import training_flow

if __name__ == "__main__":
    training_flow.serve(
        name="model-training",
        tags=["training", "model"],
        interval=timedelta(minutes=10),
    )
