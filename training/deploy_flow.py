"""
Deploy the training flow with a schedule.
"""

from datetime import timedelta

from model_training import training_flow

if __name__ == "__main__":
    training_flow.deploy(
        name="model-training",
        work_pool_name="default-pool",
        interval=timedelta(minutes=10),
    )
