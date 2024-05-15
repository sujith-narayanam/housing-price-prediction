import argparse
import importlib

import mlflow

from housing_price_prediction.ingest_data import pull_housing_data
from housing_price_prediction.mle_lib.logger import initiate_logger
from housing_price_prediction.mle_lib import api
from housing_price_prediction.mle_lib.api import (
    get_new_exp_num,
    config,
)
from housing_price_prediction.score import score_models
from housing_price_prediction.train import process_data, train_models
from ta_lib.core.api import (
    register_processor,
    DEFAULT_ARTIFACTS_PATH,
    DEFAULT_DATA_BASE_PATH,
)


@register_processor("entire-pipeline", "all")
def main(
    context=None,
    params={},
    run_data_module=True,
    run_model_module=True,
    run_scoring_module=True,
    models_path=DEFAULT_ARTIFACTS_PATH,
    logger_path=config.log_path,
    data_path=DEFAULT_DATA_BASE_PATH,
    log_level="DEBUG",
    console=True,
):
    print(context)
    initiate_logger(log_level=log_level, log_file=logger_path, console=console)

    importlib.reload(api)
    from housing_price_prediction.mle_lib.api import logger

    experiment_name = f"experiment_{get_new_exp_num()}"
    experiment_id = mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(
        experiment_id=experiment_id, run_name="Housing Price Prediction Run"
    ) as run_parent:
        logger.info(f"Parent mlflow run_id is {mlflow.active_run().info.run_id}")
        if run_data_module:
            with mlflow.start_run(
                experiment_id=experiment_id, run_name="Data pull Tracker", nested=True
            ) as run_data:
                logger.info("================================")
                logger.info("Starting Data preparation module")
                logger.info("================================")
                logger.info(f"Mlflow run_id is {mlflow.active_run().info.run_id}")
                pull_housing_data(data_path, context=context)

        if run_model_module:
            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name="Model Training Tracker",
                nested=True,
            ) as run_model_train:
                logger.info("================================")
                logger.info("Starting Model training module")
                logger.info("================================")
                logger.info(f"Mlflow run_id is {mlflow.active_run().info.run_id}")
                process_data(data_path)
                train_models(models_path)

        if run_scoring_module:
            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name="Model Scoring Tracker",
                nested=True,
            ) as run_model_score:
                logger.info("================================")
                logger.info("Starting Model scoring module")
                logger.info("================================")
                logger.info(f"Mlflow run_id is {mlflow.active_run().info.run_id}")
                score_models(models_path, data_path)


@register_processor("data-pipeline", "ingest-data")
def run_data_pipeline(context, params):
    main(
        run_data_module=True,
        run_model_module=False,
        run_scoring_module=False,
    )


@register_processor("model-pipeline", "train-model")
def run_model_pipeline(context, params):
    main(
        run_data_module=False,
        run_model_module=True,
        run_scoring_module=False,
    )


@register_processor("scoring-pipeline", "score-model")
def run_scoring_pipeline(context, params):
    main(
        run_data_module=False,
        run_model_module=False,
        run_scoring_module=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", help="Data Path")
    parser.add_argument("--models_path", help="Models Path")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="DEBUG",
        help="Set the logging level",
    )
    parser.add_argument("--log-path", help="Specify the path for the log file")
    parser.add_argument(
        "--no-console-log", action="store_true", help="Disable console logging"
    )
    args = parser.parse_args()

    models_path = args.models_path
    data_path = args.data_path
    logger_path = args.log_path
    log_level = args.log_level
    if logger_path is None:
        logger_path = config.log_path
    if log_level is None:
        log_level = "DEBUG"
    if models_path is None:
        models_path = DEFAULT_ARTIFACTS_PATH
    if data_path is None:
        data_path = DEFAULT_DATA_BASE_PATH

    main(
        models_path=models_path,
        data_path=data_path,
        logger_path=logger_path,
        log_level=log_level,
        console=not args.no_console_log,
    )
