import logging
from datetime import datetime
import os
import types


def get_logger(config):
    log_path = os.path.join(
        "logs", config.exp_name, f"{config.exp_name}.log")

    logger = logging.getLogger(config.exp_name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:

        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def log_message(logger, message):
    logger.info(message)
    print(message)


def log_config(logger, config, indent=0):
    prefix = " " * indent
    if indent == 0:
        logger.info("Configuration:")

    if isinstance(config, types.SimpleNamespace):
        config = vars(config)

    for key, value in config.items():
        if isinstance(value, (types.SimpleNamespace, dict)):
            logger.info(f"{prefix}{key}:")
            log_config(logger, value, indent=indent + 4)
        else:
            logger.info(f"{prefix}{key}: {value}")


def log_metrics(logger, metrics):
    logger.info("Metrics:")
    for key, value in metrics.items():
        logger.info(f"{key}: {value}")
    logger.info("\n")


def log_epoch(logger, epoch, metrics):
    logger.info(f"Epoch {epoch}:")
    for key, value in metrics.items():
        logger.info(f"{key}: {value}")
    logger.info("\n")


def log_training(logger, epoch, step, loss):
    logger.info(f"Epoch {epoch}, Step {step}: Loss: {loss}")
    logger.info("\n")


def log_validation(logger, epoch, step, metrics):
    logger.info(f"Epoch {epoch}, Step {step}: Validation Metrics:")
    for key, value in metrics.items():
        logger.info(f"{key}: {value}")
    logger.info("\n")


def log_test(logger, metrics):
    logger.info("Test Metrics:")
    for key, value in metrics.items():
        logger.info(f"{key}: {value}")
    logger.info("\n")
