import os
from ta_lib.core.api import DEFAULT_HOME_PATH

# data pull, split
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

test_size = 0.2


# paths
log_path = os.path.join(
    DEFAULT_HOME_PATH,
    "logs",
    "run_logger.log",
)
