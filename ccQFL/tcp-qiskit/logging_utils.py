import os
from datetime import datetime

logs_folder = None

def setup_logging(algorithm=None, data_used=None, maxiter=None, num_devices=None):
    global logs_folder
    # date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
    logs_dir = f"{algorithm}_{data_used}_maxiter={maxiter}_numDevices={num_devices}"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    logs_folder = logs_dir
    return logs_dir

def get_logs_folder():
    return logs_folder
