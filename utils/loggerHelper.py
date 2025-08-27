import logging
import logging.config
import os
from pathlib import Path
from datetime import datetime

def init_logging():
    """配置结构化日志"""
    Path("./log/LogInfo").mkdir(parents=True,exist_ok=True)
    Path("./log/LogWarn").mkdir(parents=True,exist_ok=True)
    Path("./log/LogError").mkdir(parents=True,exist_ok=True)
    Path("./log/LogDebug").mkdir(parents=True,exist_ok=True)
    logging.config.fileConfig(str(os.getcwd())+'/logging.conf')

def get_logger(name: str):
    """获取logger"""
    return logging.getLogger(name)