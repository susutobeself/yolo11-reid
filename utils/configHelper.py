import configparser
import os

def init_config():
    file = os.getcwd()+'/config.ini'
    con = configparser.ConfigParser()
    con.read(file, encoding='utf-8')
    return con