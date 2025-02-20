import configparser

def load_config():
    config = configparser.ConfigParser()
    config.read('./config/config.ini')
    return config