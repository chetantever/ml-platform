# Read Config File

import configparser

def readConfigFile(filename):
    config = configparser.ConfigParser()
    config.read(filename)
    return config