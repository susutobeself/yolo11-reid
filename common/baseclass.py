import logging
import os

from utils.configHelper import init_config
from utils.loggerHelper import init_logging,get_logger

init_logging() # 初始化日志系统
con = init_config()  # 初始化配置文件

class BaseClass:
    #global logger
    cacheImageReady = 'CacheImageReady_'
    
    def __init__(self):
        self.logger = get_logger("main")
    
    def loginfo(self,msg):
        self.logger.info('['+str(self.__class__.__name__)+'] ' +msg)
        #print('[loginfo]['+str(self.__class__.__name__)+'] ' +msg)
        
    def logdebug(self,msg):
        self.logger.debug('['+str(self.__class__.__name__)+'] ' +msg)
        #print('[logdebug]['+str(self.__class__.__name__)+'] ' +msg)
        
    def logerror(self,msg):
        self.logger.error('['+str(self.__class__.__name__)+'] ' +msg)
        
    def logwarn(self,msg):
        self.logger.warn('['+str(self.__class__.__name__)+'] ' +msg)    
            
    def getConfig(self,name):
        if con.has_section(name):
            return dict(con.items(name))   
        else:
            return None 
        