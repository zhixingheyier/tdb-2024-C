
#-*- coding: utf-8 -*-
import logging
import time
 
class myLogger():
    # creating a logger对象
    logger = logging.getLogger('mylogger')    

    # define the default level of the logger
    logger.setLevel(logging.INFO)  
    
    # creating a formatter
    formatter = logging.Formatter( '%(asctime)s | %(levelname)s => %(message)s' )  
    
    # creating a handler to log on the filesystem  
    t=time.strftime("%Y-%m-%d", time.localtime())
    name=r"./log/"+t+".log"
    handler=logging.FileHandler(name,encoding='utf-8')
    handler.setFormatter(formatter)  
    handler.setLevel(logging.INFO)  
    
    # adding handlers to our logger  
    logger.addHandler(handler)   
    
    #logger.info('this is a log message...')  
    
    def get_logger(self):    
        return self.logger

def test():
    logger=myLogger().get_logger()
    logger.info("我是谁")

if __name__=='__main__':
    test()