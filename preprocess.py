import os
import configparser
import os
import logger_config
import chat_with_LLM
import numpy as np
import json
import re
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
logger = logger_config.myLogger().get_logger()

# 获取config.ini文件中指定section和键的值
def get_config(section_name,key_name):
    config = configparser.ConfigParser()
    # 读取INI文件
    config.read("config.ini",encoding='utf-8')
    # 获取指定section的内容
    section_content = config[section_name]
    # 访问section中的键值对
    for key, value in section_content.items():
        # print(f"{key}: {value}")
        if key==key_name:
            return value
    else:
        logger.info("未找到指定键值对")
        return None
    
# 遍历某目录下的所有文件，并将它们的路径存储在一个列表中。
def get_file_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

# 遍历某目录下的所有文件，并将它们的文件名存储在一个列表中。
def get_file_names(directory):
    file_names = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_names.append(file)
    return file_names

def read_file(file_path:str=None)->str:
    if file_path is None:
        logger.info("文件路径为空")
        return None
    file_text:str=''
    pattern = re.compile('[a-zA-Z0-9_\s\(\)\[\]\=\.\'\+]{15}', re.I|re.M)
    for page_layout in extract_pages(file_path):
      for element in page_layout:
          if isinstance(element, LTTextContainer):
            text=element.get_text()
            # logger.info(text)
            flag = pattern.search(text)
            if flag is None:
              # logger.info(text)
              file_text+=text
    return file_text

def data_clean(text:str)->str:
    # # 去除文本中的换行符
    text = text.replace('\n\n', '\n')
    text = text.replace('\n \n', '')
    text = text.replace('---', '')
    text = text.replace('\t', '')
    text = text.replace('\r', '')
    
    # 去除文本中特殊字符，如：
    text = text.replace('', '')
    # 去除文本中特殊字符，如：\xa0
    text = text.replace('   ', '')
    text = text.replace('  ', ' ')
    text = text.replace('………', '')

    # 去除文本中政治敏感字眼
    text = text.replace('中国', '我国')
    # 去除性别歧视字眼
    # 去除种族歧视字眼
    # 去除宗教歧视字眼

    
    return text

if __name__ == "__main__":
    # directory = r"C:\Users\lesliu\Desktop\c\datas\standards"
    # file_paths = get_file_paths(directory)
    # print(file_paths)
    # for file_path in file_paths:
    #     file_name=file_path.split("\\")[-1]
    #     print(file_name)
    pass



