#coding:utf-8
# ls
import logger_config
import pandas as pd
import preprocess
import time
import copy
import numpy as np
logger = logger_config.myLogger().get_logger()
class comprehensive_judger:
    def __init__(self,logger=logger):
        # self.df = df
        self.logger = logger

    def entropyWeight(self,data:pd.DataFrame=None):
        data = np.array(data)
        # 归一化
        P = data / data.sum(axis=0)

        # 计算熵值
        E = np.nansum(-P * np.log(P) / np.log(len(data)), axis=0)

        # 计算权系数
        return (1 - E) / (1 - E).sum()
    

    def topsis(self,data:pd.DataFrame=None, weight:list=None):
        # 归一化
        data = data / np.sqrt((data ** 2).sum())

        # 最优最劣方案
        Z = pd.DataFrame([data.min(), data.max()], index=['负理想解', '正理想解'])

        # 距离
        weight = self.entropyWeight(data) if weight is None else np.array(weight)
        Result = data.copy()
        Result['正理想解'] = np.sqrt(((data - Z.loc['正理想解']) ** 2 * weight).sum(axis=1))
        Result['负理想解'] = np.sqrt(((data - Z.loc['负理想解']) ** 2 * weight).sum(axis=1))

        # 综合得分指数
        Result['综合得分指数'] = Result['负理想解'] / (Result['负理想解'] + Result['正理想解'])
        Result['排序'] = Result.rank(ascending=False)['综合得分指数']

        return Result, Z, weight

    def comprehensive_score(self):
        df=pd.read
        raise NotImplementedError
    
    
    # def comprehensive_score(self):

if __name__ == '__main__':
    comp = comprehensive_judger(df=pd.read_csv(preprocess.get_config('PATH','output-path')+"all-features.csv",encoding='utf-8'))
    # comprehensive_score(all_judgements)
    comp.level2number()
    # df=comp.comprehensive_score()
    # df.to_csv(preprocess.get_config('PATH','output-path')+"2024-04-21-12-14-comprehensive_score.csv",index=False,encoding='utf-8')
    # print(df)
