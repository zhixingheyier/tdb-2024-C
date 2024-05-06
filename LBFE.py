#coding:utf-8
import preprocess
import chat_with_LLM as cwl
import logger_config
import sys
from tqdm import tqdm
import CLM
import numpy as np
import copy
import pandas as pd
import re
import time
logger = logger_config.myLogger().get_logger()

class featrue_constructor(cwl.paperjudge_assistant):
    def __init__(self,api_key=None,logger=logger):
        super().__init__(api_key,logger)
        self.level2number_list = [1,3,5,7,9]
        self.level2number_path=preprocess.get_config('PATH','output-path')+time.strftime("%Y-%m-%d-%H-%M", time.localtime())+"-level2number.csv"

    def four_judgement_process(self,paper_name:str=None,initial_prompt:list=None,paper_content:str=None):
        if (initial_prompt is None) or (paper_name is None) or (paper_content is None):
            self.logger.error("初始化失败，请检查文件名！")
            sys.exit()
        history=copy.deepcopy(initial_prompt)
        warmup_prompt=history+[{"role": "user","content": "你将收到一篇古诗，古诗如下：《咏鹅》：鹅鹅鹅，曲项向天歌；白毛浮绿水，红掌拨清波。请给出这首古诗的评分档次和评价，评分档次分为五档，由低到高依次为：差、较差、一般、较好、好，输出格式为：<task>古诗名字</task>:<level>评分档次</level>:<judgement>评价</judgement>"}]
        warmup_prompt+=[{"role": "assistant","content": "<task>咏鹅</task>:<level>好</level>:<judgement>这首诗以其简洁明快的语言和生动形象的描绘，成功地刻画了鹅的优雅姿态和动态美。整首诗不仅传达了诗人对自然美景的赞美，也体现了中国古诗的韵味和意境，是一首流传广泛的经典之作。</judgement>"}]
        # 定义Prompts
        prompts={
                'completeness_prompt':"你将收到参赛者提交的竞赛论文，论文用XML标签分割。阅读竞赛论文“{}”，考察论文以下三个任务的完成情况，并分别给出三个任务完成情况的评分档次和评价，评分档次分为五档，由低到高依次为：差、较差、一般、较好、好，输出格式为：<task>任务序号</task>:<level>评分档次</level>:<judgement>评价</judgement>。任务1、“群众留言分类任务”：要求建立一级标签分类模型，对群众留言进行分类，以便分派至相应职能部门处理，要求使用F-Score评价分类方法。任务2、“热点问题挖掘任务”：要求定义合理的热度评价指标，对某一时段内反映特定地点或特定人群问题的留言进行归类，并给出排名前5的热点问题及其留言信息。任务3、“答复意见的评价任务”：要求从相关性、完整性、可解释性等角度对答复意见的质量给出评价方案，并尝试实现。".format(paper_name),
                'substantiality_prompt':"你将收到参赛者提交的竞赛论文，论文用XML标签分割。阅读竞赛论文“{}”，考察论文对以下九项研究的完成情况，并分别给出九项研究完成情况的评分档次和评价，评分档次分为五档，由低到高依次为：差、较差、一般、较好、好，输出格式为：<research>研究序号</research><level>评分档次</level><judgement>评价</judgement>。研究1、文本处理和特征提取，研究2、进行留言分类并提供分类结果，研究3、采用多种分类方法进行比较分析，研究4、提出明确的问题热度度量方法，研究5、对热点问题的提取，研究6、考虑热点问题对社会和政府部门工作的影响，研究7、对政府部门的答复意见进行评价，研究8、提出合理的答复意见评价方法，研究9、区分不同类别问题的答复意见并分别进行评价。".format(paper_name),
                "consistency_prompt":"你将收到参赛者提交的竞赛论文，论文用XML标签分割。阅读竞赛论文“{}”，按照以下五个指标考察论文摘要质量，并分别根据五个指标的考察情况给出各个指标的评分档次和评价，评分档次分为五档，由低到高依次为：差、较差、一般、较好、好，输出格式为：<index>指标序号</index>:<level>评分档次</level>:<judgement>评价</judgement>。指标1、主题一致性，指标2、方法一致性，指标3、结论一致性，指标4、内容覆盖度，指标5、摘要简洁性。".format(paper_name),
                "writing_prompt":"你将收到参赛者提交的竞赛论文，论文用XML标签分割。阅读竞赛论文“{}”，按照以下五个指标考察论文写作水平，并分别根据五个指标的考察情况给出各个指标的评分档次和评价，评分档次分为五档，由低到高依次为：差、较差、一般、较好、好，输出格式为：<index>指标序号</index>:<level>评分档次</level>:<judgement>评价</judgement>。指标1、文字流畅性，指标2、写作规范性，指标3、论文逻辑性，指标4、篇章结构合理性，指标5、论点论据一致性。".format(paper_name),
                 }
        
        result_completeness,tokenusage1=self.chat_1(prompts['completeness_prompt'],warmup_prompt[:],file_carried=paper_content)
        result_substantiality,tokenusage2=self.chat_1(prompts['substantiality_prompt'],history[:],file_carried=paper_content)
        result_consistency,tokenusage3=self.chat_1(prompts['consistency_prompt'],history[:],file_carried=paper_content)
        result_writing,tokenusage4=self.chat_1(prompts["writing_prompt"],history[:],file_carried=paper_content)

        feature_completeness=self.get_completeness_feature(result_completeness)
        feature_substantiality=self.get_substantiality_feature(result_substantiality)
        feature_consistency=self.get_consistency_feature(result_consistency)
        feature_writing=self.get_writing_feature(result_writing)

        self.logger.info("{}:的\n完整性评价为：{}\n实质性评价为：{}\n一致性评价为：{}\n写作水平评价为：{}".format(paper_name,result_completeness,result_substantiality,result_consistency,result_writing))
        self.logger.info("{}:的\n完整性特征为：{}\n实质性特征为：{}\n一致性特征为：{}\n写作水平特征为：{}".format(paper_name,feature_completeness,feature_substantiality,feature_consistency,feature_writing))

        if feature_completeness is None:
            self.logger.error("{}完整性评价失败！其完整性评价为：{}".format(paper_name,result_completeness))
        if feature_substantiality is None:
            self.logger.error("{}实质性评价失败！其实质性评价为：{}".format(paper_name,result_substantiality))
        if feature_consistency is None:
            self.logger.error("{}一致性评价失败！其一致性评价为：{}".format(paper_name,result_consistency))
        if feature_writing is None:
            self.logger.error("{}写作水平评价失败！其写作水平评价为：{}".format(paper_name,result_writing))

        return feature_completeness,feature_substantiality,feature_consistency,feature_writing,tokenusage1+tokenusage2+tokenusage3+tokenusage4
    
    def get_completeness_feature(self,result_completeness):
        pattern = re.compile('<level>(.*?)</level>')
        result = pattern.findall(result_completeness)
        if len(result)==3:
            return result
        else:
            return None
        
    def get_substantiality_feature(self,result_substantiality):
        pattern = re.compile('<level>(.*?)</level>')
        result = pattern.findall(result_substantiality)
        if len(result)==9:
            return result
        else:
            return None
        
    def get_consistency_feature(self,result_consistency):
        pattern = re.compile('<level>(.*?)</level>')
        result = pattern.findall(result_consistency)
        if len(result)==5:
            return result
        else:
            return None
        
    def get_writing_feature(self,result_writing):
        pattern = re.compile('<level>(.*?)</level>')
        result = pattern.findall(result_writing)
        if len(result)==5:
            return result
        else:
            return None
    def level2number(self,df=None):
        #遍历df的列名字
        for colname in df.columns:
            if colname=='论文名':
                continue
            df[colname] = df[colname].replace('差',self.level2number_list[0])
            df[colname] = df[colname].replace('较差',self.level2number_list[1])
            df[colname] = df[colname].replace('一般',self.level2number_list[2])
            df[colname] = df[colname].replace('较好',self.level2number_list[3])
            df[colname] = df[colname].replace('好',self.level2number_list[4])
            df[colname] = df[colname].astype(int)
        df.to_csv(self.level2number_path,index=False,encoding='utf-8')


def main():
    # raise ImportError")
    pass

    

if __name__=="__main__":
    main()