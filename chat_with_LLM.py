from openai import OpenAI
from pathlib import Path
import requests
import json
import logger_config
import preprocess
logger = logger_config.myLogger().get_logger()

class paperjudge_assistant:
    def __init__(self,api_key=None,logger=logger):
        self.logger=logger
        self.api_key=api_key
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1",
    )
    

    def prompt_init(self):
        history = [
            {
                "role": "system", 
                "content": "你是泰迪杯数据挖掘挑战赛组委会的论文评审专家，你的职责是细致入微地审阅参赛者提交的PDF论文。在评审过程中，你应致力于全面理解参赛者的研究动机、方法、过程和结论，并基于论文的质量、创新性、实用性和学术贡献，给出一个公正且合理的评价。"
                # "content": "你是泰迪杯数据挖掘挑战赛组委会的论文评审专家，你的职责是细致入微地审阅参赛者提交的PDF论文，给出一个公正且合理的评价。另外，由于对优秀论文的数量有限制，为了让论文水平具有区分度，你应该提高标准，严格评价论文"
                
            }
        ]
        
        return history
    
    def chat_1(self,query, history,file_carried=None):
        history += [{
        "role": "system", 
        "content": query
        }]
        if file_carried is not None:
            history += [{
                "role": "user",
                "content": f"<article>{file_carried}</article>",
            }]
        completion = self.client.chat.completions.create(
            model="moonshot-v1-32k",
            messages=history,
            temperature=0.2, #使用什么采样温度，介于 0 和 1 之间。较高的值（如 0.7）将使输出更加随机，而较低的值（如 0.2）将使其更加集中和确定性
        )
        result = completion.choices[0].message.content
        # self.logger.info("history为:{}".format(history))
        return result,completion.usage.total_tokens

    
    # 查询余额
    def check_balance(self)->int:
        url = "https://api.moonshot.cn/v1/users/me/balance"
        headers = {
            "Authorization": "Bearer "+self.api_key
        }
        response = requests.get(url, headers=headers)
        # 将返回的 JSON 字符串转换为 Python 字典
        response_dict = json.loads(response.text)
        # self.logger.info("available_balance:{}".format(response_dict['data']['available_balance'])
        #             +"---"+"cash_balance:{}".format(response_dict['data']['cash_balance']))
        return response_dict['data']['available_balance']

if __name__ == '__main__':
    paperjudge_assi=paperjudge_assistant()
    prompt_init=paperjudge_assi.prompt_init()
    