## 文件说明 
- datas文件夹为论文集所在文件夹  
- log文件夹为代码运行时，日志文件所在文件夹  
- output文件夹为评价过程中生成的文件存放处  
  
|文件名|  文件描述 |
| :----- | :----------- |
| config.ini |  配置文件，配置大模型接口的API-key和路径 |
| logger_config.py |  日志处理工具 |
| preprocess.py | 论文预处理代码 |
| chat_with_LLM.py | 调用kimi大模型接口代码 |
| LBFE.py | 读取论文集，调用LLM大模型接口辅助论文评价，生成论文质量特征集 |
| CLM.py | 基于LBFE生成的论文特征集，调用熵权法和TOPSIS算法生成综合评分指标并排序 |
| main.ipynb | 主函数，调用以上代码完成赛题任务 |


## 运行说明
- 运行环境：python3.8
- 运行方式：在jupyter notebook中运行main.ipynb  
  
## 运行步骤
1. 安装依赖包  使用`pip install -r requirements.txt`命令
2. 配置config.ini文件，修改API-key和路径
3. 运行main.ipynb