
#### 疫情期间网民情绪识别项目

  情感分析项目，即文本分类任务
    
- 代码：python==3.6.9，基于google的TPU v3-8

- 数据：疫情期间网民的微博相关状态评论，共计10万标记数据，90万未标记数据，1万测试数据

    数据文件名train.txt，dev.txt
    
    数据格式：
    
    __label__0  今天感冒了，怎么办？
    
    __label__1  今天感冒了，怎么办？
    
    __label__2  今天感冒了，怎么办？
 
- 框架：torch-xla-nightly

- 预训练模型：chinese_roberta_wwm_ext_pytorch

##### 训练

    sh script/run.sh
    
#### 优化记录

- 1、fasttext : f1 50

  存在预测偏重中性分类，其余两类召回率接近于0
  
- 2、bert 