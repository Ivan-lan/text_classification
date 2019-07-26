## 文本分类算法

基于深度学习的文本分类算法

- TextCNN
- TextRNN
---

#### 使用

- 数据准备

格式：带标签的txt文本: 【标签\t文本】

```
体育	马晓旭意外受伤让国奥警惕 无奈大雨格外青睐殷家军记者傅亚雨沈阳报道 来到沈阳，国奥队依然没有摆脱雨水的困扰。
游戏	网元与POPCAP从蓄势待发到锐意进取祖玛、宝石迷阵2来了！经过网元网的不懈努力，休闲游戏巨鳄宝开公司的一系列经典游戏终于即将在大陆地区销售，简单、新奇、有趣于一身的休闲小游戏再次来到了玩家们的面前。
```

- TextCNN
```
from textCNN.textCNN import TextCNN

model = TextCNN()  

# 传入训练集和验证集，训练模型
model.train("data\\train_data.txt","data\\val_data.txt")

# 传入测试集，评估模型
model.test("data\\test_data.txt")

# 在待分类文本上预测
model.predict("data\\pred_data.txt")

# 加载训练好的模型
model.load_model("checkpoints\\best_validation")

```

---
- TextRNN

```
from textRNN.textRNN import TextRNN

model = TextRNN()  

# 传入训练集和验证集，训练模型
model.train("data\\train_data.txt","data\\val_data.txt")

# 传入测试集，评估模型
model.test("data\\test_data.txt")

# 在待分类文本上预测
model.predict("data\\pred_data.txt")

# 加载训练好的模型
model.load_model("checkpoints\\best_validation")

```

#### 原理

- TextCNN

![image](https://github.com/Ivan-lan/text_classification/blob/master/images/textcnn1.png)

![image](https://github.com/Ivan-lan/text_classification/blob/master/images/textcnn2.png)

-  TextRNN

![image](https://github.com/Ivan-lan/text_classification/blob/master/images/textrnn1.jpg)