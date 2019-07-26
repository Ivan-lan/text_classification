## 文本分类算法

基于深度学习的文本分类算法

- TextCNN
- TextRNN
---

#### 使用

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

![image](https://github.com/Ivan-lan/text_classification/blob/master/images/textrnn1.png)