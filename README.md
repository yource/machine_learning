# machine_learning
---- 机器学习练习实践 ----  
  
## conda命令
```
# 创建环境
conda create -n ml python=3.9
# 激活环境
conda activate ml
# 查看环境列表
conda env list
# 查看当前环境下用conda安装的库
conda list
```
环境搭建 参考 https://www.cnblogs.com/yangshifu/p/17071889.html

## 机器学习基础
### 二分类问题  
标签为0和1  
loss = 'binary_crossentropy' 二元交叉熵  
optimizer = 'rmsprop'  
metrics=['accuracy']  
最后一层使用sigmoid，输出0~1之间的概率值
>layers.Dense(1, activation='sigmoid')  

### 多分类问题  
标签为0/1/2.../n  
loss = 'categorical_crossentropy' 分类交叉熵  
optimizer = 'rmsprop'  
metrics=['accuracy'] 
最后一层使用softmax，输出总和为1的各类型概率
>layers.Dense(n, activation='softmax')
### 标量回归
标签为单一连续值  
loss = 'mse' 均方误差  
optimizer = 'rmsprop'  
metrics=['MAE'] 平均绝对误差  
最后一层不使用激活函数，输出任意范围内的值
>layers.Dense(1) 
  
## 数据处理流程  
1. 导入数据
2. 选取需要的列和行
3. 数据归一化，减去平均值、除以标准差
4. 拆分数据 训练集、验证集、测试集
5. 配置模型 进行训练
6. 考虑是否进行归一化，简单平均数、移动平均数、加权平均数
7. 考虑是否包含时间、时间转化成周期
8. 如何考察准确度
