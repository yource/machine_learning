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
  
## 各品种一手价格 （202306）
- 黑色金属-铁矿石 19k
- 黑色金属-螺纹钢 8k
- 黑色金属-热卷 8k
- 黑色金属-不锈钢 12k
- 黑色金属-硅铁 8k
- 黑色金属-锰硅 7k
- 有色金属-沪铜 58k
- 有色金属-国际铜 48k
- 有色金属-沪铝 18k
- 有色金属-氧化铝 10k
- 有色金属-沪锌 20k
- 有色金属-沪铅 13k
- 有色金属-沪镍 47k
- 有色金属-沪锡 76k
- 有色金属-工业硅 11k
- 贵金属-沪金 85k
- 贵金属-沪银 17k
- 油脂油料-豆一 8k
- 油脂油料-豆二 7k
- 油脂油料-豆油 14k
- 油脂油料-菜油 14k
- 油脂油料-棕榈油 13k
- 油脂油料-豆粕 6k
- 油脂油料-菜粕 6k
- 农产品-玉米 4k
- 农产品-淀粉 4k
- 农产品-鸡蛋 7k
- 农产品-棉花 14k
- 农产品-棉纱 17k
- 农产品-苹果 17k
- 农产品-红枣 11k
- 农产品-花生 7k
- 农产品-粳米 4k
- 农产品-生猪 41k
- 农产品-白糖 15k
- 能源化工-原油 154k
- 能源化工-燃油 8k
- 能源化工-LPG 14k
- 能源化工-低硫燃油 9k
- 能源化工-沥青 9k
- 能源化工-乙二醇 7k
- 能源化工-塑料 6k
- 能源化工-PTA 5k
- 能源化工-聚丙烯 5k
- 能源化工-苯乙烯 6k
- 能源化工-橡胶 22k
- 能源化工-20号胶 17k
- 能源化工-纸浆 10k
- 能源化工-短纤 6k
- 能源化工-甲醇 4k
- 能源化工-PVC 4k
- 能源化工-纯碱 8k
- 能源化工-玻璃 8k
- 能源化工-尿素 6k
- 煤炭板块-焦煤 28k
- 煤炭板块-焦炭 75k