## 神经网络结构化稀疏剪枝

...暂时懒得写，之后再来补充

## 介绍

使用TensorFlow2.0搭建的多个卷积网络框架，自定义训练过程，自定义学习率，用于结构化稀疏剪枝。

...
如果对本代码有疑问，请联系我：
```
@author: Kang Xiatao (kangxiatao@gmail.com)
```

## 结构
 - experiment文件夹中是测试程序和数据分析程序
 - models文件夹中包含lenet、resnet、vgg三个模型
 - train_model用于存放权重、评估信息等
 - utility自定义的辅组函数
 - main.py and main_tpu.py 为主函数（对应在GPU or TPU运行）
 - mycallback.py 自定义回调
 - myparser.py 超参数设定相关
 - penalty.py 惩罚相关
 - prune.py 剪枝相关

## 环境

* Python == 3.8.5
* tensorflow == 2.3.1
* numpy == 1.18.5
* pandas == 1.1.3
* sklearn == 0.23.2

## 运行

无惩罚原始网络
```
python main.py --model vgg --data_name 'cifar10' --train 1 --prune 1
```
使用TPU训练
```
python main_tpu.py --model vgg --data_name 'cifar10' --train 1 --prune 1
```
L1稀疏惩罚
```
python main.py --model vgg --data_name 'cifar10' --train 1 --prune 1 --l1_value 0.0001
```
L2惩罚
```
python main.py --model vgg --data_name 'cifar10' --train 1 --prune 1 --l2_value 0.01
```
Group Lasso惩罚
```
python main.py --model vgg --data_name 'cifar10' --train 1 --prune 1 --gl_a 0.001
```
角相异惩罚（我们的）
```
python main.py --model='vgg'  --data_name='cifar10' --var_2 0.0001 --gl_a 0.001 --is_restore 1 --restore_path=trained_model/vgg/cifar10/_gl1_0.001_gl2_0.001_/
```
拟合解惩罚（我们的）
```
python main.py --model='vgg'  --data_name='cifar10' --prop_1 0.001
```

- 模型（model）可选：```lenet, vgg, resnet```（默认为lenet5，vgg16，resnet18）

- 数据集（data_name）可选：```mnist, cifar10, cifar100```（其他数据集需本地加载）

- 其他参数：

    | 参数                | 备注                                              |
    | :------------------ | :------------------------------------------------ |
    | restore_path = ''   | # 权重恢复路径                                    |
    | train = 1           | # 训练                                            |
    | test = 1            | # 评估                                            |
    | prune = 1           | # 剪枝                                            |
    | prior_prune = 0     | # 预剪枝                                          |
    | store_weight = 0    | # 保存权重                                        |
    | is_restore = 0      | # 恢复权重                                        |
    | init_lr = 0.1       | # 学习率                                          |
    | epochs = 200        | # 训练回合数                                      |
    | batch_size = 256    | # 一个批次的数据大小                              |
    | threshold = 0.001   | # 权重剪枝阈值                                    |
    | penalty_ratio = 1.0 | # 角相异惩罚比例                                  |
    | stop_acc = 0.992    | # 早停精度（暂时没用）                            |
    | l1_value = 0.000    | # L1范数惩罚超参                                  |
    | l2_value = 0.00     | # L2范数惩罚超参                                  |
    | gl_1 = 0.000        | # Group Lasso惩罚超参                             |
    | gl_2 = 0.000        | # Group Lasso惩罚超参                             |
    | gl_a = 0.000        | # Group Lasso惩罚超参（细分到每一个过滤器和通道） |
    | var_1 = 0.0         | # 角相异惩罚超参（过滤器角）                      |
    | var_2 = 0.0         | # 角相异惩罚超参（通道角）                        |
    | prop_1 = 0.000      | # 拟合解惩罚                                      |
    | prop_2 = 0.000      | # 拟合解惩罚（暂时没用）                          |
    | prop_a = 0.000      | # 拟合解惩罚（暂时没用）                          |

## 数据集
...待补充

## 结果
...待补充

## 
...待补充

