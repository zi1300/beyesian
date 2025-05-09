# EEG 时变自回归-贝叶斯网络分析工具

这个项目是一个用于分析游戏成瘾与非游戏成瘾受试者脑电数据的工具，实现了从数据预处理到构建时变自回归(TVAR)-贝叶斯网络的完整流程。该工具可以帮助研究人员探索脑电数据中蕴含的信息，特别是不同脑区之间的功能连接模式及其在游戏成瘾中的变化。

## 功能特点

- **模块化设计**: 将数据处理流程分为多个独立模块，便于后续修改和扩展
- **检查点机制**: 每个处理步骤都会保存检查点，避免程序出错后从头重新运行
- **丰富的可视化**: 提供多种可视化方法，直观展示分析结果
- **组间比较**: 可以比较游戏成瘾组和对照组的差异，帮助发现潜在的生物标记物

## 项目结构

```
├── config.py          # 配置参数模块
├── data_loader.py     # 数据加载模块
├── preprocessing.py   # 数据预处理模块
├── feature_extraction.py # 特征提取模块
├── tvar_model.py      # 时变自回归模型模块
├── bayesian_network.py # 贝叶斯网络模块
├── visualization.py   # 可视化模块
├── utils.py           # 工具函数模块
├── main.py            # 主程序
├── requirements.txt   # 依赖库列表
└── README.md          # 项目说明文档
```

## 安装依赖

建议使用Python 3.8或更高版本，并使用conda或virtualenv创建虚拟环境。

```bash
# 创建虚拟环境（可选）
conda create -n eeg_tvar python=3.8
conda activate eeg_tvar

# 安装依赖
pip install -r requirements.txt
```

## 数据要求

输入数据应满足以下条件：

1. 数据目录结构：包含A（游戏成瘾组）和C（对照组）两个子目录
2. 每个子目录中包含多个.mat文件，每个文件代表一个受试者的EEG数据
3. 每个.mat文件中应包含名为`eegdata`的变量，形状为[32, samples]

## 使用方法

### 基本用法

```bash
python main.py --data_dir /path/to/data
```

### 完整参数说明

```bash
python main.py --help
```

主要参数：

- `--data_dir`: 数据根目录，默认为`./data`
- `--addiction_dir`: 游戏成瘾组目录，默认为`<data_dir>/A`
- `--control_dir`: 对照组目录，默认为`<data_dir>/C`
- `--output_dir`: 结果保存目录，默认为`./results`
- `--checkpoint_dir`: 检查点保存目录，默认为`./checkpoints`
- `--force`: 强制重新计算所有步骤，忽略检查点
- `--skip_preprocess`: 跳过预处理步骤
- `--skip_feature`: 跳过特征提取步骤
- `--skip_tvar`: 跳过TVAR模型步骤
- `--skip_bayesian`: 跳过贝叶斯网络步骤
- `--skip_visualization`: 跳过可视化步骤
- `--seed`: 随机种子，默认为42
- `--verbose`: 显示详细日志

### 高级用法示例

1. 从头开始完整运行分析流程：

```bash
python main.py --data_dir /path/to/data --force
```

2. 只运行可视化部分（使用已有的检查点）：

```bash
python main.py --skip_preprocess --skip_feature --skip_tvar --skip_bayesian
```

3. 使用特定目录的游戏成瘾组和对照组数据：

```bash
python main.py --addiction_dir /path/to/addiction/data --control_dir /path/to/control/data
```

## 输出结果

程序运行后会生成以下结果：

1. `checkpoints/`: 各个处理步骤的检查点文件
2. `results/`: 分析结果和图表
   - `figures/`: 可视化图表，包括：
     - 脑电地形图
     - 功率谱密度图
     - 频段功率比较图
     - 时变自回归连接度图
     - 贝叶斯网络图
     - 组间差异网络图

## 分析流程

1. **数据加载**: 读取.mat文件中的EEG数据
2. **预处理**: 滤波、降采样、伪迹去除
3. **特征提取**: 提取时域特征、频域特征和连接度特征
4. **TVAR模型**: 构建时变自回归模型，估计动态功能连接
5. **贝叶斯网络**: 构建贝叶斯网络，分析因果关系
6. **可视化**: 生成各种可视化图表，直观展示结果

## 注意事项

- 确保输入数据符合要求的格式
- 处理大量数据时，可能需要较高的计算资源
- 如果出现内存不足的问题，可以尝试降低采样率或减少通道数
- 对于复杂的分析，建议先在小数据集上测试

## 常见问题

1. **报错：找不到.mat文件**
   - 检查数据目录结构是否正确
   - 检查文件扩展名是否为.mat

2. **报错：找不到eegdata变量**
   - 检查.mat文件中的变量名是否为eegdata
   - 如果变量名不同，可以修改data_loader.py中的相关代码

3. **处理速度很慢**
   - 尝试减少数据量或通道数
   - 确保使用了检查点机制，避免重复计算

## 扩展和自定义

1. **添加新的特征提取方法**：
   - 在feature_extraction.py中添加新的特征提取函数
   - 在extract_features函数中调用新添加的函数

2. **使用不同的连接度指标**：
   - 在tvar_model.py中修改或添加连接度计算方法
   - 更新get_connectivity_measures函数

3. **自定义可视化**：
   - 在visualization.py中添加新的可视化函数
   - 在create_dashboard函数中调用新添加的函数

## 引用和致谢

如果您在研究中使用了本工具，请引用：

```
待添加引用信息
```

## 版权和许可

本项目采用MIT许可证。详见LICENSE文件。

## 联系方式

如有问题或建议，请联系：[待添加联系方式]
