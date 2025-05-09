"""
配置文件，包含所有可配置的参数
"""
import os
from datetime import datetime

# 路径配置
DATA_ROOT_DIR = "./data"  # 数据根目录
ADDICTION_DIR = os.path.join(DATA_ROOT_DIR, "A")  # 游戏成瘾组目录
CONTROL_DIR = os.path.join(DATA_ROOT_DIR, "C")  # 对照组目录
CHECKPOINT_DIR = "./checkpoints"  # 检查点保存目录
RESULTS_DIR = "./results"  # 结果保存目录
FIG_DIR = os.path.join(RESULTS_DIR, "figures")  # 图表保存目录

# 创建必要的目录
for directory in [CHECKPOINT_DIR, RESULTS_DIR, FIG_DIR]:
    os.makedirs(directory, exist_ok=True)

# 数据参数
SAMPLING_RATE = 1200  # 原始采样率(Hz)
DOWN_SAMPLING_RATE = 250  # 降采样后的采样率(Hz)
NUM_CHANNELS = 32  # EEG通道数

# 预处理参数
FILTER_LOW = 1.0  # 高通滤波器截止频率
FILTER_HIGH = 40.0  # 低通滤波器截止频率
NOTCH_FREQ = 50.0  # 陷波滤波器频率(用于去除电源干扰)
EPOCH_TIME = [-0.2, 1.0]  # 事件相关的epoch时间范围(秒)
ICA_COMPONENTS = 15  # ICA分解中保留的成分数量
REJECT_CRITERIA = {
    'eeg': 100e-6  # 波幅超过100μV的数据点将被标记为伪迹
}

# TVAR模型参数
MODEL_ORDER = 5  # AR模型阶数
WINDOW_SIZE = 30  # 滑动窗口大小(数据点)
WINDOW_STEP = 5  # 滑动窗口步长(数据点)
LAMBDA = 0.99  # 遗忘因子
BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 40)
}

# 贝叶斯网络参数
BN_ITERATIONS = 1000  # MCMC迭代次数
BN_BURN_IN = 100  # burn-in样本数
BN_K2_MAX_PARENTS = 3  # 结构学习时每个节点的最大父节点数

# 可视化参数
PLOT_DPI = 300  # 图像DPI
PLOT_FORMAT = 'png'  # 图像格式
COLOR_ADDICTION = '#E41A1C'  # 游戏成瘾组颜色
COLOR_CONTROL = '#377EB8'  # 对照组颜色
MONTAGE_NAME = 'standard_1020'  # EEG电极布局

# 随机种子，保证结果可复现
RANDOM_SEED = 42


# 检查点文件名模板
def get_checkpoint_path(stage, group=None):
    """获取检查点文件路径

    Args:
        stage: 阶段名称，如'preprocess', 'feature', 'model'
        group: 组别，如'A'或'C'，如果是处理所有数据则为None

    Returns:
        检查点文件路径
    """
    if group:
        return os.path.join(CHECKPOINT_DIR, f"{stage}_{group}.pkl")
    else:
        return os.path.join(CHECKPOINT_DIR, f"{stage}.pkl")


# 结果文件名模板
def get_result_path(name, extension=None):
    """获取结果文件路径

    Args:
        name: 结果名称
        extension: 文件扩展名，默认为None

    Returns:
        结果文件路径
    """
    if extension is None:
        return os.path.join(RESULTS_DIR, name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(RESULTS_DIR, f"{name}_{timestamp}.{extension}")


# 图表文件名模板
def get_figure_path(name):
    """获取图表文件路径

    Args:
        name: 图表名称

    Returns:
        图表文件路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(FIG_DIR, f"{name}_{timestamp}.{PLOT_FORMAT}")