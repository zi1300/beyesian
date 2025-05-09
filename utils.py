"""
工具函数模块，包括检查点管理、日志记录等功能
修复了checkpoint装饰器的参数传递问题
"""
import os
import pickle
import logging
from time import time
from functools import wraps
import numpy as np
import inspect

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("eeg_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EEG_Analysis")

def save_checkpoint(data, file_path):
    """保存检查点

    Args:
        data: 要保存的数据
        file_path: 检查点文件路径
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"检查点已保存到: {file_path}")
        return True
    except Exception as e:
        logger.error(f"保存检查点失败: {e}")
        return False

def load_checkpoint(file_path):
    """加载检查点

    Args:
        file_path: 检查点文件路径

    Returns:
        加载的数据，如果文件不存在则返回None
    """
    if not os.path.exists(file_path):
        logger.info(f"检查点文件不存在: {file_path}")
        return None

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"检查点已加载: {file_path}")
        return data
    except Exception as e:
        logger.error(f"加载检查点失败: {e}")
        return None

def checkpoint(stage):
    """装饰器，用于实现函数级别的检查点功能

    Args:
        stage: 阶段名称，用于构建检查点文件名

    Returns:
        装饰后的函数
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 从kwargs中提取组别信息，如果有的话
            group = kwargs.pop('group', None) if 'group' not in inspect.signature(func).parameters else None

            # 导入配置模块
            import config

            # 生成检查点文件路径
            checkpoint_path = config.get_checkpoint_path(stage, group)

            # 尝试加载检查点
            result = load_checkpoint(checkpoint_path)

            # 如果检查点不存在或强制重新计算，则执行函数
            force = kwargs.pop('force', False) if 'force' not in inspect.signature(func).parameters else kwargs.get('force', False)
            if result is None or force:
                logger.info(f"开始执行{stage}阶段...")
                start_time = time()

                # 如果我们之前移除了group参数但函数实际需要它，则添加回来
                if group is not None and 'group' in inspect.signature(func).parameters:
                    kwargs['group'] = group

                result = func(*args, **kwargs)
                elapsed_time = time() - start_time
                logger.info(f"{stage}阶段完成，耗时: {elapsed_time:.2f}秒")

                # 保存检查点
                save_checkpoint(result, checkpoint_path)
            else:
                logger.info(f"已从检查点加载{stage}阶段结果")

            return result
        return wrapper
    return decorator

def timer(func):
    """计时器装饰器，用于测量函数执行时间

    Args:
        func: 要装饰的函数

    Returns:
        装饰后的函数
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        elapsed_time = time() - start_time
        logger.info(f"函数 {func.__name__} 执行耗时: {elapsed_time:.2f}秒")
        return result
    return wrapper

def set_seed(seed):
    """设置随机种子，保证结果可复现

    Args:
        seed: 随机种子
    """
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"已设置随机种子: {seed}")

def get_channel_indices(channel_names, target_channels):
    """获取目标通道的索引

    Args:
        channel_names: 所有通道名称列表
        target_channels: 目标通道名称列表

    Returns:
        目标通道的索引列表
    """
    indices = []
    for ch in target_channels:
        try:
            indices.append(channel_names.index(ch))
        except ValueError:
            logger.warning(f"通道 {ch} 不存在")
    return indices

def compute_connectivity_matrix(data, method='correlation'):
    """计算连接矩阵

    Args:
        data: 形状为[channels, samples]的数据
        method: 连接度计算方法，可选值: 'correlation', 'coherence', 'pli'

    Returns:
        连接矩阵，形状为[channels, channels]
    """
    n_channels = data.shape[0]
    conn_matrix = np.zeros((n_channels, n_channels))

    if method == 'correlation':
        # 计算皮尔逊相关系数
        for i in range(n_channels):
            for j in range(n_channels):
                conn_matrix[i, j] = np.corrcoef(data[i], data[j])[0, 1]

    # 其他连接度计算方法可以在这里添加

    return conn_matrix

def create_time_frequency_data(data, fs, window_size, window_step, freq_bands):
    """创建时频数据

    Args:
        data: EEG数据，形状为[channels, samples]
        fs: 采样率
        window_size: 窗口大小
        window_step: 窗口步长
        freq_bands: 频段字典

    Returns:
        时频数据字典
    """
    from scipy import signal

    n_channels, n_samples = data.shape

    # 计算滑动窗口数量
    n_windows = 1 + (n_samples - window_size) // window_step

    # 初始化结果字典
    tf_data = {band: np.zeros((n_channels, n_windows)) for band in freq_bands}

    for i in range(n_windows):
        start = i * window_step
        end = start + window_size

        # 对每个通道的窗口数据进行频谱分析
        freqs, psd = signal.welch(data[:, start:end], fs=fs, nperseg=min(window_size, 256))

        # 对每个频段计算功率
        for band, (fmin, fmax) in freq_bands.items():
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            tf_data[band][:, i] = np.mean(psd[:, idx], axis=1)

    return tf_data