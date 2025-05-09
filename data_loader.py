"""
数据加载模块，负责读取.mat文件中的EEG数据
修复了组参数的处理
"""
import os
import glob
import numpy as np
import scipy.io as sio
from utils import logger, checkpoint
import config

def list_mat_files(directory):
    """列出目录中的所有.mat文件

    Args:
        directory: 目录路径

    Returns:
        .mat文件路径列表
    """
    if not os.path.exists(directory):
        logger.error(f"目录不存在: {directory}")
        return []

    mat_files = glob.glob(os.path.join(directory, "*.mat"))
    logger.info(f"在 {directory} 中找到 {len(mat_files)} 个.mat文件")
    return mat_files

def load_mat_file(file_path):
    """加载单个.mat文件中的EEG数据

    Args:
        file_path: .mat文件路径

    Returns:
        eeg_data: EEG数据 (channels x samples)
        subject_id: 受试者ID
    """
    try:
        mat_data = sio.loadmat(file_path)

        # 尝试获取eegdata变量
        if 'eegdata' in mat_data:
            eeg_data = mat_data['eegdata']
        else:
            # 如果没有找到eegdata，尝试寻找其他可能的变量名
            for key in mat_data.keys():
                if isinstance(mat_data[key], np.ndarray) and len(mat_data[key].shape) == 2:
                    # 寻找形状合适的数组
                    if mat_data[key].shape[0] == config.NUM_CHANNELS or mat_data[key].shape[1] == config.NUM_CHANNELS:
                        eeg_data = mat_data[key]
                        logger.warning(f"未找到'eegdata'变量，使用'{key}'变量代替")
                        break
            else:
                raise KeyError("未找到EEG数据变量")

        # 确保数据形状为 channels x samples
        if eeg_data.shape[0] != config.NUM_CHANNELS and eeg_data.shape[1] == config.NUM_CHANNELS:
            eeg_data = eeg_data.T

        # 提取受试者ID（从文件名中）
        subject_id = os.path.basename(file_path).split('.')[0]

        logger.info(f"已加载文件: {file_path}, 数据形状: {eeg_data.shape}")
        return eeg_data, subject_id

    except Exception as e:
        logger.error(f"加载文件 {file_path} 失败: {e}")
        return None, None

@checkpoint("load_data")
def load_group_data(group_dir, group_name):
    """加载一个组的所有EEG数据

    Args:
        group_dir: 组目录路径
        group_name: 组名称（'A'或'C'）

    Returns:
        包含所有受试者EEG数据的字典: {subject_id: eeg_data}
    """
    mat_files = list_mat_files(group_dir)

    if not mat_files:
        logger.error(f"在{group_dir}中没有找到.mat文件")
        return {}

    all_data = {}

    for file_path in mat_files:
        eeg_data, subject_id = load_mat_file(file_path)

        if eeg_data is not None:
            all_data[f"{group_name}_{subject_id}"] = eeg_data

    logger.info(f"成功加载{group_name}组{len(all_data)}个受试者的数据")
    return all_data

@checkpoint("load_all_data")
def load_all_data(addiction_dir=None, control_dir=None, force=False):
    """加载所有EEG数据（包括游戏成瘾组和对照组）

    Args:
        addiction_dir: 游戏成瘾组目录路径，默认使用config中的设置
        control_dir: 对照组目录路径，默认使用config中的设置
        force: 是否强制重新加载，默认为False

    Returns:
        包含所有数据的字典: {'A': addiction_data, 'C': control_data}
        其中addiction_data和control_data是包含各自组受试者数据的字典
    """
    if addiction_dir is None:
        addiction_dir = config.ADDICTION_DIR

    if control_dir is None:
        control_dir = config.CONTROL_DIR

    logger.info("开始加载所有EEG数据...")

    # 加载游戏成瘾组数据
    addiction_data = load_group_data(addiction_dir, 'A', group='A', force=force)

    # 加载对照组数据
    control_data = load_group_data(control_dir, 'C', group='C', force=force)

    all_data = {
        'A': addiction_data,
        'C': control_data
    }

    total_subjects = len(addiction_data) + len(control_data)
    logger.info(f"总共加载了{total_subjects}个受试者的数据")

    return all_data

def get_subject_info(all_data):
    """获取受试者信息统计

    Args:
        all_data: 由load_all_data返回的数据字典

    Returns:
        包含受试者信息的字典
    """
    addiction_subjects = list(all_data['A'].keys())
    control_subjects = list(all_data['C'].keys())

    info = {
        'addiction_count': len(addiction_subjects),
        'control_count': len(control_subjects),
        'addiction_subjects': addiction_subjects,
        'control_subjects': control_subjects,
        'total_subjects': len(addiction_subjects) + len(control_subjects)
    }

    return info

# 在模块导入时执行，测试文件是否存在
def check_data_dirs():
    """检查数据目录是否存在"""
    if not os.path.exists(config.DATA_ROOT_DIR):
        logger.warning(f"数据根目录不存在: {config.DATA_ROOT_DIR}")

    if not os.path.exists(config.ADDICTION_DIR):
        logger.warning(f"游戏成瘾组目录不存在: {config.ADDICTION_DIR}")

    if not os.path.exists(config.CONTROL_DIR):
        logger.warning(f"对照组目录不存在: {config.CONTROL_DIR}")

# 在模块导入时执行目录检查
check_data_dirs()