"""
特征提取模块，从预处理后的EEG数据中提取特征
"""
import numpy as np
import pandas as pd
from scipy import signal, stats
import mne
from utils import logger, checkpoint, timer, compute_connectivity_matrix
import config


@timer
def extract_time_domain_features(raw):
    """提取时域特征

    Args:
        raw: mne.io.Raw对象

    Returns:
        包含时域特征的字典
    """
    # 获取数据
    data = raw.get_data()

    # 初始化特征字典
    features = {}

    # 计算统计特征
    features['mean'] = np.mean(data, axis=1)
    features['std'] = np.std(data, axis=1)
    features['var'] = np.var(data, axis=1)
    features['skew'] = stats.skew(data, axis=1)
    features['kurtosis'] = stats.kurtosis(data, axis=1)
    features['ptp'] = np.ptp(data, axis=1)  # 峰峰值
    features['rms'] = np.sqrt(np.mean(np.square(data), axis=1))  # 均方根

    # 计算Hjorth参数 - 活动度（方差）
    features['hjorth_activity'] = np.var(data, axis=1)

    # 计算Hjorth参数 - 移动度
    diff1 = np.diff(data, axis=1)
    var_diff1 = np.var(diff1, axis=1)
    features['hjorth_mobility'] = np.sqrt(var_diff1 / (features['var'] + 1e-10))

    # 计算Hjorth参数 - 复杂度
    diff2 = np.diff(diff1, axis=1)
    var_diff2 = np.var(diff2, axis=1)
    mobility_diff1 = np.sqrt(var_diff2 / (var_diff1 + 1e-10))
    features['hjorth_complexity'] = mobility_diff1 / (features['hjorth_mobility'] + 1e-10)

    return features


@timer
def extract_frequency_domain_features(raw, bands=None):
    """提取频域特征

    Args:
        raw: mne.io.Raw对象
        bands: 频段字典，默认为None（使用config中的设置）

    Returns:
        包含频域特征的字典
    """
    if bands is None:
        bands = config.BANDS

    # 获取数据和采样率
    data = raw.get_data()
    sfreq = raw.info['sfreq']

    # 初始化特征字典
    features = {}

    # 计算功率谱密度（使用Welch方法）
    freqs, psd = signal.welch(data, fs=sfreq, nperseg=min(512, data.shape[1]))

    # 计算各频段的绝对功率
    for band_name, (fmin, fmax) in bands.items():
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        band_power = np.mean(psd[:, idx], axis=1)
        features[f'power_{band_name}'] = band_power

    # 计算总功率（1-40Hz）
    idx_total = np.logical_and(freqs >= 1, freqs <= 40)
    total_power = np.sum(psd[:, idx_total], axis=1)

    # 计算各频段的相对功率
    for band_name in bands.keys():
        features[f'rel_power_{band_name}'] = features[f'power_{band_name}'] / (total_power + 1e-10)

    # 计算频段功率比
    features['theta_beta_ratio'] = features['power_theta'] / (features['power_beta'] + 1e-10)
    features['alpha_beta_ratio'] = features['power_alpha'] / (features['power_beta'] + 1e-10)
    features['delta_beta_ratio'] = features['power_delta'] / (features['power_beta'] + 1e-10)

    # 计算频谱熵
    psd_norm = psd / (np.sum(psd, axis=1, keepdims=True) + 1e-10)
    features['spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm + 1e-10), axis=1)

    # 计算频谱质心
    features['spectral_centroid'] = np.sum(freqs[np.newaxis, :] * psd, axis=1) / (np.sum(psd, axis=1) + 1e-10)

    return features


@timer
def extract_connectivity_features(raw):
    """提取连接度特征

    Args:
        raw: mne.io.Raw对象

    Returns:
        包含连接度特征的字典
    """
    # 获取数据
    data = raw.get_data()

    # 初始化特征字典
    features = {}

    # 计算相关矩阵
    conn_matrix = compute_connectivity_matrix(data, method='correlation')

    # 提取上三角部分（不包括对角线）
    triu_indices = np.triu_indices(conn_matrix.shape[0], k=1)
    features['corr_values'] = conn_matrix[triu_indices]

    # 计算平均连接度
    features['mean_connectivity'] = np.mean(np.abs(features['corr_values']))

    # 计算各通道的平均连接度
    features['node_connectivity'] = np.mean(np.abs(conn_matrix), axis=1)

    return features


@checkpoint("extract_features")
def extract_features(raw, subject_id=None):
    """从预处理后的EEG数据中提取所有特征

    Args:
        raw: mne.io.Raw对象
        subject_id: 受试者ID，默认为None

    Returns:
        包含所有特征的字典
    """
    logger.info(f"开始提取受试者{subject_id}的特征...")

    # 提取时域特征
    time_features = extract_time_domain_features(raw)

    # 提取频域特征
    freq_features = extract_frequency_domain_features(raw)

    # 提取连接度特征
    conn_features = extract_connectivity_features(raw)

    # 合并所有特征
    all_features = {}
    all_features.update(time_features)
    all_features.update(freq_features)
    all_features.update(conn_features)

    logger.info(f"受试者{subject_id}的特征提取完成")

    return all_features


@checkpoint("extract_group_features")
def extract_group_features(group_data, group_name):
    """提取一个组的所有受试者的特征

    Args:
        group_data: 包含该组所有受试者预处理后数据的字典 {subject_id: raw}
        group_name: 组名称

    Returns:
        包含所有特征的字典 {subject_id: features}
    """
    logger.info(f"开始提取{group_name}组的特征...")

    all_features = {}

    for subject_id, raw in group_data.items():
        # 提取该受试者的特征
        features = extract_features(raw, subject_id=subject_id, group=group_name)
        all_features[subject_id] = features

    logger.info(f"{group_name}组的特征提取完成，共{len(all_features)}个受试者")

    return all_features


@checkpoint("extract_all_features")
def extract_all_features(all_processed_data):
    """提取所有受试者的特征

    Args:
        all_processed_data: 包含所有预处理后数据的字典 {'A': processed_addiction_data, 'C': processed_control_data}

    Returns:
        包含所有特征的字典 {'A': addiction_features, 'C': control_features}
    """
    logger.info("开始提取所有受试者的特征...")

    # 提取游戏成瘾组特征
    addiction_features = extract_group_features(all_processed_data['A'], 'A')

    # 提取对照组特征
    control_features = extract_group_features(all_processed_data['C'], 'C')

    all_features = {
        'A': addiction_features,
        'C': control_features
    }

    logger.info("所有受试者的特征提取完成")

    return all_features


@timer
def prepare_features_for_tvar(raw, window_size=None, window_step=None):
    """准备用于TVAR模型的数据

    Args:
        raw: mne.io.Raw对象
        window_size: 窗口大小（数据点），默认为None（使用config中的设置）
        window_step: 窗口步长（数据点），默认为None（使用config中的设置）

    Returns:
        准备好的数据和时间点
    """
    if window_size is None:
        window_size = config.WINDOW_SIZE

    if window_step is None:
        window_step = config.WINDOW_STEP

    # 获取数据和采样率
    data = raw.get_data()
    sfreq = raw.info['sfreq']

    # 计算滑动窗口数量
    n_samples = data.shape[1]
    n_windows = 1 + (n_samples - window_size) // window_step

    # 准备结果
    windows_data = []
    time_points = []

    for i in range(n_windows):
        start = i * window_step
        end = start + window_size

        # 提取窗口数据
        window_data = data[:, start:end]

        # 保存数据
        windows_data.append(window_data)

        # 保存时间点（取窗口中点）
        time_points.append((start + end) / 2 / sfreq)

    return windows_data, np.array(time_points)


@checkpoint("prepare_tvar_data")
def prepare_tvar_data(raw, subject_id=None, bands=None):
    """准备用于TVAR模型的带通滤波数据

    Args:
        raw: mne.io.Raw对象
        subject_id: 受试者ID，默认为None
        bands: 频段字典，默认为None（使用config中的设置）

    Returns:
        准备好的TVAR数据字典
    """
    if bands is None:
        bands = config.BANDS

    logger.info(f"准备受试者{subject_id}的TVAR数据...")

    tvar_data = {}

    # 对每个频段进行带通滤波
    for band_name, (fmin, fmax) in bands.items():
        # 应用带通滤波器
        raw_band = raw.copy().filter(l_freq=fmin, h_freq=fmax, method='fir',
                                     phase='zero-double', fir_design='firwin')

        # 准备TVAR数据
        windows_data, time_points = prepare_features_for_tvar(raw_band)

        tvar_data[band_name] = {
            'windows': windows_data,
            'times': time_points
        }

    logger.info(f"受试者{subject_id}的TVAR数据准备完成")

    return tvar_data


@checkpoint("prepare_group_tvar_data")
def prepare_group_tvar_data(group_data, group_name):
    """准备一个组的所有受试者的TVAR数据

    Args:
        group_data: 包含该组所有受试者预处理后数据的字典 {subject_id: raw}
        group_name: 组名称

    Returns:
        包含所有TVAR数据的字典 {subject_id: tvar_data}
    """
    logger.info(f"开始准备{group_name}组的TVAR数据...")

    all_tvar_data = {}

    for subject_id, raw in group_data.items():
        # 准备该受试者的TVAR数据
        tvar_data = prepare_tvar_data(raw, subject_id=subject_id, group=group_name)
        all_tvar_data[subject_id] = tvar_data

    logger.info(f"{group_name}组的TVAR数据准备完成，共{len(all_tvar_data)}个受试者")

    return all_tvar_data


@checkpoint("prepare_all_tvar_data")
def prepare_all_tvar_data(all_processed_data):
    """准备所有受试者的TVAR数据

    Args:
        all_processed_data: 包含所有预处理后数据的字典 {'A': processed_addiction_data, 'C': processed_control_data}

    Returns:
        包含所有TVAR数据的字典 {'A': addiction_tvar_data, 'C': control_tvar_data}
    """
    logger.info("开始准备所有受试者的TVAR数据...")

    # 准备游戏成瘾组的TVAR数据
    addiction_tvar_data = prepare_group_tvar_data(all_processed_data['A'], 'A')

    # 准备对照组的TVAR数据
    control_tvar_data = prepare_group_tvar_data(all_processed_data['C'], 'C')

    all_tvar_data = {
        'A': addiction_tvar_data,
        'C': control_tvar_data
    }

    logger.info("所有受试者的TVAR数据准备完成")

    return all_tvar_data


def features_to_dataframe(all_features):
    """将特征转换为DataFrame格式

    Args:
        all_features: 包含所有特征的字典 {'A': addiction_features, 'C': control_features}

    Returns:
        包含所有特征的DataFrame
    """
    # 准备数据
    data = []

    # 处理游戏成瘾组
    for subject_id, features in all_features['A'].items():
        for feature_name, feature_values in features.items():
            if feature_name != 'corr_values':  # 跳过连接矩阵数据
                if np.isscalar(feature_values):
                    data.append({
                        'subject_id': subject_id,
                        'group': 'A',
                        'feature': feature_name,
                        'value': feature_values
                    })
                else:
                    for ch_idx, value in enumerate(feature_values):
                        data.append({
                            'subject_id': subject_id,
                            'group': 'A',
                            'feature': f"{feature_name}_ch{ch_idx + 1}",
                            'value': value
                        })

    # 处理对照组
    for subject_id, features in all_features['C'].items():
        for feature_name, feature_values in features.items():
            if feature_name != 'corr_values':  # 跳过连接矩阵数据
                if np.isscalar(feature_values):
                    data.append({
                        'subject_id': subject_id,
                        'group': 'C',
                        'feature': feature_name,
                        'value': feature_values
                    })
                else:
                    for ch_idx, value in enumerate(feature_values):
                        data.append({
                            'subject_id': subject_id,
                            'group': 'C',
                            'feature': f"{feature_name}_ch{ch_idx + 1}",
                            'value': value
                        })

    # 创建DataFrame
    df = pd.DataFrame(data)

    return df