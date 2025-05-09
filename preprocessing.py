"""
EEG数据预处理模块，包括滤波、降采样、伪迹去除等功能
"""
import numpy as np
import mne
from scipy import signal
from utils import logger, checkpoint, timer
import config

# 设置MNE日志级别
mne.set_log_level('WARNING')


@timer
def create_mne_raw(eeg_data, ch_names=None, sfreq=None):
    """将numpy数组转换为MNE Raw对象

    Args:
        eeg_data: 形状为[channels, samples]的EEG数据
        ch_names: 通道名称列表，默认为None（将自动生成）
        sfreq: 采样率，默认为None（使用config中的设置）

    Returns:
        mne.io.Raw对象
    """
    if sfreq is None:
        sfreq = config.SAMPLING_RATE

    # 如果未提供通道名称，则自动生成
    if ch_names is None:
        ch_names = [f'EEG{i + 1:03d}' for i in range(eeg_data.shape[0])]

    # 创建信息对象
    info = mne.create_info(ch_names, sfreq, ch_types='eeg')

    # 创建Raw对象
    raw = mne.io.RawArray(eeg_data, info)

    # 设置montage
    try:
        montage = mne.channels.make_standard_montage(config.MONTAGE_NAME)
        raw.set_montage(montage)
    except Exception as e:
        logger.warning(f"设置电极布局失败: {e}")

    return raw


@timer
def filter_raw(raw, l_freq=None, h_freq=None, notch_freq=None):
    """对Raw对象应用滤波器

    Args:
        raw: mne.io.Raw对象
        l_freq: 高通滤波器频率，默认为None（使用config中的设置）
        h_freq: 低通滤波器频率，默认为None（使用config中的设置）
        notch_freq: 陷波滤波器频率，默认为None（使用config中的设置）

    Returns:
        滤波后的mne.io.Raw对象
    """
    if l_freq is None:
        l_freq = config.FILTER_LOW

    if h_freq is None:
        h_freq = config.FILTER_HIGH

    # 应用带通滤波器
    raw_filtered = raw.copy().filter(l_freq=l_freq, h_freq=h_freq,
                                     method='fir', phase='zero-double',
                                     fir_window='hamming', fir_design='firwin')

    # 应用陷波滤波器（如果需要）
    if notch_freq is not None:
        if isinstance(notch_freq, (int, float)):
            notch_freq = [notch_freq]
        raw_filtered = raw_filtered.notch_filter(freqs=notch_freq, method='fir',
                                                 phase='zero-double', fir_design='firwin')

    return raw_filtered


@timer
def downsample_raw(raw, sfreq=None):
    """对Raw对象进行降采样

    Args:
        raw: mne.io.Raw对象
        sfreq: 目标采样率，默认为None（使用config中的设置）

    Returns:
        降采样后的mne.io.Raw对象
    """
    if sfreq is None:
        sfreq = config.DOWN_SAMPLING_RATE

    # 应用降采样
    raw_downsampled = raw.copy().resample(sfreq)

    return raw_downsampled


@timer
def remove_artifacts_ica(raw, n_components=None):
    """使用ICA去除伪迹

    Args:
        raw: mne.io.Raw对象
        n_components: ICA成分数量，默认为None（使用config中的设置）

    Returns:
        去除伪迹后的mne.io.Raw对象
    """
    if n_components is None:
        n_components = config.ICA_COMPONENTS

    # 创建ICA对象
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=config.RANDOM_SEED)

    # 应用ICA
    ica.fit(raw)

    # 查找并排除眼动伪迹
    try:
        eog_indices, eog_scores = ica.find_bads_eog(raw)
        if eog_indices:
            ica.exclude = eog_indices
            logger.info(f"已识别并排除{len(eog_indices)}个眼动相关的ICA成分")
    except Exception as e:
        logger.warning(f"自动眼动检测失败: {e}")

    # 应用ICA去除伪迹
    raw_clean = raw.copy()
    ica.apply(raw_clean)

    return raw_clean


@timer
def segment_data(raw, tmin=None, tmax=None):
    """将连续数据分割成不重叠的小段

    Args:
        raw: mne.io.Raw对象
        tmin: 每段起始时间（秒），默认为None
        tmax: 每段终止时间（秒），默认为None

    Returns:
        列表，包含多个mne.Epochs对象
    """
    if tmin is None and tmax is None:
        # 如果未指定时间范围，则每段为2秒
        segment_duration = 2.0  # 秒

        # 计算一共可以分成多少段
        data_duration = raw.times[-1]
        n_segments = int(data_duration / segment_duration)

        # 创建事件数组
        events = np.zeros((n_segments, 3), dtype=int)
        for i in range(n_segments):
            # 事件时间点
            events[i, 0] = int(i * segment_duration * raw.info['sfreq'])
            # 事件ID为1
            events[i, 2] = 1

        # 创建Epochs对象
        epochs = mne.Epochs(raw, events, tmin=0, tmax=segment_duration,
                            baseline=None, preload=True)
    else:
        # 如果指定了时间范围，则按指定范围分段
        raise NotImplementedError("自定义时间范围的分段功能尚未实现")

    return epochs


@checkpoint("preprocess")
def preprocess_eeg_data(eeg_data, subject_id=None, sfreq=None):
    """对单个受试者的EEG数据进行预处理

    Args:
        eeg_data: 形状为[channels, samples]的EEG数据
        subject_id: 受试者ID，默认为None
        sfreq: 原始采样率，默认为None（使用config中的设置）

    Returns:
        预处理后的EEG数据（mne.io.Raw对象）
    """
    if sfreq is None:
        sfreq = config.SAMPLING_RATE

    logger.info(f"开始预处理受试者{subject_id}的EEG数据...")

    # 转换为MNE Raw对象
    raw = create_mne_raw(eeg_data, sfreq=sfreq)

    # 应用滤波器
    raw_filtered = filter_raw(raw,
                              l_freq=config.FILTER_LOW,
                              h_freq=config.FILTER_HIGH,
                              notch_freq=config.NOTCH_FREQ)

    # 降采样
    raw_downsampled = downsample_raw(raw_filtered, sfreq=config.DOWN_SAMPLING_RATE)

    # 去除伪迹
    raw_clean = remove_artifacts_ica(raw_downsampled, n_components=config.ICA_COMPONENTS)

    logger.info(f"受试者{subject_id}的EEG数据预处理完成")

    return raw_clean


@checkpoint("preprocess_group")
def preprocess_group_data(group_data, group_name):
    """对一个组的所有EEG数据进行预处理

    Args:
        group_data: 包含该组所有受试者EEG数据的字典 {subject_id: eeg_data}
        group_name: 组名称

    Returns:
        包含预处理后数据的字典 {subject_id: processed_data}
    """
    logger.info(f"开始预处理{group_name}组的EEG数据...")

    processed_data = {}

    for subject_id, eeg_data in group_data.items():
        # 预处理该受试者的数据
        processed = preprocess_eeg_data(eeg_data, subject_id=subject_id, group=group_name)
        processed_data[subject_id] = processed

    logger.info(f"{group_name}组的EEG数据预处理完成，共{len(processed_data)}个受试者")

    return processed_data


@checkpoint("preprocess_all")
def preprocess_all_data(all_data):
    """对所有EEG数据进行预处理

    Args:
        all_data: 由data_loader.load_all_data返回的数据字典

    Returns:
        包含预处理后数据的字典 {'A': processed_addiction_data, 'C': processed_control_data}
    """
    logger.info("开始预处理所有EEG数据...")

    # 预处理游戏成瘾组数据
    processed_addiction_data = preprocess_group_data(all_data['A'], 'A')

    # 预处理对照组数据
    processed_control_data = preprocess_group_data(all_data['C'], 'C')

    all_processed_data = {
        'A': processed_addiction_data,
        'C': processed_control_data
    }

    logger.info("所有EEG数据预处理完成")

    return all_processed_data


@timer
def extract_epochs_features(epochs, include_bands=True):
    """从Epochs对象中提取特征

    Args:
        epochs: mne.Epochs对象
        include_bands: 是否包括频段功率特征，默认为True

    Returns:
        特征字典
    """
    features = {}

    # 提取时域特征
    data = epochs.get_data()  # shape: (epochs, channels, times)

    # 计算每个通道的平均值
    features['mean'] = np.mean(data, axis=2)  # shape: (epochs, channels)

    # 计算每个通道的标准差
    features['std'] = np.std(data, axis=2)  # shape: (epochs, channels)

    # 计算每个通道的偏度
    from scipy import stats
    features['skew'] = stats.skew(data, axis=2)  # shape: (epochs, channels)

    # 计算每个通道的峰度
    features['kurtosis'] = stats.kurtosis(data, axis=2)  # shape: (epochs, channels)

    # 计算每个通道的最大绝对值
    features['max_abs'] = np.max(np.abs(data), axis=2)  # shape: (epochs, channels)

    # 如果需要，提取频域特征
    if include_bands:
        # 计算功率谱密度
        psds, freqs = mne.time_frequency.psd_welch(epochs, fmin=1, fmax=40, n_fft=256)

        # 对每个频段计算平均功率
        for band_name, (fmin, fmax) in config.BANDS.items():
            # 找出该频段对应的频率索引
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)

            # 计算该频段的平均功率
            band_power = np.mean(psds[:, :, idx], axis=2)  # shape: (epochs, channels)

            features[f'power_{band_name}'] = band_power

    return features