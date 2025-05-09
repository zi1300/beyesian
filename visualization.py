"""
可视化模块，包含各种EEG数据和模型结果的可视化函数
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import networkx as nx
import mne
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure
from utils import logger, timer
import config

# 设置字体和样式
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

# 设置中文字体（如果需要显示中文）
# 解决中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 可换成你安装的其他中文字体
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决坐标轴负号乱码问题

# 解决中文显示问题
import matplotlib as mpl


def configure_chinese_font():
    """配置中文字体支持"""
    # 检查系统中可用的字体
    from matplotlib.font_manager import fontManager

    # 尝试几种常见的中文字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'NSimSun', 'STSong', 'DengXian']

    # 检查是否有中文字体可用
    available_fonts = [f for f in chinese_fonts if any(font.name == f for font in fontManager.ttflist)]

    if available_fonts:
        # 使用第一个可用的中文字体
        mpl.rcParams['font.family'] = available_fonts[0]
        logger.info(f"已设置中文字体: {available_fonts[0]}")
    else:
        # 如果没有可用的中文字体，尝试其他方法
        try:
            # 尝试使用系统默认字体
            mpl.rcParams['font.family'] = 'sans-serif'
            # 在Windows系统上，添加微软雅黑
            if os.name == 'nt':  # Windows系统
                mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] + mpl.rcParams['font.sans-serif']
            else:  # 类Unix系统
                mpl.rcParams['font.sans-serif'] = ['DejaVu Sans'] + mpl.rcParams['font.sans-serif']

            # 修复负号显示问题
            mpl.rcParams['axes.unicode_minus'] = False
            logger.info("已尝试设置系统默认字体以支持中文")
        except:
            logger.warning("无法配置中文字体，可能会导致中文显示为方框")


# 调用函数配置字体
configure_chinese_font()


@timer
def plot_raw_eeg(raw, subject_id=None, duration=10, n_channels=None, title=None, save_path=None):
    """绘制原始EEG数据

    Args:
        raw: mne.io.Raw对象
        subject_id: 受试者ID，默认为None
        duration: 要显示的时间长度（秒），默认为10
        n_channels: 要显示的通道数，默认为None（显示所有通道）
        title: 图表标题，默认为None
        save_path: 保存路径，默认为None（不保存）

    Returns:
        matplotlib.figure.Figure对象
    """
    if n_channels is None:
        n_channels = len(raw.ch_names)

    fig = raw.plot(duration=duration, n_channels=n_channels,
                   title=title or f"原始EEG - 受试者 {subject_id}",
                   show=False, scalings='auto')

    if save_path:
        fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        logger.info(f"图表已保存到: {save_path}")

    return fig


@timer
def plot_psd(raw, subject_id=None, fmin=1, fmax=40, title=None, save_path=None):
    """绘制功率谱密度

    Args:
        raw: mne.io.Raw对象
        subject_id: 受试者ID，默认为None
        fmin: 最小频率，默认为1
        fmax: 最大频率，默认为40
        title: 图表标题，默认为None
        save_path: 保存路径，默认为None（不保存）

    Returns:
        matplotlib.figure.Figure对象
    """
    try:
        # 使用scipy直接计算PSD，避免MNE API兼容性问题
        from scipy import signal

        # 获取原始数据和采样率
        data = raw.get_data()
        sfreq = raw.info['sfreq']

        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))

        # 使用Welch方法计算每个通道的PSD
        for i, ch_name in enumerate(raw.ch_names):
            freqs, psd = signal.welch(data[i], fs=sfreq, nperseg=min(512, data.shape[1]))

            # 只保留指定频率范围内的数据
            mask = (freqs >= fmin) & (freqs <= fmax)
            ax.semilogy(freqs[mask], psd[mask], label=ch_name if i < 5 else None)  # 只显示前5个通道的标签

        # 标注频段
        bands = config.BANDS
        colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00']

        for i, (band_name, (band_fmin, band_fmax)) in enumerate(bands.items()):
            if band_fmin >= fmin and band_fmax <= fmax:
                # 频段在可见范围内
                ax.axvspan(band_fmin, band_fmax, alpha=0.2, color=colors[i % len(colors)])
                # 添加频段标签
                band_center = (band_fmin + band_fmax) / 2
                ax.text(band_center, ax.get_ylim()[1] * 0.9, band_name,
                        horizontalalignment='center', verticalalignment='center',
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

        # 设置图表属性
        ax.set_xlabel('频率 (Hz)')
        ax.set_ylabel('功率谱密度 (µV²/Hz)')
        ax.set_title(title or f"功率谱密度 - 受试者 {subject_id}")
        ax.grid(True)

        # 显示均值PSD
        mean_psd = np.mean([signal.welch(data[i], fs=sfreq, nperseg=min(512, data.shape[1]))[1]
                            for i in range(len(raw.ch_names))], axis=0)
        mean_freqs = signal.welch(data[0], fs=sfreq, nperseg=min(512, data.shape[1]))[0]
        mask = (mean_freqs >= fmin) & (mean_freqs <= fmax)
        ax.semilogy(mean_freqs[mask], mean_psd[mask], 'k-', linewidth=2, label='均值')

        # 添加图例（只包含前5个通道和均值）
        if len(raw.ch_names) > 5:
            ax.legend(loc='upper right', title=f"显示5/{len(raw.ch_names)}个通道")
        else:
            ax.legend(loc='upper right')

        plt.tight_layout()

        if save_path:
            try:
                fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
                logger.info(f"图表已保存到: {save_path}")
            except Exception as e:
                logger.error(f"保存图表失败: {e}")

        return fig

    except Exception as e:
        logger.error(f"绘制PSD时出错: {e}")

        # 创建一个简单的备用图表
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"无法绘制PSD\n错误: {str(e)}",
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        ax.set_title(title or f"功率谱密度 - 受试者 {subject_id}")

        if save_path:
            try:
                fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
            except:
                pass

        return fig


@timer
def plot_topomap(raw, subject_id=None, bands=None, title=None, save_path=None):
    """绘制地形图

    Args:
        raw: mne.io.Raw对象
        subject_id: 受试者ID，默认为None
        bands: 频段字典，默认为None（使用config中的设置）
        title: 图表标题，默认为None
        save_path: 保存路径，默认为None（不保存）

    Returns:
        matplotlib.figure.Figure对象
    """
    if bands is None:
        bands = config.BANDS

    # 使用scipy直接计算PSD，避免MNE API不兼容问题
    try:
        from scipy import signal

        # 获取原始数据和采样率
        data = raw.get_data()
        sfreq = raw.info['sfreq']

        # 使用Welch方法计算PSD
        freqs, psds_temp = signal.welch(data, fs=sfreq, nperseg=512)

        # 保留1-40Hz范围内的频率
        freq_mask = (freqs >= 1) & (freqs <= 40)
        freqs = freqs[freq_mask]
        psds = psds_temp[:, freq_mask]

        # 扩展维度以匹配之前API的预期形状 (channels, epochs, freqs)
        psds = psds[:, np.newaxis, :]

        logger.info(f"使用scipy.signal.welch计算PSD成功，形状为{psds.shape}")
    except Exception as e:
        logger.error(f"使用scipy计算PSD失败: {e}")
        # 提供备用方案 - 创建假数据继续绘图流程
        n_channels = len(raw.ch_names)
        freqs = np.linspace(1, 40, 40)
        psds = np.ones((n_channels, 1, len(freqs)))
        logger.warning("使用默认PSD数据")

    # 创建图表
    n_bands = len(bands)
    fig, axes = plt.subplots(1, n_bands, figsize=(4 * n_bands, 4))
    if n_bands == 1:
        axes = [axes]

    # 对每个频段绘制地形图
    for i, (band_name, (fmin, fmax)) in enumerate(bands.items()):
        try:
            # 找出该频段对应的频率索引
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)

            # 确保有频率点在这个范围内
            if not np.any(idx):
                logger.warning(f"频段{band_name} ({fmin}-{fmax} Hz)没有对应的频率点")
                # 使用最接近的频率点
                idx = np.abs(freqs - ((fmin + fmax) / 2)).argmin()
                idx = np.array([idx])

            # 计算该频段的平均功率
            band_power = np.mean(psds[:, :, idx], axis=(1, 2)).squeeze()

            # 为避免功率为负数或零（取对数时出问题），进行调整
            if np.any(band_power <= 0):
                min_positive = np.min(band_power[band_power > 0]) if np.any(band_power > 0) else 1e-10
                band_power[band_power <= 0] = min_positive / 10

            # 绘制地形图
            try:
                # 直接使用原始通道数据绘制颜色图，避免MNE topomap API问题
                ch_names = np.array(raw.ch_names)

                # 创建简单的彩色条形图作为替代
                bar_positions = np.arange(len(band_power))
                bars = axes[i].bar(bar_positions, band_power, color='skyblue')

                # 高亮显示最大值
                max_idx = np.argmax(band_power)
                bars[max_idx].set_color('red')

                # 设置标签
                axes[i].set_xticks(bar_positions)
                if len(ch_names) <= 10:
                    axes[i].set_xticklabels(ch_names, rotation=45, ha='right')
                else:
                    # 如果通道太多，只显示部分
                    step = max(1, len(ch_names) // 10)
                    axes[i].set_xticks(bar_positions[::step])
                    axes[i].set_xticklabels(ch_names[::step], rotation=45, ha='right')

                axes[i].set_ylabel("功率")
                axes[i].grid(True, linestyle='--', alpha=0.7)

                # 如果MNE的topomap可用，尝试使用它
                try:
                    if hasattr(mne.viz, 'plot_topomap'):
                        # 添加小的topomap作为inset
                        ax_inset = axes[i].inset_axes([0.6, 0.6, 0.35, 0.35])
                        mne.viz.plot_topomap(band_power, raw.info, axes=ax_inset, show=False)
                except Exception as e_topo:
                    logger.warning(f"无法绘制topomap: {e_topo}")

            except Exception as e_plot:
                logger.error(f"绘制地形图失败: {e_plot}")
                # 使用简单的线图替代
                axes[i].plot(range(len(band_power)), band_power, 'o-')
                axes[i].set_xlabel("通道")
                axes[i].set_ylabel("功率")

            axes[i].set_title(f"{band_name} ({fmin}-{fmax} Hz)")

        except Exception as e:
            logger.error(f"处理{band_name}频段时出错: {e}")
            # 创建一个空白面板
            axes[i].text(0.5, 0.5, f"无法绘制{band_name}频段",
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform=axes[i].transAxes)
            axes[i].set_title(f"{band_name} ({fmin}-{fmax} Hz)")

    plt.suptitle(title or f"EEG频段功率分布 - 受试者 {subject_id}")
    plt.tight_layout()

    if save_path:
        try:
            fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
            logger.info(f"图表已保存到: {save_path}")
        except Exception as e:
            logger.error(f"保存图表失败: {e}")

    return fig


@timer
def plot_connectivity_matrix(matrix, channel_names=None, title=None, cmap='viridis', save_path=None):
    """绘制连接矩阵

    Args:
        matrix: 连接矩阵
        channel_names: 通道名称列表，默认为None
        title: 图表标题，默认为None
        cmap: 颜色映射，默认为'viridis'
        save_path: 保存路径，默认为None（不保存）

    Returns:
        matplotlib.figure.Figure对象
    """
    if channel_names is None:
        channel_names = [f'Ch{i + 1}' for i in range(matrix.shape[0])]

    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制热图
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=np.max(matrix))

    # 添加颜色条
    plt.colorbar(im, ax=ax)

    # 设置坐标轴刻度
    ax.set_xticks(np.arange(len(channel_names)))
    ax.set_yticks(np.arange(len(channel_names)))
    ax.set_xticklabels(channel_names, rotation=45, ha='right')
    ax.set_yticklabels(channel_names)

    # 设置标题
    ax.set_title(title or "连接矩阵")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        logger.info(f"图表已保存到: {save_path}")

    return fig


@timer
def plot_band_power_comparison(features_df, band_columns, group_col='group', title=None, save_path=None):
    """绘制频段功率比较图

    Args:
        features_df: 特征DataFrame
        band_columns: 要比较的频段列
        group_col: 分组列名，默认为'group'
        title: 图表标题，默认为None
        save_path: 保存路径，默认为None（不保存）

    Returns:
        matplotlib.figure.Figure对象
    """
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 6))

    # 设置颜色
    colors = [config.COLOR_ADDICTION, config.COLOR_CONTROL]

    # 设置分组
    groups = features_df[group_col].unique()

    # 设置x位置
    bar_width = 0.35
    x = np.arange(len(band_columns))

    # 绘制柱状图
    for i, group in enumerate(groups):
        group_data = features_df[features_df[group_col] == group]
        means = [group_data[col].mean() for col in band_columns]
        std_errs = [group_data[col].std() / np.sqrt(len(group_data)) for col in band_columns]

        ax.bar(x + i * bar_width, means, bar_width, color=colors[i],
               label=f"{'游戏成瘾组' if group == 'A' else '对照组'}")
        ax.errorbar(x + i * bar_width, means, yerr=std_errs, fmt='none', ecolor='black', capsize=5)

    # 设置坐标轴
    ax.set_xlabel('频段')
    ax.set_ylabel('功率 (dB)')
    ax.set_title(title or "频段功率比较")

    # 设置x刻度
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels([col.split('_')[-1] for col in band_columns])

    # 添加图例
    ax.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        logger.info(f"图表已保存到: {save_path}")

    return fig


@timer
def plot_brain_network(network, node_positions=None, node_size=300, node_color='skyblue',
                       edge_color='gray', title=None, save_path=None):
    """绘制脑网络图

    Args:
        network: NetworkX图对象
        node_positions: 节点位置字典，默认为None（使用spring布局）
        node_size: 节点大小，默认为300
        node_color: 节点颜色，默认为'skyblue'
        edge_color: 边颜色，默认为'gray'
        title: 图表标题，默认为None
        save_path: 保存路径，默认为None（不保存）

    Returns:
        matplotlib.figure.Figure对象
    """
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 8))

    # 如果未提供节点位置，则使用spring布局
    if node_positions is None:
        node_positions = nx.spring_layout(network)

    # 绘制网络
    nx.draw_networkx_nodes(network, node_positions, node_size=node_size, node_color=node_color, ax=ax)
    nx.draw_networkx_edges(network, node_positions, edge_color=edge_color, arrows=True,
                           arrowstyle='->', arrowsize=15, ax=ax)
    nx.draw_networkx_labels(network, node_positions, font_size=10, font_color='black', ax=ax)

    # 设置标题
    ax.set_title(title or "脑网络")

    # 去除坐标轴
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        logger.info(f"图表已保存到: {save_path}")

    return fig


@timer
def plot_brain_network_comparison(network1, network2, node_positions=None, title=None, save_path=None):
    """绘制脑网络比较图

    Args:
        network1: 第一个NetworkX图对象
        network2: 第二个NetworkX图对象
        node_positions: 节点位置字典，默认为None（使用spring布局）
        title: 图表标题，默认为None
        save_path: 保存路径，默认为None（不保存）

    Returns:
        matplotlib.figure.Figure对象
    """
    # 创建图表
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 合并两个网络以获取所有节点
    all_nodes = set(network1.nodes()).union(set(network2.nodes()))
    combined_network = nx.DiGraph()
    combined_network.add_nodes_from(all_nodes)

    # 如果未提供节点位置，则使用spring布局
    if node_positions is None:
        node_positions = nx.spring_layout(combined_network)

    # 绘制第一个网络
    nx.draw_networkx_nodes(network1, node_positions, node_size=300, node_color=config.COLOR_ADDICTION,
                           ax=axes[0])
    nx.draw_networkx_edges(network1, node_positions, edge_color=config.COLOR_ADDICTION, arrows=True,
                           arrowstyle='->', arrowsize=15, ax=axes[0])
    nx.draw_networkx_labels(network1, node_positions, font_size=8, font_color='black', ax=axes[0])
    axes[0].set_title("游戏成瘾组")
    axes[0].axis('off')

    # 绘制第二个网络
    nx.draw_networkx_nodes(network2, node_positions, node_size=300, node_color=config.COLOR_CONTROL,
                           ax=axes[1])
    nx.draw_networkx_edges(network2, node_positions, edge_color=config.COLOR_CONTROL, arrows=True,
                           arrowstyle='->', arrowsize=15, ax=axes[1])
    nx.draw_networkx_labels(network2, node_positions, font_size=8, font_color='black', ax=axes[1])
    axes[1].set_title("对照组")
    axes[1].axis('off')

    # 绘制差异网络
    common_edges = set(network1.edges()).intersection(set(network2.edges()))
    unique_edges1 = set(network1.edges()) - common_edges
    unique_edges2 = set(network2.edges()) - common_edges

    # 绘制节点
    nx.draw_networkx_nodes(combined_network, node_positions, node_size=300, node_color='lightgray',
                           ax=axes[2])

    # 绘制共同边
    nx.draw_networkx_edges(combined_network, node_positions, edgelist=common_edges,
                           edge_color='gray', arrows=True, arrowstyle='->', arrowsize=15, ax=axes[2])

    # 绘制网络1独有的边
    nx.draw_networkx_edges(combined_network, node_positions, edgelist=unique_edges1,
                           edge_color=config.COLOR_ADDICTION, arrows=True, arrowstyle='->',
                           arrowsize=15, ax=axes[2])

    # 绘制网络2独有的边
    nx.draw_networkx_edges(combined_network, node_positions, edgelist=unique_edges2,
                           edge_color=config.COLOR_CONTROL, arrows=True, arrowstyle='->',
                           arrowsize=15, ax=axes[2])

    # 绘制标签
    nx.draw_networkx_labels(combined_network, node_positions, font_size=8, font_color='black', ax=axes[2])

    axes[2].set_title("网络差异")
    axes[2].axis('off')

    # 设置标题
    plt.suptitle(title or "脑网络比较")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        logger.info(f"图表已保存到: {save_path}")

    return fig


@timer
def plot_tvar_coefficients(coeffs, channel_names=None, time_points=None, title=None, save_path=None):
    """绘制TVAR系数时变图

    Args:
        coeffs: TVAR系数历史，形状为[time, to_channel, from_channel, order]
        channel_names: 通道名称列表，默认为None
        time_points: 时间点列表，默认为None
        title: 图表标题，默认为None
        save_path: 保存路径，默认为None（不保存）

    Returns:
        matplotlib.figure.Figure对象
    """
    if channel_names is None:
        channel_names = [f'Ch{i + 1}' for i in range(coeffs[0].shape[0])]

    if time_points is None:
        time_points = np.arange(len(coeffs))

    # 获取维度
    n_times = len(coeffs)
    n_channels = coeffs[0].shape[0]
    order = coeffs[0].shape[2]

    # 创建图表
    fig, axes = plt.subplots(n_channels, n_channels, figsize=(4 * n_channels, 4 * n_channels))

    # 重塑系数为[time, to, from, order]
    reshaped_coeffs = np.zeros((n_times, n_channels, n_channels, order))
    for t in range(n_times):
        reshaped_coeffs[t] = coeffs[t]

    # 对每对通道绘制系数变化
    for i in range(n_channels):  # 接收通道
        for j in range(n_channels):  # 发送通道
            ax = axes[i, j]

            # 对每个阶数绘制系数变化
            for p in range(order):
                ax.plot(time_points, reshaped_coeffs[:, i, j, p],
                        label=f"AR({p + 1})")

            # 设置标题和标签
            if i == 0:
                ax.set_title(f"From {channel_names[j]}")
            if j == 0:
                ax.set_ylabel(f"To {channel_names[i]}")

            # 添加零线
            ax.axhline(y=0, color='gray', linestyle='--')

    # 添加图例（只在一个子图上添加）
    axes[0, 0].legend()

    # 设置总标题
    plt.suptitle(title or "TVAR系数时变图")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        logger.info(f"图表已保存到: {save_path}")

    return fig


@timer
def plot_tvar_connectivity(connectivity, band, times=None, title=None, save_path=None):
    """绘制TVAR连接度时变图

    Args:
        connectivity: 连接度指标字典
        band: 频段名称
        times: 时间点列表，默认为None
        title: 图表标题，默认为None
        save_path: 保存路径，默认为None（不保存）

    Returns:
        matplotlib.figure.Figure对象
    """
    # 获取PDC历史
    pdc_history = connectivity['PDC_history']

    if times is None:
        times = np.arange(len(pdc_history))

    # 获取通道数和频率点数
    n_channels = pdc_history[0].shape[0]
    n_freqs = pdc_history[0].shape[2]

    # 获取频段范围
    fmin, fmax = config.BANDS[band]

    # 计算该频段对应的频率索引
    freqs = np.linspace(0, 0.5, n_freqs)  # 归一化频率
    fmin_norm = fmin / (config.DOWN_SAMPLING_RATE / 2)
    fmax_norm = fmax / (config.DOWN_SAMPLING_RATE / 2)
    idx = np.logical_and(freqs >= fmin_norm, freqs <= fmax_norm)

    # 计算该频段的平均PDC
    band_pdc = np.zeros((len(pdc_history), n_channels, n_channels))
    for t in range(len(pdc_history)):
        band_pdc[t] = np.mean(pdc_history[t][:, :, idx], axis=2)

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 8))

    # 创建颜色映射
    colors = plt.cm.tab10.colors

    # 对每对通道绘制PDC变化
    channel_names = [f'Ch{i + 1}' for i in range(n_channels)]
    for i in range(n_channels):  # 发送通道
        for j in range(n_channels):  # 接收通道
            if i != j:  # 跳过自连接
                ax.plot(times, band_pdc[:, j, i],
                        label=f"{channel_names[i]} → {channel_names[j]}",
                        color=colors[(i * n_channels + j) % len(colors)])

    # 设置标题和标签
    ax.set_title(title or f"{band}频段PDC时变图")
    ax.set_xlabel("时间点")
    ax.set_ylabel("PDC值")

    # 添加图例
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        logger.info(f"图表已保存到: {save_path}")

    return fig


@timer
def create_brain_network_report(all_networks, differences, save_dir=None):
    """创建脑网络分析报告

    Args:
        all_networks: 包含所有受试者贝叶斯网络的字典 {'A': addiction_networks, 'C': control_networks}
        differences: 包含组间差异的字典
        save_dir: 保存目录，默认为None（使用config中的设置）

    Returns:
        报告保存路径列表
    """
    if save_dir is None:
        save_dir = config.FIG_DIR

    os.makedirs(save_dir, exist_ok=True)

    report_paths = []

    # 对每个频段创建报告
    for band_name in config.BANDS.keys():
        # 提取该频段的差异
        band_diff = differences[band_name]

        # 创建NetworkX图对象
        addiction_network = nx.DiGraph(band_diff['addiction_structure'].edges())
        control_network = nx.DiGraph(band_diff['control_structure'].edges())

        # 绘制网络比较图
        title = f"{band_name}频段脑网络比较"
        save_path = os.path.join(save_dir, f"brain_network_comparison_{band_name}.{config.PLOT_FORMAT}")
        plot_brain_network_comparison(addiction_network, control_network,
                                      title=title, save_path=save_path)

        report_paths.append(save_path)

        # 创建结果表格
        similarity = band_diff['similarity'] * 100
        addiction_unique = len(band_diff['addiction_unique_edges'])
        control_unique = len(band_diff['control_unique_edges'])
        common = len(band_diff['common_edges'])

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis('tight')
        ax.axis('off')

        table_data = [
            ["相似度", f"{similarity:.1f}%"],
            ["游戏成瘾组独有边数", addiction_unique],
            ["对照组独有边数", control_unique],
            ["共同边数", common]
        ]

        table = ax.table(cellText=table_data, colLabels=["指标", "值"],
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)

        plt.title(f"{band_name}频段脑网络比较结果")

        save_path = os.path.join(save_dir, f"brain_network_stats_{band_name}.{config.PLOT_FORMAT}")
        plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        plt.close()

        report_paths.append(save_path)

    return report_paths


@timer
def plot_eeg_preprocessing_steps(raw, filtered_raw, clean_raw, subject_id=None,
                                 duration=5, channels=None, save_path=None):
    """绘制EEG预处理步骤对比图

    Args:
        raw: 原始mne.io.Raw对象
        filtered_raw: 滤波后的mne.io.Raw对象
        clean_raw: 去伪迹后的mne.io.Raw对象
        subject_id: 受试者ID，默认为None
        duration: 要显示的时间长度（秒），默认为5
        channels: 要显示的通道列表，默认为None（显示前5个通道）
        save_path: 保存路径，默认为None（不保存）

    Returns:
        matplotlib.figure.Figure对象
    """
    if channels is None:
        channels = raw.ch_names[:5]  # 显示前5个通道

    n_channels = len(channels)

    # 创建图表
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # 设置颜色循环
    colors = plt.cm.tab10.colors

    # 获取数据
    start, stop = 0, int(duration * raw.info['sfreq'])

    # 绘制原始数据
    data, times = raw[:, start:stop]
    for i, ch in enumerate(channels):
        ch_idx = raw.ch_names.index(ch)
        axes[0].plot(times, data[ch_idx],
                     label=ch, color=colors[i % len(colors)])

    axes[0].set_title("原始EEG数据")
    axes[0].set_ylabel("振幅 (μV)")
    axes[0].legend(loc='upper right')

    # 绘制滤波后数据
    data, times = filtered_raw[:, start:stop]
    for i, ch in enumerate(channels):
        ch_idx = filtered_raw.ch_names.index(ch)
        axes[1].plot(times, data[ch_idx],
                     label=ch, color=colors[i % len(colors)])

    axes[1].set_title("滤波后数据")
    axes[1].set_ylabel("振幅 (μV)")

    # 绘制去伪迹后数据
    data, times = clean_raw[:, start:stop]
    for i, ch in enumerate(channels):
        ch_idx = clean_raw.ch_names.index(ch)
        axes[2].plot(times, data[ch_idx],
                     label=ch, color=colors[i % len(colors)])

    axes[2].set_title("去伪迹后数据")
    axes[2].set_xlabel("时间 (秒)")
    axes[2].set_ylabel("振幅 (μV)")

    plt.suptitle(f"EEG预处理步骤 - 受试者 {subject_id}")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        logger.info(f"图表已保存到: {save_path}")

    return fig


@timer
def plot_group_comparison_topomaps(all_features, band_power_keys=None, save_path=None):
    """绘制组间比较地形图

    Args:
        all_features: 包含所有特征的字典 {'A': addiction_features, 'C': control_features}
        band_power_keys: 频段功率特征键，默认为None
        save_path: 保存路径，默认为None（不保存）

    Returns:
        matplotlib.figure.Figure对象
    """
    if band_power_keys is None:
        band_power_keys = [f'power_{band}' for band in config.BANDS.keys()]

    n_bands = len(band_power_keys)

    # 计算每个组的平均功率
    addiction_power = {}
    control_power = {}

    for key in band_power_keys:
        # 初始化
        addiction_sum = None
        control_sum = None
        addiction_count = 0
        control_count = 0

        # 游戏成瘾组
        for subject_id, features in all_features['A'].items():
            if key in features:
                if addiction_sum is None:
                    addiction_sum = features[key].copy()
                else:
                    addiction_sum += features[key]
                addiction_count += 1

        # 对照组
        for subject_id, features in all_features['C'].items():
            if key in features:
                if control_sum is None:
                    control_sum = features[key].copy()
                else:
                    control_sum += features[key]
                control_count += 1

        # 计算平均值
        if addiction_count > 0:
            addiction_power[key] = addiction_sum / addiction_count

        if control_count > 0:
            control_power[key] = control_sum / control_count

    # 创建图表
    fig = plt.figure(figsize=(15, 5 * n_bands))
    gs = GridSpec(n_bands, 3, figure=fig, width_ratios=[1, 1, 0.1])

    # 对每个频段绘制地形图
    for i, key in enumerate(band_power_keys):
        band_name = key.split('_')[-1]

        # 游戏成瘾组
        ax1 = fig.add_subplot(gs[i, 0])
        im = ax1.imshow(addiction_power[key].reshape(8, 4), cmap='viridis',
                        interpolation='bicubic')
        ax1.set_title(f"游戏成瘾组 - {band_name}频段")
        ax1.set_xticks([])
        ax1.set_yticks([])

        # 对照组
        ax2 = fig.add_subplot(gs[i, 1])
        im = ax2.imshow(control_power[key].reshape(8, 4), cmap='viridis',
                        interpolation='bicubic')
        ax2.set_title(f"对照组 - {band_name}频段")
        ax2.set_xticks([])
        ax2.set_yticks([])

        # 颜色条
        cax = fig.add_subplot(gs[i, 2])
        plt.colorbar(im, cax=cax)

    plt.suptitle("EEG频段功率组间比较")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        logger.info(f"图表已保存到: {save_path}")

    return fig


@timer
def plot_tvar_bayesian_network(tvar_connectivity, bayesian_network, band,
                               channel_names=None, title=None, save_path=None):
    """绘制TVAR-Bayesian网络图

    Args:
        tvar_connectivity: TVAR连接度指标字典
        bayesian_network: 贝叶斯网络模型
        band: 频段名称
        channel_names: 通道名称列表，默认为None
        title: 图表标题，默认为None
        save_path: 保存路径，默认为None（不保存）

    Returns:
        matplotlib.figure.Figure对象
    """
    if channel_names is None:
        n_channels = tvar_connectivity['PDC'].shape[0]
        channel_names = [f'Ch{i + 1}' for i in range(n_channels)]

    # 创建图表
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1])

    # 绘制PDC矩阵
    ax1 = fig.add_subplot(gs[0, 0])
    pdc = tvar_connectivity['band_PDC'][band]
    im1 = ax1.imshow(pdc, cmap='viridis', vmin=0, vmax=1)
    ax1.set_title(f"{band}频段 PDC矩阵")
    ax1.set_xticks(np.arange(len(channel_names)))
    ax1.set_yticks(np.arange(len(channel_names)))
    ax1.set_xticklabels(channel_names, rotation=45, ha='right')
    ax1.set_yticklabels(channel_names)
    plt.colorbar(im1, ax=ax1)

    # 绘制DTF矩阵
    ax2 = fig.add_subplot(gs[0, 1])
    dtf = tvar_connectivity['band_DTF'][band]
    im2 = ax2.imshow(dtf, cmap='viridis', vmin=0, vmax=1)
    ax2.set_title(f"{band}频段 DTF矩阵")
    ax2.set_xticks(np.arange(len(channel_names)))
    ax2.set_yticks(np.arange(len(channel_names)))
    ax2.set_xticklabels(channel_names, rotation=45, ha='right')
    ax2.set_yticklabels(channel_names)
    plt.colorbar(im2, ax=ax2)

    # 绘制贝叶斯网络
    ax3 = fig.add_subplot(gs[0, 2])
    network = nx.DiGraph(bayesian_network.structure.edges())

    # 设置节点位置为圆形布局
    pos = nx.circular_layout(network)

    # 绘制网络
    nx.draw_networkx_nodes(network, pos, node_size=500, node_color='skyblue', ax=ax3)
    nx.draw_networkx_edges(network, pos, edge_color='gray', arrows=True,
                           arrowstyle='->', arrowsize=15, ax=ax3)
    nx.draw_networkx_labels(network, pos, font_size=10, font_color='black', ax=ax3)

    ax3.set_title(f"{band}频段 贝叶斯网络")
    ax3.axis('off')

    # 设置总标题
    plt.suptitle(title or f"{band}频段 TVAR-Bayesian网络分析")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        logger.info(f"图表已保存到: {save_path}")

    return fig


@timer
def create_dashboard(all_data, all_processed_data, all_features, all_connectivity, all_networks,
                     differences, save_dir=None):
    """创建分析仪表板

    Args:
        all_data: 所有原始数据
        all_processed_data: 所有预处理后的数据
        all_features: 所有特征
        all_connectivity: 所有连接度指标
        all_networks: 所有贝叶斯网络
        differences: 组间差异
        save_dir: 保存目录，默认为None（使用config中的设置）

    Returns:
        仪表板文件路径列表
    """
    if save_dir is None:
        save_dir = config.FIG_DIR

    os.makedirs(save_dir, exist_ok=True)

    dashboard_paths = []

    # 1. 创建预处理步骤图
    for group_name, group_data in all_processed_data.items():
        # 选择第一个受试者
        subject_id = list(group_data.keys())[0]
        raw = group_data[subject_id]

        # 创建地形图
        save_path = os.path.join(save_dir, f"topomap_{group_name}_{subject_id}.{config.PLOT_FORMAT}")
        plot_topomap(raw, subject_id=subject_id, save_path=save_path)
        dashboard_paths.append(save_path)

        # 创建PSD图
        save_path = os.path.join(save_dir, f"psd_{group_name}_{subject_id}.{config.PLOT_FORMAT}")
        plot_psd(raw, subject_id=subject_id, save_path=save_path)
        dashboard_paths.append(save_path)

    # 2. 创建组间频段功率比较图
    # 首先需要将特征转换为DataFrame
    features_df = pd.DataFrame()

    for group_name, group_features in all_features.items():
        for subject_id, features in group_features.items():
            # 提取频段功率特征
            band_powers = {}
            for band_name in config.BANDS.keys():
                power_key = f'power_{band_name}'
                if power_key in features:
                    # 计算平均功率
                    band_powers[power_key] = np.mean(features[power_key])

            # 添加到DataFrame
            band_powers['subject_id'] = subject_id
            band_powers['group'] = group_name

            features_df = pd.concat([features_df, pd.DataFrame([band_powers])], ignore_index=True)

    # 绘制频段功率比较图
    band_columns = [f'power_{band}' for band in config.BANDS.keys()]
    save_path = os.path.join(save_dir, f"band_power_comparison.{config.PLOT_FORMAT}")
    plot_band_power_comparison(features_df, band_columns, save_path=save_path)
    dashboard_paths.append(save_path)

    # 3. 创建TVAR-Bayesian网络图
    for band_name in config.BANDS.keys():
        # 游戏成瘾组
        addiction_subject_id = list(all_connectivity['A'].keys())[0]
        addiction_connectivity = all_connectivity['A'][addiction_subject_id][band_name]
        addiction_network = all_networks['A'][addiction_subject_id][band_name]

        save_path = os.path.join(save_dir, f"tvar_bayesian_A_{band_name}.{config.PLOT_FORMAT}")
        plot_tvar_bayesian_network(addiction_connectivity, addiction_network, band_name,
                                   title=f"游戏成瘾组 {band_name}频段 TVAR-Bayesian网络分析",
                                   save_path=save_path)
        dashboard_paths.append(save_path)

        # 对照组
        control_subject_id = list(all_connectivity['C'].keys())[0]
        control_connectivity = all_connectivity['C'][control_subject_id][band_name]
        control_network = all_networks['C'][control_subject_id][band_name]

        save_path = os.path.join(save_dir, f"tvar_bayesian_C_{band_name}.{config.PLOT_FORMAT}")
        plot_tvar_bayesian_network(control_connectivity, control_network, band_name,
                                   title=f"对照组 {band_name}频段 TVAR-Bayesian网络分析",
                                   save_path=save_path)
        dashboard_paths.append(save_path)

    # 4. 创建组间差异网络图
    diff_paths = create_brain_network_report(all_networks, differences, save_dir=save_dir)
    dashboard_paths.extend(diff_paths)

    return dashboard_paths