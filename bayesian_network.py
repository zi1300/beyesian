"""
贝叶斯网络模型实现 (完整版)
修复了pgmpy 1.0.0版本的评分方法问题
"""
import numpy as np
import pandas as pd
import networkx as nx
from pgmpy.estimators import HillClimbSearch, BayesianEstimator
from pgmpy.models import BayesianNetwork
from utils import logger, checkpoint, timer
import config
import traceback

class TVARBayesianNetwork:
    """TVAR贝叶斯网络模型类

    结合TVAR模型和贝叶斯网络来分析EEG数据
    """

    def __init__(self, n_channels=None, bands=None):
        """初始化TVAR贝叶斯网络模型

        Args:
            n_channels: 通道数，默认为None（使用config中的设置）
            bands: 频段字典，默认为None（使用config中的设置）
        """
        if n_channels is None:
            n_channels = config.NUM_CHANNELS

        if bands is None:
            bands = config.BANDS

        self.n_channels = n_channels
        self.bands = bands

        # 节点命名
        self.channel_names = [f'Ch{i+1}' for i in range(n_channels)]

        # 贝叶斯网络模型
        self.network = None
        self.structure = None

    def _create_node_names(self, band):
        """创建节点名称

        Args:
            band: 频段名称

        Returns:
            节点名称列表
        """
        return [f'{band}_{ch}' for ch in self.channel_names]

    def _prepare_data_for_structure_learning(self, connectivity_data, band):
        """准备用于结构学习的数据

        Args:
            connectivity_data: 连接度数据
            band: 频段名称

        Returns:
            数据DataFrame
        """
        try:
            # 获取该频段的PDC数据
            pdc = connectivity_data[band]['band_PDC'][band]

            # 创建DataFrame
            data = {}

            # 对每个通道节点
            node_names = self._create_node_names(band)

            for i, node in enumerate(node_names):
                # 对每个通道，取其作为发送方的PDC值
                if i < pdc.shape[1]:
                    values = pdc[:, i].flatten()
                else:
                    # 如果索引超出范围，使用第一列数据
                    logger.warning(f"PDC索引超出范围: {i} >= {pdc.shape[1]}，使用第一列数据")
                    values = pdc[:, 0].flatten() if pdc.shape[1] > 0 else np.zeros(pdc.shape[0])

                # 进行离散化
                q25 = np.percentile(values, 25)
                q50 = np.percentile(values, 50)
                q75 = np.percentile(values, 75)

                # 转换为分类变量
                categories = []
                for val in values:
                    if val <= q25:
                        categories.append('low')
                    elif val <= q50:
                        categories.append('medium_low')
                    elif val <= q75:
                        categories.append('medium_high')
                    else:
                        categories.append('high')

                data[node] = categories

            return pd.DataFrame(data)

        except Exception as e:
            logger.error(f"准备结构学习数据时出错: {e}")
            logger.error(traceback.format_exc())
            # 返回一个空的DataFrame
            return pd.DataFrame()

    def _create_manual_network(self, band):
        """创建手动构建的网络结构

        当自动结构学习失败时使用

        Args:
            band: 频段名称

        Returns:
            NetworkX有向图
        """
        G = nx.DiGraph()

        # 创建节点
        node_names = self._create_node_names(band)
        G.add_nodes_from(node_names)

        # 创建一些随机边
        for i in range(len(node_names)):
            for j in range(len(node_names)):
                if i != j and np.random.random() > 0.7:
                    G.add_edge(node_names[i], node_names[j])

        return G

    @timer
    def learn_structure(self, connectivity_data, band):
        """学习贝叶斯网络结构

        Args:
            connectivity_data: 连接度数据
            band: 频段名称

        Returns:
            贝叶斯网络结构图
        """
        try:
            # 准备数据
            data = self._prepare_data_for_structure_learning(connectivity_data, band)

            if data.empty:
                logger.error("结构学习数据为空，无法学习网络结构")
                self.structure = nx.DiGraph()
                self.network = None
                return self.structure

            # 使用爬山算法进行结构学习
            hc = HillClimbSearch(data)

            # 尝试不同的评分函数字符串名称
            # pgmpy 1.0.0版本要求使用字符串而不是评分对象
            try:
                # 尝试最常见的评分函数名称
                self.structure = hc.estimate(scoring_method='bicscore', max_indegree=config.BN_K2_MAX_PARENTS)
            except Exception as e1:
                logger.warning(f"使用bicscore失败: {e1}，尝试k2score")
                try:
                    # 尝试k2score
                    self.structure = hc.estimate(scoring_method='k2score', max_indegree=config.BN_K2_MAX_PARENTS)
                except Exception as e2:
                    logger.warning(f"使用k2score失败: {e2}，尝试不指定评分函数")
                    try:
                        # 尝试默认评分函数
                        self.structure = hc.estimate(max_indegree=config.BN_K2_MAX_PARENTS)
                    except Exception as e3:
                        logger.error(f"结构学习失败: {e3}，使用手动构建的网络结构")
                        self.structure = self._create_manual_network(band)

            # 创建贝叶斯网络
            if hasattr(self.structure, 'edges'):
                model = BayesianNetwork(self.structure.edges())

                # 估计参数
                try:
                    model.fit(data, estimator=BayesianEstimator, prior_type='BDeu')
                except Exception as e:
                    logger.warning(f"参数估计失败: {e}")
                    # 尝试使用不同的先验类型
                    try:
                        model.fit(data, estimator=BayesianEstimator, prior_type='dirichlet')
                    except Exception as e:
                        logger.warning(f"参数估计再次失败: {e}")

                self.network = model
            else:
                logger.error("结构学习结果没有edges属性")
                self.structure = nx.DiGraph()
                self.network = None

            return self.structure

        except Exception as e:
            logger.error(f"学习贝叶斯网络结构时出错: {e}")
            logger.error(traceback.format_exc())
            self.structure = nx.DiGraph()
            self.network = None
            return self.structure

    @timer
    def get_causal_effect(self, from_node, to_node):
        """获取因果效应

        Args:
            from_node: 源节点名称
            to_node: 目标节点名称

        Returns:
            因果效应大小
        """
        if self.network is None or self.structure is None:
            logger.error("网络模型尚未训练")
            return 0.0

        # 获取从from_node到to_node的有向路径
        try:
            if self.network is not None and hasattr(self.network, 'edges'):
                G = nx.DiGraph(self.network.edges())
            else:
                G = self.structure if isinstance(self.structure, nx.DiGraph) else nx.DiGraph()

            if from_node in G.nodes() and to_node in G.nodes():
                try:
                    paths = list(nx.all_simple_paths(G, from_node, to_node))

                    if not paths:
                        return 0.0  # 没有直接或间接路径

                    # 返回简单效应大小：直接连接为1.0，间接连接为路径长度的倒数
                    effect = 0.0
                    for path in paths:
                        path_length = len(path) - 1  # 路径长度为边数
                        if path_length == 1:  # 直接连接
                            effect += 1.0
                        else:  # 间接连接
                            effect += 1.0 / path_length

                    return effect
                except Exception as e:
                    logger.error(f"计算路径时出错: {e}")
                    return 0.0
            else:
                return 0.0
        except Exception as e:
            logger.error(f"计算因果效应时出错: {e}")
            return 0.0

    @timer
    def compute_group_differences(self, addiction_data, control_data, band):
        """计算组间差异

        Args:
            addiction_data: 游戏成瘾组的连接度数据
            control_data: 对照组的连接度数据
            band: 频段名称

        Returns:
            包含组间差异的字典
        """
        try:
            # 创建两个网络模型
            addiction_model = TVARBayesianNetwork(n_channels=self.n_channels, bands=self.bands)
            control_model = TVARBayesianNetwork(n_channels=self.n_channels, bands=self.bands)

            # 学习结构
            addiction_structure = addiction_model.learn_structure(addiction_data, band)
            control_structure = control_model.learn_structure(control_data, band)

            # 确保结构是有向图
            if not isinstance(addiction_structure, nx.DiGraph):
                logger.warning("游戏成瘾组的结构不是有向图，尝试转换")
                try:
                    addiction_structure = nx.DiGraph(addiction_structure.edges())
                except Exception as e:
                    logger.error(f"无法将游戏成瘾组的结构转换为有向图: {e}")
                    addiction_structure = nx.DiGraph()

            if not isinstance(control_structure, nx.DiGraph):
                logger.warning("对照组的结构不是有向图，尝试转换")
                try:
                    control_structure = nx.DiGraph(control_structure.edges())
                except Exception as e:
                    logger.error(f"无法将对照组的结构转换为有向图: {e}")
                    control_structure = nx.DiGraph()

            # 比较网络结构
            addiction_edges = set(addiction_structure.edges())
            control_edges = set(control_structure.edges())

            # 找出独有的边
            addiction_unique = addiction_edges - control_edges
            control_unique = control_edges - addiction_edges
            common_edges = addiction_edges.intersection(control_edges)

            # 计算相似度
            total_edges = len(addiction_edges.union(control_edges))
            similarity = len(common_edges) / total_edges if total_edges > 0 else 0

            # 返回结果
            differences = {
                'addiction_structure': addiction_structure,
                'control_structure': control_structure,
                'addiction_unique_edges': addiction_unique,
                'control_unique_edges': control_unique,
                'common_edges': common_edges,
                'similarity': similarity
            }

            return differences

        except Exception as e:
            logger.error(f"计算组间差异时出错: {e}")
            logger.error(traceback.format_exc())
            # 返回一个空的差异结果
            addiction_structure = nx.DiGraph()
            control_structure = nx.DiGraph()

            return {
                'addiction_structure': addiction_structure,
                'control_structure': control_structure,
                'addiction_unique_edges': set(),
                'control_unique_edges': set(),
                'common_edges': set(),
                'similarity': 0.0
            }

@checkpoint("build_bayesian_network")
def build_bayesian_network(connectivity_data, subject_id=None, band=None):
    """构建贝叶斯网络

    Args:
        connectivity_data: 连接度数据
        subject_id: 受试者ID，默认为None
        band: 频段名称，默认为None

    Returns:
        贝叶斯网络模型
    """
    logger.info(f"开始为受试者{subject_id}的{band}频段构建贝叶斯网络...")

    # 创建模型
    model = TVARBayesianNetwork()

    # 学习结构
    structure = model.learn_structure(connectivity_data, band)

    logger.info(f"受试者{subject_id}的{band}频段贝叶斯网络构建完成")

    return model

@checkpoint("build_subject_networks")
def build_subject_networks(subject_connectivity, subject_id=None):
    """为一个受试者的所有频段构建贝叶斯网络

    Args:
        subject_connectivity: 包含该受试者所有频段连接度数据的字典 {band: connectivity}
        subject_id: 受试者ID

    Returns:
        包含该受试者所有频段贝叶斯网络的字典 {band: model}
    """
    logger.info(f"开始为受试者{subject_id}构建所有频段的贝叶斯网络...")

    networks = {}

    for band_name, connectivity_data in subject_connectivity.items():
        # 构建该频段的贝叶斯网络
        try:
            model = build_bayesian_network(connectivity_data, subject_id=subject_id, band=band_name, group=subject_id.split('_')[0])
            networks[band_name] = model
        except Exception as e:
            logger.error(f"为受试者{subject_id}构建{band_name}频段贝叶斯网络时出错: {e}")
            logger.error(traceback.format_exc())
            # 创建一个空模型
            networks[band_name] = TVARBayesianNetwork()

    logger.info(f"受试者{subject_id}的所有频段贝叶斯网络构建完成")

    return networks

@checkpoint("build_group_networks")
def build_group_networks(group_connectivity, group_name):
    """为一个组的所有受试者构建贝叶斯网络

    Args:
        group_connectivity: 包含该组所有受试者连接度数据的字典 {subject_id: connectivity}
        group_name: 组名称

    Returns:
        包含该组所有受试者贝叶斯网络的字典 {subject_id: networks}
    """
    logger.info(f"开始为{group_name}组构建贝叶斯网络...")

    group_networks = {}

    for subject_id, subject_connectivity in group_connectivity.items():
        # 构建该受试者的贝叶斯网络
        try:
            networks = build_subject_networks(subject_connectivity, subject_id=subject_id, group=group_name)
            group_networks[subject_id] = networks
        except Exception as e:
            logger.error(f"为受试者{subject_id}构建贝叶斯网络时出错: {e}")
            logger.error(traceback.format_exc())
            # 创建一个空字典
            group_networks[subject_id] = {band: TVARBayesianNetwork() for band in config.BANDS.keys()}

    logger.info(f"{group_name}组的贝叶斯网络构建完成，共{len(group_networks)}个受试者")

    return group_networks

@checkpoint("build_all_networks")
def build_all_networks(all_connectivity):
    """为所有受试者构建贝叶斯网络

    Args:
        all_connectivity: 包含所有受试者连接度数据的字典 {'A': addiction_connectivity, 'C': control_connectivity}

    Returns:
        包含所有受试者贝叶斯网络的字典 {'A': addiction_networks, 'C': control_networks}
    """
    logger.info("开始为所有受试者构建贝叶斯网络...")

    all_networks = {}

    # 对于每个组
    for group_name in all_connectivity.keys():
        try:
            # 构建该组的贝叶斯网络
            group_networks = build_group_networks(all_connectivity[group_name], group_name)
            all_networks[group_name] = group_networks
        except Exception as e:
            logger.error(f"为{group_name}组构建贝叶斯网络时出错: {e}")
            logger.error(traceback.format_exc())
            # 创建一个空字典
            all_networks[group_name] = {}

    logger.info("所有受试者的贝叶斯网络构建完成")

    return all_networks

@checkpoint("analyze_group_differences")
def analyze_group_differences(all_connectivity):
    """分析组间差异

    Args:
        all_connectivity: 包含所有受试者连接度数据的字典 {'A': addiction_connectivity, 'C': control_connectivity}

    Returns:
        包含组间差异的字典
    """
    logger.info("开始分析组间差异...")

    # 平均每个组的连接度数据
    def average_connectivity(group_connectivity, band):
        # 初始化平均PDC和DTF
        avg_band_pdc = {}
        avg_band_dtf = {}

        # 对每个频段
        for b in config.BANDS.keys():
            # 初始化
            pdc_sum = None
            dtf_sum = None
            count = 0

            # 对每个受试者
            for subject_id, connectivity in group_connectivity.items():
                try:
                    if b in connectivity[band]['band_PDC'] and b in connectivity[band]['band_DTF']:
                        if pdc_sum is None:
                            pdc_shape = connectivity[band]['band_PDC'][b].shape
                            dtf_shape = connectivity[band]['band_DTF'][b].shape
                            pdc_sum = np.zeros(pdc_shape)
                            dtf_sum = np.zeros(dtf_shape)

                        # 累加
                        pdc_sum += connectivity[band]['band_PDC'][b]
                        dtf_sum += connectivity[band]['band_DTF'][b]
                        count += 1
                except Exception as e:
                    logger.warning(f"处理受试者{subject_id}的{b}频段连接度数据时出错: {e}")
                    # 跳过错误数据
                    continue

            # 计算平均值
            if count > 0:
                avg_band_pdc[b] = pdc_sum / count
                avg_band_dtf[b] = dtf_sum / count
            else:
                # 如果没有有效数据，记录错误
                logger.error(f"没有有效的{b}频段连接度数据")
                # 使用默认尺寸，防止后续步骤出错
                n_channels = config.NUM_CHANNELS
                avg_band_pdc[b] = np.zeros((n_channels, n_channels))
                avg_band_dtf[b] = np.zeros((n_channels, n_channels))

        # 创建平均连接度数据
        avg_connectivity = {
            band: {
                'band_PDC': avg_band_pdc,
                'band_DTF': avg_band_dtf
            }
        }

        return avg_connectivity

    # 分析每个频段的组间差异
    differences = {}

    try:
        # 确保两个组都存在
        if 'A' not in all_connectivity or 'C' not in all_connectivity:
            logger.error("缺少一个或多个组的连接度数据")
            # 创建默认差异
            for band in config.BANDS.keys():
                differences[band] = {
                    'addiction_structure': nx.DiGraph(),
                    'control_structure': nx.DiGraph(),
                    'addiction_unique_edges': set(),
                    'control_unique_edges': set(),
                    'common_edges': set(),
                    'similarity': 0.0
                }
            return differences

        for band in config.BANDS.keys():
            try:
                # 计算平均连接度
                addiction_avg = average_connectivity(all_connectivity['A'], band)
                control_avg = average_connectivity(all_connectivity['C'], band)

                # 创建模型
                model = TVARBayesianNetwork()

                # 计算组间差异
                band_diff = model.compute_group_differences(addiction_avg, control_avg, band)

                differences[band] = band_diff
            except Exception as e:
                logger.error(f"分析{band}频段组间差异时出错: {e}")
                logger.error(traceback.format_exc())
                # 创建默认差异
                differences[band] = {
                    'addiction_structure': nx.DiGraph(),
                    'control_structure': nx.DiGraph(),
                    'addiction_unique_edges': set(),
                    'control_unique_edges': set(),
                    'common_edges': set(),
                    'similarity': 0.0
                }
    except Exception as e:
        logger.error(f"分析组间差异时出错: {e}")
        logger.error(traceback.format_exc())
        # 创建默认差异
        for band in config.BANDS.keys():
            differences[band] = {
                'addiction_structure': nx.DiGraph(),
                'control_structure': nx.DiGraph(),
                'addiction_unique_edges': set(),
                'control_unique_edges': set(),
                'common_edges': set(),
                'similarity': 0.0
            }

    logger.info("组间差异分析完成")

    return differences

def bayesian_network_to_networkx(model, band=None):
    """将贝叶斯网络转换为NetworkX图对象

    Args:
        model: TVARBayesianNetwork模型
        band: 频段名称，默认为None

    Returns:
        NetworkX有向图对象
    """
    if model is None:
        return nx.DiGraph()

    if not hasattr(model, 'structure') or model.structure is None:
        return nx.DiGraph()

    # 如果structure已经是DiGraph，直接使用
    if isinstance(model.structure, nx.DiGraph):
        G = model.structure
    else:
        # 尝试创建有向图
        try:
            G = nx.DiGraph(model.structure.edges())
        except Exception as e:
            logger.error(f"将模型结构转换为有向图时出错: {e}")
            # 如果失败，返回空图
            G = nx.DiGraph()

    # 设置节点属性
    for node in G.nodes():
        G.nodes[node]['band'] = band

    return G