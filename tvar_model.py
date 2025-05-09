"""
时变自回归(TVAR)模型实现 - 科研严谨版
为科研目的修改，严格追踪错误，保证数据真实性
"""
import numpy as np
from scipy import linalg
import os
from utils import logger, checkpoint, timer
import config
import traceback
import datetime

# 创建错误日志目录
ERROR_LOG_DIR = './error_logs'
os.makedirs(ERROR_LOG_DIR, exist_ok=True)

class ComputationError(Exception):
    """计算错误的自定义异常类"""
    pass

class TVARModel:
    """时变自回归(TVAR)模型类

    使用卡尔曼滤波器实现时变自回归模型，用于估计动态连接性
    科研严谨版：优先保证数据真实性，而不是程序平稳运行
    """

    def __init__(self, order=None, n_channels=None, forgetting_factor=None):
        """初始化TVAR模型

        Args:
            order: 模型阶数，默认为None（使用config中的设置）
            n_channels: 通道数，默认为None（使用config中的设置）
            forgetting_factor: 遗忘因子，默认为None（使用config中的设置）
        """
        if order is None:
            order = config.MODEL_ORDER

        if n_channels is None:
            n_channels = config.NUM_CHANNELS

        if forgetting_factor is None:
            forgetting_factor = config.LAMBDA

        self.order = order
        self.n_channels = n_channels
        self.lambda_ = forgetting_factor

        # 状态空间维度
        self.state_dim = n_channels * n_channels * order

        # 初始化参数
        self.A = np.zeros((n_channels, n_channels, order))  # AR系数
        self.P = np.eye(self.state_dim)  # 状态协方差矩阵
        self.Q = np.eye(self.state_dim) * 1e-3  # 过程噪声协方差矩阵

        # 保存结果
        self.A_history = []  # AR系数历史
        self.PDC_history = []  # 部分有向相干性历史
        self.DTF_history = []  # 有向传递函数历史

        # 错误追踪
        self.computation_success = True  # 计算是否成功的标志
        self.error_details = []  # 记录详细错误信息

        logger.info(f"TVAR模型初始化: order={order}, n_channels={n_channels}, state_dim={self.state_dim}")

    def _log_error(self, stage, error, data_info=None):
        """记录错误详细信息

        Args:
            stage: 出错的阶段
            error: 错误对象
            data_info: 相关数据信息
        """
        error_detail = {
            'stage': stage,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.datetime.now().isoformat(),
            'data_info': data_info
        }

        self.error_details.append(error_detail)
        self.computation_success = False

        # 记录到系统日志
        logger.error(f"TVAR模型{stage}阶段计算错误: {error}")

        # 将详细错误信息保存到文件
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        error_file = os.path.join(ERROR_LOG_DIR, f"tvar_error_{stage}_{timestamp}.txt")

        with open(error_file, 'w') as f:
            f.write(f"TVAR模型计算错误\n")
            f.write(f"阶段: {stage}\n")
            f.write(f"错误类型: {type(error).__name__}\n")
            f.write(f"错误信息: {str(error)}\n")
            f.write(f"发生时间: {datetime.datetime.now().isoformat()}\n")

            f.write("\n数据信息:\n")
            if data_info:
                for key, value in data_info.items():
                    f.write(f"{key}: {value}\n")

            f.write("\n详细堆栈跟踪:\n")
            f.write(traceback.format_exc())

        logger.info(f"错误详细信息已保存到: {error_file}")

        # 对于科研用途，我们希望错误传播出去，而不是隐藏
        raise ComputationError(f"TVAR模型{stage}阶段计算错误: {error}")

    def _build_regression_matrix(self, data, t):
        """构建回归矩阵

        Args:
            data: 形状为[n_channels, samples]的数据
            t: 当前时间点

        Returns:
            回归矩阵
        """
        # 验证数据维度
        if data.shape[0] != self.n_channels:
            error_msg = f"数据通道数({data.shape[0]})与模型通道数({self.n_channels})不匹配"
            self._log_error("构建回归矩阵", ValueError(error_msg),
                            {'data_shape': data.shape, 'expected_channels': self.n_channels})

        # 初始化回归矩阵
        X = np.zeros((self.n_channels, self.state_dim))

        try:
            # 填充回归矩阵
            for i in range(self.n_channels):
                for j in range(self.order):
                    if t - j - 1 >= 0:
                        # 计算起始和结束索引
                        start_idx = j * self.n_channels * self.n_channels + i * self.n_channels
                        end_idx = start_idx + self.n_channels

                        # 确保索引合法
                        if end_idx <= self.state_dim:
                            # 将数据填入回归矩阵
                            X[i, start_idx:end_idx] = data[:, t-j-1]
                        else:
                            error_msg = f"索引超出范围: end_idx={end_idx} > state_dim={self.state_dim}"
                            self._log_error("构建回归矩阵-索引错误", ValueError(error_msg),
                                          {'i': i, 'j': j, 'start_idx': start_idx, 'end_idx': end_idx})

            return X

        except Exception as e:
            self._log_error("构建回归矩阵", e, {'data_shape': data.shape, 't': t})

    def _kalman_prediction(self, a, P):
        """卡尔曼预测步骤

        Args:
            a: 状态向量
            P: 状态协方差矩阵

        Returns:
            预测后的状态向量和协方差矩阵
        """
        try:
            # 状态转移
            a_pred = a
            P_pred = P + self.Q

            return a_pred, P_pred

        except Exception as e:
            self._log_error("卡尔曼预测", e, {'a_shape': a.shape, 'P_shape': P.shape})

    def _kalman_update(self, a_pred, P_pred, y, X):
        """卡尔曼更新步骤

        Args:
            a_pred: 预测后的状态向量
            P_pred: 预测后的协方差矩阵
            y: 观测值
            X: 回归矩阵

        Returns:
            更新后的状态向量和协方差矩阵
        """
        try:
            # 计算中间量，添加小的对角量以确保数值稳定性
            temp = X @ P_pred @ X.T + np.eye(self.n_channels) * 1e-10

            # 计算逆矩阵
            temp_inv = np.linalg.inv(temp)

            # 计算卡尔曼增益
            K = P_pred @ X.T @ temp_inv

            # 更新状态
            a = a_pred + K @ (y - X @ a_pred)
            P = (np.eye(self.state_dim) - K @ X) @ P_pred

            # 验证更新结果的有效性
            if np.any(np.isnan(a)) or np.any(np.isnan(P)):
                error_msg = "卡尔曼更新产生NaN值"
                self._log_error("卡尔曼更新-NaN值", ValueError(error_msg),
                              {'a_contains_nan': np.any(np.isnan(a)),
                               'P_contains_nan': np.any(np.isnan(P))})

            return a, P

        except np.linalg.LinAlgError as e:
            # 特别处理线性代数错误
            self._log_error("卡尔曼更新-矩阵求逆", e,
                          {'temp_shape': temp.shape, 'temp_cond': np.linalg.cond(temp)})

        except Exception as e:
            self._log_error("卡尔曼更新", e,
                          {'a_pred_shape': a_pred.shape, 'P_pred_shape': P_pred.shape,
                           'y_shape': y.shape, 'X_shape': X.shape})

    def _reshape_ar_coefs(self, a):
        """将状态向量重塑为AR系数矩阵

        Args:
            a: 状态向量

        Returns:
            AR系数矩阵
        """
        try:
            A = np.zeros((self.n_channels, self.n_channels, self.order))

            for i in range(self.n_channels):
                for j in range(self.order):
                    # 计算起始和结束索引
                    start_idx = j * self.n_channels * self.n_channels + i * self.n_channels
                    end_idx = start_idx + self.n_channels

                    # 确保索引合法
                    if end_idx <= len(a):
                        A[:, i, j] = a[start_idx:end_idx]
                    else:
                        error_msg = f"重塑AR系数时索引超出范围: end_idx={end_idx} > len(a)={len(a)}"
                        self._log_error("重塑AR系数-索引错误", ValueError(error_msg),
                                      {'i': i, 'j': j, 'start_idx': start_idx, 'end_idx': end_idx})

            return A

        except Exception as e:
            self._log_error("重塑AR系数", e, {'a_shape': a.shape if hasattr(a, 'shape') else None})

    def _compute_pdc(self, A):
        """计算部分有向相干性

        Args:
            A: AR系数矩阵

        Returns:
            PDC矩阵
        """
        try:
            n_freqs = 128
            freqs = np.linspace(0, 0.5, n_freqs)
            PDC = np.zeros((self.n_channels, self.n_channels, n_freqs), dtype=complex)

            for f_idx, freq in enumerate(freqs):
                # 计算A(f)
                A_f = np.eye(self.n_channels, dtype=complex)
                for p in range(self.order):
                    A_f -= A[:, :, p] * np.exp(-1j * 2 * np.pi * freq * (p+1))

                # 计算PDC
                for i in range(self.n_channels):
                    denominator = np.sqrt(np.sum(np.abs(A_f[:, i])**2))
                    if denominator > 0:
                        PDC[:, i, f_idx] = np.abs(A_f[:, i]) / denominator
                    else:
                        error_msg = f"PDC计算中的分母为零: freq={freq}, i={i}"
                        self._log_error("PDC计算-零分母", ValueError(error_msg),
                                      {'freq': freq, 'i': i, 'A_f_col_norm': np.sum(np.abs(A_f[:, i])**2)})

            # 返回实部，丢弃微小的虚部（由于数值精度可能残留）
            return np.abs(PDC)

        except Exception as e:
            self._log_error("计算PDC", e, {'A_shape': A.shape if hasattr(A, 'shape') else None})

    def _compute_dtf(self, A):
        """计算有向传递函数

        Args:
            A: AR系数矩阵

        Returns:
            DTF矩阵
        """
        try:
            n_freqs = 128
            freqs = np.linspace(0, 0.5, n_freqs)
            DTF = np.zeros((self.n_channels, self.n_channels, n_freqs), dtype=complex)

            for f_idx, freq in enumerate(freqs):
                # 计算A(f)
                A_f = np.eye(self.n_channels, dtype=complex)
                for p in range(self.order):
                    A_f -= A[:, :, p] * np.exp(-1j * 2 * np.pi * freq * (p+1))

                try:
                    # 计算H(f) = A(f)^(-1)
                    H_f = np.linalg.inv(A_f)

                    # 计算DTF
                    for i in range(self.n_channels):
                        denominator = np.sqrt(np.sum(np.abs(H_f[i, :])**2))
                        if denominator > 0:
                            DTF[i, :, f_idx] = np.abs(H_f[i, :]) / denominator
                        else:
                            error_msg = f"DTF计算中的分母为零: freq={freq}, i={i}"
                            logger.warning(error_msg)
                            # 记录警告但不终止计算

                except np.linalg.LinAlgError as e:
                    # 特别处理线性代数错误
                    error_msg = f"计算A(f)的逆矩阵失败，频率={freq}"
                    self._log_error("DTF计算-矩阵求逆", ValueError(error_msg),
                                  {'freq': freq, 'A_f_cond': np.linalg.cond(A_f)})

            # 返回实部，丢弃微小的虚部（由于数值精度可能残留）
            return np.abs(DTF)

        except Exception as e:
            self._log_error("计算DTF", e, {'A_shape': A.shape if hasattr(A, 'shape') else None})

    @timer
    def fit(self, data):
        """拟合TVAR模型

        Args:
            data: 形状为[n_channels, samples]的数据

        Returns:
            self
        """
        # 重置错误追踪状态
        self.computation_success = True
        self.error_details = []

        logger.info(f"开始拟合TVAR模型，数据形状: {data.shape}")

        # 验证数据形状
        if len(data.shape) != 2:
            error_msg = f"数据维度错误: 期望2维数组，实际为{len(data.shape)}维"
            self._log_error("拟合验证", ValueError(error_msg), {'data_shape': data.shape})
            return self

        if data.shape[0] != self.n_channels:
            error_msg = f"数据通道数({data.shape[0]})与模型通道数({self.n_channels})不匹配"
            self._log_error("拟合验证", ValueError(error_msg),
                          {'data_shape': data.shape, 'expected_channels': self.n_channels})
            return self

        n_samples = data.shape[1]
        logger.info(f"数据样本数: {n_samples}")

        # 如果样本数太少，无法拟合
        if n_samples <= self.order:
            error_msg = f"样本数({n_samples})必须大于模型阶数({self.order})"
            self._log_error("拟合验证", ValueError(error_msg),
                          {'n_samples': n_samples, 'order': self.order})
            return self

        # 将状态向量展开为一维
        a = np.zeros(self.state_dim)

        # 初始化状态协方差矩阵
        P = np.eye(self.state_dim)

        # 清空历史记录
        self.A_history = []
        self.PDC_history = []
        self.DTF_history = []

        # 对每个时间点进行迭代
        pdc_dtf_interval = max(1, n_samples // 10)  # 计算PDC和DTF的间隔
        logger.info(f"开始迭代，计算PDC/DTF间隔: {pdc_dtf_interval}")

        try:
            for t in range(self.order, n_samples):
                # 获取当前观测值
                y = data[:, t]

                # 构建回归矩阵
                X = self._build_regression_matrix(data, t)

                # 卡尔曼预测
                a_pred, P_pred = self._kalman_prediction(a, P)

                # 卡尔曼更新
                a, P = self._kalman_update(a_pred, P_pred, y, X)

                # 重塑AR系数
                A = self._reshape_ar_coefs(a)

                # 保存结果
                self.A_history.append(A)

                # 每pdc_dtf_interval个时间点计算一次PDC和DTF
                if t % pdc_dtf_interval == 0:
                    logger.debug(f"计算时间点{t}的PDC和DTF")
                    self.PDC_history.append(self._compute_pdc(A))
                    self.DTF_history.append(self._compute_dtf(A))

        except ComputationError:
            # 这些错误已经在各个步骤中被记录
            logger.error("TVAR拟合过程中出现计算错误，停止拟合")
            return self

        except Exception as e:
            # 捕获未预期的错误
            self._log_error("拟合过程", e, {'current_t': t if 't' in locals() else None})
            return self

        # 保存最终的AR系数
        if len(self.A_history) > 0:
            self.A = self.A_history[-1]
            logger.info(f"TVAR模型拟合完成，历史记录长度: {len(self.A_history)}")
        else:
            error_msg = "拟合过程未产生有效的AR系数"
            self._log_error("拟合完成", ValueError(error_msg))

        return self

    @timer
    def get_connectivity_measures(self):
        """获取连接度指标

        Args:
            None

        Returns:
            连接度指标字典或者None（如果计算失败）

        Raises:
            ComputationError: 如果之前的计算过程有错误
        """
        if not self.computation_success:
            error_msg = "由于之前的计算错误，无法获取有效的连接度指标"
            logger.error(error_msg)

            # 对于科研用途，我们应该明确报告错误而不是返回无效数据
            raise ComputationError(error_msg)

        logger.info("计算TVAR连接度指标")

        # 计算最终的PDC和DTF
        PDC = self._compute_pdc(self.A)
        DTF = self._compute_dtf(self.A)

        # 计算频段平均PDC和DTF
        band_PDC = {}
        band_DTF = {}

        n_freqs = PDC.shape[2]
        freqs = np.linspace(0, 0.5, n_freqs)

        for band_name, (fmin, fmax) in config.BANDS.items():
            # 转换为归一化频率
            fmin_norm = fmin / (config.DOWN_SAMPLING_RATE / 2)
            fmax_norm = fmax / (config.DOWN_SAMPLING_RATE / 2)

            # 找到对应频率索引
            idx = np.logical_and(freqs >= fmin_norm, freqs <= fmax_norm)

            if np.any(idx):  # 确保有有效索引
                # 计算频段平均
                band_PDC[band_name] = np.mean(PDC[:, :, idx], axis=2)
                band_DTF[band_name] = np.mean(DTF[:, :, idx], axis=2)
            else:
                error_msg = f"频段{band_name}({fmin}-{fmax}Hz)没有对应的频率索引"
                self._log_error("频段平均计算", ValueError(error_msg),
                              {'fmin_norm': fmin_norm, 'fmax_norm': fmax_norm})

        # 返回结果
        connectivity = {
            'PDC': PDC,
            'DTF': DTF,
            'band_PDC': band_PDC,
            'band_DTF': band_DTF,
            'A': self.A,
            'A_history': self.A_history,
            'PDC_history': self.PDC_history,
            'DTF_history': self.DTF_history,
            'computation_success': self.computation_success,
            'error_details': self.error_details
        }

        return connectivity

def fit_tvar_model(window_data, subject_id=None, band=None):
    """拟合TVAR模型并计算连接度指标

    Args:
        window_data: 形状为[n_channels, samples]的数据窗口
        subject_id: 受试者ID，默认为None
        band: 频段名称，默认为None

    Returns:
        连接度指标字典，如果计算失败则返回None

    Raises:
        Exception: 如果计算过程中出现错误
    """
    logger.info(f"开始为受试者{subject_id}的{band}频段拟合TVAR模型...")

    # 验证输入数据
    if window_data is None or not isinstance(window_data, np.ndarray):
        error_msg = f"无效的窗口数据: {type(window_data)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # 打印数据形状以便调试
    logger.info(f"窗口数据形状: {window_data.shape}")

    # 创建TVAR模型
    n_channels = window_data.shape[0]
    model = TVARModel(order=config.MODEL_ORDER, n_channels=n_channels, forgetting_factor=config.LAMBDA)

    # 拟合模型
    model.fit(window_data)

    # 检查计算是否成功
    if not model.computation_success:
        error_msg = f"受试者{subject_id}的{band}频段TVAR模型拟合失败"
        logger.error(error_msg)

        # 科研用途：报告错误而不是继续使用不可靠的结果
        error_file = os.path.join(ERROR_LOG_DIR, f"tvar_failure_{subject_id}_{band}.txt")
        with open(error_file, 'w') as f:
            f.write(f"TVAR模型拟合失败\n")
            f.write(f"受试者ID: {subject_id}\n")
            f.write(f"频段: {band}\n")
            f.write(f"时间: {datetime.datetime.now().isoformat()}\n\n")

            f.write("错误详情:\n")
            for i, error in enumerate(model.error_details, 1):
                f.write(f"错误 {i}:\n")
                f.write(f"  阶段: {error['stage']}\n")
                f.write(f"  类型: {error['error_type']}\n")
                f.write(f"  信息: {error['error_message']}\n\n")

        logger.info(f"失败详情已保存到: {error_file}")

        # 对于科研用途，我们返回None表示没有有效结果
        # 而不是返回默认值或者无效数据
        return None

    try:
        # 获取连接度指标
        connectivity = model.get_connectivity_measures()
        logger.info(f"受试者{subject_id}的{band}频段TVAR模型拟合完成")
        return connectivity

    except Exception as e:
        error_msg = f"计算连接度指标时出错: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())

        # 科研用途：不返回无效数据
        return None

@checkpoint("fit_subject_tvar_models")
def fit_subject_tvar_models(subject_tvar_data, subject_id=None):
    """为一个受试者的所有频段拟合TVAR模型

    Args:
        subject_tvar_data: 包含该受试者所有频段TVAR数据的字典 {band: {windows, times}}
        subject_id: 受试者ID

    Returns:
        包含该受试者所有频段连接度指标的字典 {band: connectivity}
        成功的频段会有结果，失败的频段对应的值为None
    """
    logger.info(f"开始为受试者{subject_id}拟合所有频段的TVAR模型...")

    connectivity_measures = {}
    success_count = 0
    fail_count = 0

    # 获取可用的频段
    available_bands = list(subject_tvar_data.keys())
    logger.info(f"可用频段: {available_bands}")

    for band_name in available_bands:
        try:
            # 检查该频段是否有窗口数据
            if 'windows' not in subject_tvar_data[band_name] or not subject_tvar_data[band_name]['windows']:
                logger.warning(f"受试者{subject_id}的{band_name}频段没有窗口数据")
                connectivity_measures[band_name] = None
                fail_count += 1
                continue

            # 获取该频段的第一个窗口数据
            window_data = subject_tvar_data[band_name]['windows'][0]

            # 拟合TVAR模型
            connectivity = fit_tvar_model(window_data, subject_id=subject_id, band=band_name)

            if connectivity is not None:
                connectivity_measures[band_name] = connectivity
                success_count += 1
            else:
                connectivity_measures[band_name] = None
                fail_count += 1

        except Exception as e:
            logger.error(f"为受试者{subject_id}的{band_name}频段拟合TVAR模型时出错: {e}")
            logger.error(traceback.format_exc())
            connectivity_measures[band_name] = None
            fail_count += 1

    # 报告成功和失败的频段数量
    logger.info(f"受试者{subject_id}的TVAR模型拟合完成: {success_count}个频段成功, {fail_count}个频段失败")

    # 如果所有频段都失败，报告严重警告
    if fail_count == len(available_bands):
        logger.critical(f"警告: 受试者{subject_id}的所有频段TVAR模型拟合均失败")

    return connectivity_measures

@checkpoint("fit_group_tvar_models")
def fit_group_tvar_models(group_tvar_data, group_name):
    """为一个组的所有受试者拟合TVAR模型

    Args:
        group_tvar_data: 包含该组所有受试者TVAR数据的字典 {subject_id: tvar_data}
        group_name: 组名称

    Returns:
        包含该组所有受试者连接度指标的字典 {subject_id: connectivity_measures}
        对于失败的受试者，记录为None
    """
    logger.info(f"开始为{group_name}组拟合TVAR模型...")

    group_connectivity = {}
    success_count = 0
    fail_count = 0

    # 获取该组的所有受试者
    subjects = list(group_tvar_data.keys())
    logger.info(f"{group_name}组有{len(subjects)}个受试者")

    for subject_id in subjects:
        try:
            # 拟合该受试者的TVAR模型
            connectivity_measures = fit_subject_tvar_models(group_tvar_data[subject_id], subject_id=subject_id)

            # 检查是否所有频段都失败
            all_failed = all(v is None for v in connectivity_measures.values())

            if not all_failed:
                group_connectivity[subject_id] = connectivity_measures
                success_count += 1
            else:
                group_connectivity[subject_id] = None
                fail_count += 1
                logger.error(f"受试者{subject_id}的所有频段TVAR模型拟合均失败")

        except Exception as e:
            logger.error(f"为受试者{subject_id}拟合TVAR模型时出错: {e}")
            logger.error(traceback.format_exc())
            group_connectivity[subject_id] = None
            fail_count += 1

    # 报告成功和失败的受试者数量
    logger.info(f"{group_name}组的TVAR模型拟合完成: {success_count}个受试者成功, {fail_count}个受试者失败")

    # 如果所有受试者都失败，报告严重警告
    if fail_count == len(subjects):
        logger.critical(f"警告: {group_name}组的所有受试者TVAR模型拟合均失败")

    return group_connectivity

@checkpoint("fit_all_tvar_models")
def fit_all_tvar_models(all_tvar_data, force=False):
    """为所有受试者拟合TVAR模型

    Args:
        all_tvar_data: 包含所有受试者TVAR数据的字典 {'A': addiction_tvar_data, 'C': control_tvar_data}
        force: 是否强制重新计算，默认为False

    Returns:
        包含所有受试者连接度指标的字典 {'A': addiction_connectivity, 'C': control_connectivity}
        包含成功和失败信息的摘要报告
    """
    logger.info("开始为所有受试者拟合TVAR模型...")

    all_connectivity = {}
    success_groups = []
    fail_groups = []

    # 处理每个组
    for group_name, group_data in all_tvar_data.items():
        try:
            # 拟合该组的TVAR模型
            group_connectivity = fit_group_tvar_models(group_data, group_name, force=force)

            # 检查该组是否完全失败
            all_failed = all(v is None for v in group_connectivity.values())

            if not all_failed:
                all_connectivity[group_name] = group_connectivity
                success_groups.append(group_name)
            else:
                all_connectivity[group_name] = None
                fail_groups.append(group_name)
                logger.critical(f"{group_name}组的所有受试者TVAR模型拟合均失败")

        except Exception as e:
            logger.error(f"为{group_name}组拟合TVAR模型时出错: {e}")
            logger.error(traceback.format_exc())
            all_connectivity[group_name] = None
            fail_groups.append(group_name)

    # 报告总体成功和失败情况
    logger.info(f"所有受试者的TVAR模型拟合完成: {len(success_groups)}个组成功, {len(fail_groups)}个组失败")
    if success_groups:
        logger.info(f"成功的组: {', '.join(success_groups)}")
    if fail_groups:
        logger.error(f"失败的组: {', '.join(fail_groups)}")

    # 如果所有组都失败，记录严重错误
    if len(fail_groups) == len(all_tvar_data):
        logger.critical("严重错误: 所有组的TVAR模型拟合均失败")
        # 创建总结报告
        report_file = os.path.join(ERROR_LOG_DIR, f"tvar_complete_failure_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(report_file, 'w') as f:
            f.write("TVAR模型拟合完全失败报告\n")
            f.write(f"时间: {datetime.datetime.now().isoformat()}\n")
            f.write(f"失败的组: {', '.join(fail_groups)}\n")
            f.write("\n请检查数据格式和模型参数，或者联系技术支持获取帮助。")

        logger.info(f"失败报告已保存到: {report_file}")

    # 创建成功/失败摘要
    summary = {
        'success_groups': success_groups,
        'fail_groups': fail_groups,
        'total_subjects': sum(len(group_data) for group_name, group_data in all_tvar_data.items()),
        'success_subjects': sum(1 for group_name in success_groups
                             for subject_id, subject_data in all_connectivity.get(group_name, {}).items()
                             if subject_data is not None),
        'timestamp': datetime.datetime.now().isoformat()
    }

    # 将摘要添加到结果中
    all_connectivity['summary'] = summary

    return all_connectivity