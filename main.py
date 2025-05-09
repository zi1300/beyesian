"""
主程序，整合所有模块，实现EEG数据的预处理、特征提取、TVAR模型构建和贝叶斯网络分析
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# 导入自定义模块
import config
from utils import logger, set_seed, timer
import data_loader
import preprocessing
import feature_extraction
import tvar_model
import bayesian_network
import visualization


def parse_arguments():
    """解析命令行参数

    Returns:
        解析后的参数
    """
    parser = argparse.ArgumentParser(description='EEG数据分析与TVAR-Bayesian网络构建')

    parser.add_argument('--data_dir', type=str, default=config.DATA_ROOT_DIR,
                        help='数据根目录，包含A和C两个子目录')

    parser.add_argument('--addiction_dir', type=str, default=None,
                        help='游戏成瘾组目录，如果不指定则使用data_dir/A')

    parser.add_argument('--control_dir', type=str, default=None,
                        help='对照组目录，如果不指定则使用data_dir/C')

    parser.add_argument('--output_dir', type=str, default=config.RESULTS_DIR,
                        help='结果保存目录')

    parser.add_argument('--checkpoint_dir', type=str, default=config.CHECKPOINT_DIR,
                        help='检查点保存目录')

    parser.add_argument('--force', action='store_true',
                        help='强制重新计算所有步骤，忽略检查点')

    parser.add_argument('--skip_preprocess', action='store_true',
                        help='跳过预处理步骤')

    parser.add_argument('--skip_feature', action='store_true',
                        help='跳过特征提取步骤')

    parser.add_argument('--skip_tvar', action='store_true',
                        help='跳过TVAR模型步骤')

    parser.add_argument('--skip_bayesian', action='store_true',
                        help='跳过贝叶斯网络步骤')

    parser.add_argument('--skip_visualization', action='store_true',
                        help='跳过可视化步骤')

    parser.add_argument('--seed', type=int, default=config.RANDOM_SEED,
                        help='随机种子')

    parser.add_argument('--verbose', action='store_true',
                        help='显示详细日志')

    return parser.parse_args()


@timer
def main():
    """主函数，按顺序执行各个分析步骤"""
    # 解析命令行参数
    args = parse_arguments()

    # 设置随机种子
    set_seed(args.seed)

    # 更新配置
    if args.data_dir != config.DATA_ROOT_DIR:
        config.DATA_ROOT_DIR = args.data_dir
        config.ADDICTION_DIR = os.path.join(args.data_dir, "A")
        config.CONTROL_DIR = os.path.join(args.data_dir, "C")

    if args.addiction_dir:
        config.ADDICTION_DIR = args.addiction_dir

    if args.control_dir:
        config.CONTROL_DIR = args.control_dir

    if args.output_dir != config.RESULTS_DIR:
        config.RESULTS_DIR = args.output_dir
        config.FIG_DIR = os.path.join(args.output_dir, "figures")

    if args.checkpoint_dir != config.CHECKPOINT_DIR:
        config.CHECKPOINT_DIR = args.checkpoint_dir

    # 创建必要的目录
    for directory in [config.CHECKPOINT_DIR, config.RESULTS_DIR, config.FIG_DIR]:
        os.makedirs(directory, exist_ok=True)

    logger.info("开始EEG数据分析...")

    try:
        # 加载数据
        logger.info("第1步：加载数据...")
        all_data = data_loader.load_all_data(force=args.force)

        # 输出数据统计信息
        subject_info = data_loader.get_subject_info(all_data)
        logger.info(f"共加载{subject_info['total_subjects']}个受试者的数据")
        logger.info(f"游戏成瘾组: {subject_info['addiction_count']}个受试者")
        logger.info(f"对照组: {subject_info['control_count']}个受试者")

        # 预处理
        if not args.skip_preprocess:
            logger.info("第2步：预处理...")
            all_processed_data = preprocessing.preprocess_all_data(all_data, force=args.force)
        else:
            logger.info("跳过预处理步骤，尝试从检查点加载...")
            all_processed_data = preprocessing.preprocess_all_data(all_data, force=False)

        # 特征提取
        if not args.skip_feature:
            logger.info("第3步：特征提取...")
            # 提取普通特征
            all_features = feature_extraction.extract_all_features(all_processed_data, force=args.force)

            # 准备TVAR数据
            all_tvar_data = feature_extraction.prepare_all_tvar_data(all_processed_data, force=args.force)
        else:
            logger.info("跳过特征提取步骤，尝试从检查点加载...")
            all_features = feature_extraction.extract_all_features(all_processed_data, force=False)
            all_tvar_data = feature_extraction.prepare_all_tvar_data(all_processed_data, force=False)

        # TVAR模型
        if not args.skip_tvar:
            logger.info("第4步：TVAR模型构建...")
            all_connectivity = tvar_model.fit_all_tvar_models(all_tvar_data, force=args.force)
        else:
            logger.info("跳过TVAR模型步骤，尝试从检查点加载...")
            all_connectivity = tvar_model.fit_all_tvar_models(all_tvar_data, force=False)

        # 贝叶斯网络
        if not args.skip_bayesian:
            logger.info("第5步：贝叶斯网络构建...")
            all_networks = bayesian_network.build_all_networks(all_connectivity, force=args.force)

            # 分析组间差异
            differences = bayesian_network.analyze_group_differences(all_connectivity, force=args.force)
        else:
            logger.info("跳过贝叶斯网络步骤，尝试从检查点加载...")
            all_networks = bayesian_network.build_all_networks(all_connectivity, force=False)
            differences = bayesian_network.analyze_group_differences(all_connectivity, force=False)

        # 可视化
        if not args.skip_visualization:
            logger.info("第6步：可视化...")
            dashboard_paths = visualization.create_dashboard(
                all_data, all_processed_data, all_features,
                all_connectivity, all_networks, differences
            )

            logger.info(f"生成了{len(dashboard_paths)}个可视化图表")
            logger.info(f"可视化结果保存在: {config.FIG_DIR}")

        logger.info("所有分析步骤完成!")
        logger.info(f"结果保存在: {config.RESULTS_DIR}")

        return 0

    except Exception as e:
        logger.error(f"程序执行过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())