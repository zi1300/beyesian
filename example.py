"""
示例脚本，展示如何使用各个模块进行分析
"""
import os
import matplotlib.pyplot as plt
import numpy as np

# 导入自定义模块
import config
from utils import logger, set_seed
import data_loader
import preprocessing
import feature_extraction
import tvar_model
import bayesian_network
import visualization

def example_single_subject():
    """单个受试者分析示例"""
    # 设置随机种子
    set_seed(config.RANDOM_SEED)
    
    logger.info("开始单个受试者分析示例...")
    
    # 1. 加载数据
    logger.info("加载数据...")
    all_data = data_loader.load_all_data(force=False)
    
    # 选择第一个游戏成瘾组的受试者
    addiction_subject_id = list(all_data['A'].keys())[0]
    addiction_subject_data = all_data['A'][addiction_subject_id]
    
    logger.info(f"选择受试者: {addiction_subject_id}")
    logger.info(f"数据形状: {addiction_subject_data.shape}")
    
    # 2. 预处理
    logger.info("预处理...")
    raw = preprocessing.preprocess_eeg_data(addiction_subject_data, subject_id=addiction_subject_id)
    
    # 3. 特征提取
    logger.info("特征提取...")
    features = feature_extraction.extract_features(raw, subject_id=addiction_subject_id)
    
    # 4. 显示一些特征
    logger.info("部分提取的特征:")
    for feature_name in ['mean', 'std', 'power_alpha', 'power_beta', 'mean_connectivity']:
        if feature_name in features:
            value = features[feature_name]
            if np.isscalar(value):
                logger.info(f"  {feature_name}: {value:.4f}")
            else:
                logger.info(f"  {feature_name}: 形状={value.shape}, 均值={np.mean(value):.4f}")
    
    # 5. 准备TVAR数据
    logger.info("准备TVAR数据...")
    tvar_data = feature_extraction.prepare_tvar_data(raw, subject_id=addiction_subject_id)
    
    # 6. 选择一个频段拟合TVAR模型
    band_name = 'alpha'
    logger.info(f"为{band_name}频段拟合TVAR模型...")
    
    # 获取第一个窗口的数据
    window_data = tvar_data[band_name]['windows'][0]
    
    # 拟合TVAR模型
    connectivity = tvar_model.fit_tvar_model(window_data, subject_id=addiction_subject_id, band=band_name)
    
    # 7. 构建贝叶斯网络
    logger.info(f"为{band_name}频段构建贝叶斯网络...")
    model = bayesian_network.build_bayesian_network(connectivity, subject_id=addiction_subject_id, band=band_name)
    
    # 8. 可视化
    logger.info("生成可视化...")
    
    # 绘制原始EEG数据
    fig = visualization.plot_raw_eeg(raw, subject_id=addiction_subject_id, duration=5)
    plt.savefig("example_raw_eeg.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 绘制频谱
    fig = visualization.plot_psd(raw, subject_id=addiction_subject_id)
    plt.savefig("example_psd.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 绘制地形图
    fig = visualization.plot_topomap(raw, subject_id=addiction_subject_id)
    plt.savefig("example_topomap.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 绘制TVAR-Bayesian网络
    fig = visualization.plot_tvar_bayesian_network(
        connectivity, model, band_name, 
        title=f"受试者 {addiction_subject_id} - {band_name}频段 TVAR-Bayesian网络分析"
    )
    plt.savefig("example_tvar_bayesian.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logger.info("单个受试者分析示例完成!")
    logger.info("图表已保存到当前目录")

def example_group_comparison():
    """组间比较示例"""
    # 设置随机种子
    set_seed(config.RANDOM_SEED)
    
    logger.info("开始组间比较示例...")
    
    # 1. 加载数据
    logger.info("加载数据...")
    all_data = data_loader.load_all_data(force=False)
    
    # 2. 预处理
    logger.info("预处理所有数据...")
    all_processed_data = preprocessing.preprocess_all_data(all_data)
    
    # 3. 特征提取
    logger.info("提取所有特征...")
    all_features = feature_extraction.extract_all_features(all_processed_data)
    
    # 4. 准备TVAR数据
    logger.info("准备所有TVAR数据...")
    all_tvar_data = feature_extraction.prepare_all_tvar_data(all_processed_data)
    
    # 5. 拟合TVAR模型
    logger.info("拟合所有TVAR模型...")
    all_connectivity = tvar_model.fit_all_tvar_models(all_tvar_data)
    
    # 6. 构建贝叶斯网络
    logger.info("构建所有贝叶斯网络...")
    all_networks = bayesian_network.build_all_networks(all_connectivity)
    
    # 7. 分析组间差异
    logger.info("分析组间差异...")
    differences = bayesian_network.analyze_group_differences(all_connectivity)
    
    # 8. 可视化
    logger.info("生成组间比较可视化...")
    
    # 选择一个频段
    band_name = 'alpha'
    
    # 创建组间网络比较图
    band_diff = differences[band_name]
    addiction_network = nx.DiGraph(band_diff['addiction_structure'].edges())
    control_network = nx.DiGraph(band_diff['control_structure'].edges())
    
    fig = visualization.plot_brain_network_comparison(
        addiction_network, control_network, 
        title=f"{band_name}频段脑网络组间比较"
    )
    plt.savefig("example_network_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 创建频段功率比较图
    import pandas as pd
    
    # 将特征转换为DataFrame
    features_df = pd.DataFrame()
    
    for group_name, group_features in all_features.items():
        for subject_id, features in group_features.items():
            # 提取频段功率特征
            band_powers = {}
            for b_name in config.BANDS.keys():
                power_key = f'power_{b_name}'
                if power_key in features:
                    # 计算平均功率
                    band_powers[power_key] = np.mean(features[power_key])
            
            # 添加到DataFrame
            band_powers['subject_id'] = subject_id
            band_powers['group'] = group_name
            
            features_df = pd.concat([features_df, pd.DataFrame([band_powers])], ignore_index=True)
    
    # 绘制频段功率比较图
    band_columns = [f'power_{band}' for band in config.BANDS.keys()]
    fig = visualization.plot_band_power_comparison(
        features_df, band_columns, 
        title="频段功率组间比较"
    )
    plt.savefig("example_band_power_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logger.info("组间比较示例完成!")
    logger.info("图表已保存到当前目录")

if __name__ == "__main__":
    import networkx as nx
    
    # 如果是从Python脚本运行，则执行示例
    os.makedirs("example_results", exist_ok=True)
    os.chdir("example_results")
    
    try:
        # 运行单个受试者分析示例
        example_single_subject()
        
        # 运行组间比较示例
        example_group_comparison()
    
    except Exception as e:
        logger.error(f"示例运行出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
