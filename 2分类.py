import os
import shutil


def copy_eeg_mat_files(source_dir, target_dir):
    """
    从源目录中查找包含eeg的mat文件并复制到目标目录

    参数:
        source_dir: 源目录路径
        target_dir: 目标目录路径
    """
    # 确保目标目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 遍历A和C文件夹
    for folder_name in ['A', 'C']:
        source_folder = os.path.join(source_dir, folder_name)
        target_folder = os.path.join(target_dir, folder_name)

        # 确保目标A和C文件夹存在
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        # 检查源文件夹是否存在
        if not os.path.exists(source_folder):
            print(f"警告: 源文件夹 {source_folder} 不存在")
            continue

        # 遍历被试名字文件夹
        for subject in os.listdir(source_folder):
            subject_path = os.path.join(source_folder, subject)

            # 确保这是一个文件夹
            if not os.path.isdir(subject_path):
                continue

            # 遍历被试文件夹下的所有子文件夹和文件
            for root, dirs, files in os.walk(subject_path):
                for file in files:
                    # 检查是否为包含eeg的mat文件
                    if file.endswith('.mat') and 'eeg' in file.lower():
                        source_file = os.path.join(root, file)
                        target_file = os.path.join(target_folder, file)

                        # 复制文件
                        shutil.copy2(source_file, target_file)
                        print(f"已复制: {source_file} -> {target_file}")


if __name__ == "__main__":
    # 获取用户输入
    source_directory =r"D:\yuze\研究生\实验1处理数据\2分类"
    target_directory = r"D:\yuze\研究生\实验1处理数据\2分类-1"

    # 执行复制操作
    copy_eeg_mat_files(source_directory, target_directory)
    print("复制完成！")