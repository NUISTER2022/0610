import pandas as pd
import os


def merge_csv_files(mat_path, por_path, output_path):
    # 检查文件是否存在
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"数学数据文件不存在: {mat_path}")
    if not os.path.exists(por_path):
        raise FileNotFoundError(f"葡萄牙语数据文件不存在: {por_path}")

    # 读取数据
    df_mat = pd.read_csv(mat_path)
    df_por = pd.read_csv(por_path)

    # 检查列名是否一致（避免合并错误）
    if not df_mat.columns.equals(df_por.columns):
        raise ValueError("数学与葡萄牙语数据列名不一致，无法合并！")

    # 合并数据（按行拼接）
    combined_df = pd.concat([df_mat, df_por], axis=0, ignore_index=True)

    # 保存结果
    combined_df.to_csv(output_path, index=False)
    print(f"合并完成！文件保存至: {output_path}")


if __name__ == "__main__":
    # 根据你的路径修改（D:\pycharm2017\data\Data）
    mat_data_path = "D:\\pycharm2017\\data\\Data\\mat_dummies.csv"  # 数学数据绝对路径
    por_data_path = "D:\\pycharm2017\\data\\Data\\por_dummies.csv"  # 葡萄牙语数据绝对路径
    output_path = "D:\\pycharm2017\\data\\Data\\combination_dummies.csv"  # 合并后输出路径

    merge_csv_files(mat_data_path, por_data_path, output_path)
