#!/usr/bin/env python3
"""
read_xlsx.py

简单接口：读取指定的 `.xlsx` 文件（默认首个工作表，首行为标题），
并返回一个 numpy 数组（dtype=float）。

假设所有数据均为数字；遇到非数字或缺失值会抛出 ValueError。
"""

from pathlib import Path
import numpy as np
import pandas as pd

__all__ = ["read_xlsx_to_numpy"]


def read_xlsx_to_numpy(file_path, sheet=0):
    """读取单个 `.xlsx` 文件并返回 numpy.ndarray。

    参数:
      - file_path: 文件路径（字符串或 Path）
      - sheet: 工作表名或索引（默认 0，即第一个表）

    返回:
      - numpy.ndarray，形状 (n_rows, n_cols)，dtype=float

    抛出:
      - FileNotFoundError: 文件不存在
      - ValueError: 存在无法转换为数字的值或缺失值
    """
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    # 读取表格，第一行为列名
    df = pd.read_excel(p, sheet_name=sheet, engine="openpyxl", header=0)

    # 尝试将所有列转换为数值类型
    df_num = df.apply(pd.to_numeric, errors="coerce")
    if df_num.isnull().values.any():
        # 找到首个出现问题的单元格以便提示
        coords = np.argwhere(pd.isnull(df_num.values))
        r, c = coords[0]
        col = df_num.columns[c]
        raise ValueError(f"文件 `{p.name}` 在第 {r+2} 行、列 '{col}' 含有非数字或缺失值。")

    arr = df_num.to_numpy(dtype=float)
    return arr


if __name__ == "__main__":
    # 简单自检（不会修改文件），仅在直接运行时提示用法
    print("此模块只提供函数 `read_xlsx_to_numpy(file_path, sheet=0)`。")
    print(read_xlsx_to_numpy("14/kp0_3 type0 amp50 freq0.5.xlsx"))
