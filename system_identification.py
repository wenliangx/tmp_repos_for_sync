#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统辨识工具 - 基于阶跃响应数据
支持一阶、二阶系统的参数辨识
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import odeint
import warnings


def first_order_step_response(t, K, tau, t0=0.0):
    """
    一阶系统阶跃响应: G(s) = K / (tau*s + 1)
    
    参数:
        t: 时间数组
        K: 增益
        tau: 时间常数
        t0: 初始时间偏移
    
    返回:
        y: 输出响应
    """
    t_shifted = t - t0
    y = np.zeros_like(t)
    mask = t_shifted >= 0
    y[mask] = K * (1 - np.exp(-t_shifted[mask] / tau))
    return y


def second_order_step_response(t, K, wn, zeta, t0=0.0):
    """
    二阶系统阶跃响应: G(s) = K*wn^2 / (s^2 + 2*zeta*wn*s + wn^2)
    
    参数:
        t: 时间数组
        K: 增益
        wn: 自然频率 (rad/s)
        zeta: 阻尼比
        t0: 初始时间偏移
    
    返回:
        y: 输出响应
    """
    t_shifted = t - t0
    y = np.zeros_like(t)
    mask = t_shifted >= 0
    
    if zeta < 1:  # 欠阻尼
        wd = wn * np.sqrt(1 - zeta**2)
        y[mask] = K * (1 - np.exp(-zeta * wn * t_shifted[mask]) * 
                       (np.cos(wd * t_shifted[mask]) + 
                        (zeta / np.sqrt(1 - zeta**2)) * np.sin(wd * t_shifted[mask])))
    elif zeta == 1:  # 临界阻尼
        y[mask] = K * (1 - np.exp(-wn * t_shifted[mask]) * 
                       (1 + wn * t_shifted[mask]))
    else:  # 过阻尼
        s1 = -zeta * wn + wn * np.sqrt(zeta**2 - 1)
        s2 = -zeta * wn - wn * np.sqrt(zeta**2 - 1)
        y[mask] = K * (1 + (s1 * np.exp(s2 * t_shifted[mask]) - 
                            s2 * np.exp(s1 * t_shifted[mask])) / (s1 - s2))
    
    return y


def identify_system(time, input_signal, output_signal, order=1, normalize=True):
    """
    基于阶跃响应的系统辨识
    
    参数:
        time: 时间数组 (1D numpy array)
        input_signal: 输入信号 (1D numpy array) - 阶跃信号
        output_signal: 输出信号 (1D numpy array) - 系统响应
        order: 系统阶数 (1 或 2)
        normalize: 是否归一化输入幅值到1 (默认True)
    
    返回:
        dict: 包含辨识结果的字典
            {
                'order': 系统阶数,
                'parameters': 参数字典 (一阶: {'K', 'tau'}, 二阶: {'K', 'wn', 'zeta'}),
                'covariance': 参数协方差矩阵,
                'fitted_output': 拟合的输出信号,
                'rmse': 均方根误差,
                'r_squared': 拟合优度 R^2,
                'step_amplitude': 阶跃幅值,
                'steady_state': 稳态值,
                'transfer_function': 传递函数字符串表示
            }
    """
    # 输入验证
    time = np.asarray(time).flatten()
    input_signal = np.asarray(input_signal).flatten()
    output_signal = np.asarray(output_signal).flatten()
    
    if len(time) != len(input_signal) or len(time) != len(output_signal):
        raise ValueError("时间、输入、输出数组长度必须相同")
    
    if order not in [1, 2]:
        raise ValueError("系统阶数必须为 1 或 2")
    
    # 检测阶跃幅值（取输入信号的稳态值与初始值之差）
    step_start = np.mean(input_signal[:max(1, len(input_signal)//10)])
    step_end = np.mean(input_signal[-max(1, len(input_signal)//10):])
    step_amplitude = step_end - step_start
    
    if abs(step_amplitude) < 1e-6:
        raise ValueError("输入信号不是有效的阶跃信号（幅值变化太小）")
    
    # 归一化：将输出按输入阶跃幅值缩放
    output_normalized = output_signal.copy()
    if normalize and abs(step_amplitude) > 1e-6:
        output_normalized = output_signal / abs(step_amplitude)
        effective_step = np.sign(step_amplitude)
    else:
        effective_step = step_amplitude
    
    # 估计初始时间（阶跃开始时刻）
    input_diff = np.abs(np.diff(input_signal))
    if np.max(input_diff) > 1e-6:
        step_index = np.argmax(input_diff)
        t0_init = time[step_index]
    else:
        t0_init = time[0]
    
    # 估计稳态值
    steady_state = np.mean(output_normalized[-max(1, len(output_normalized)//10):])
    
    try:
        if order == 1:
            # 一阶系统辨识
            # 初始猜测: K = 稳态值, tau = 时间跨度的10%, t0 = 检测到的阶跃时刻
            K_init = steady_state if abs(steady_state) > 1e-6 else 1.0
            tau_init = (time[-1] - time[0]) * 0.1
            
            # 定义拟合函数
            def fit_func(t, K, tau, t0):
                return first_order_step_response(t, K, tau, t0)
            
            # 参数边界
            bounds = ([0, 1e-6, time[0] - 1], 
                     [np.inf, (time[-1] - time[0]) * 2, time[-1]])
            
            # 曲线拟合
            popt, pcov = curve_fit(
                fit_func, 
                time, 
                output_normalized,
                p0=[K_init, tau_init, t0_init],
                bounds=bounds,
                maxfev=10000
            )
            
            K_fit, tau_fit, t0_fit = popt
            fitted_output = first_order_step_response(time, K_fit, tau_fit, t0_fit)
            
            # 恢复原始尺度
            if normalize:
                K_fit_original = K_fit * abs(step_amplitude)
                fitted_output_original = fitted_output * abs(step_amplitude)
            else:
                K_fit_original = K_fit
                fitted_output_original = fitted_output
            
            parameters = {
                'K': K_fit_original,
                'tau': tau_fit,
                't0': t0_fit
            }
            
            tf_str = f"G(s) = {K_fit_original:.4f} / ({tau_fit:.4f}*s + 1)"
            
        else:  # order == 2
            # 二阶系统辨识
            # 初始猜测
            K_init = steady_state if abs(steady_state) > 1e-6 else 1.0
            wn_init = 2 * np.pi / ((time[-1] - time[0]) * 0.2)  # 假设周期约为20%时间跨度
            zeta_init = 0.7  # 常见阻尼比
            
            def fit_func(t, K, wn, zeta, t0):
                return second_order_step_response(t, K, wn, zeta, t0)
            
            # 参数边界: K>0, wn>0, 0<zeta<5, t0在合理范围
            bounds = ([0, 0.1, 0, time[0] - 1],
                     [np.inf, 1000, 5, time[-1]])
            
            popt, pcov = curve_fit(
                fit_func,
                time,
                output_normalized,
                p0=[K_init, wn_init, zeta_init, t0_init],
                bounds=bounds,
                maxfev=20000
            )
            
            K_fit, wn_fit, zeta_fit, t0_fit = popt
            fitted_output = second_order_step_response(time, K_fit, wn_fit, zeta_fit, t0_fit)
            
            # 恢复原始尺度
            if normalize:
                K_fit_original = K_fit * abs(step_amplitude)
                fitted_output_original = fitted_output * abs(step_amplitude)
            else:
                K_fit_original = K_fit
                fitted_output_original = fitted_output
            
            parameters = {
                'K': K_fit_original,
                'wn': wn_fit,
                'zeta': zeta_fit,
                't0': t0_fit
            }
            
            tf_str = f"G(s) = {K_fit_original:.4f} * {wn_fit:.4f}^2 / (s^2 + 2*{zeta_fit:.4f}*{wn_fit:.4f}*s + {wn_fit:.4f}^2)"
        
        # 计算拟合误差
        residuals = output_signal - fitted_output_original
        rmse = np.sqrt(np.mean(residuals**2))
        
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((output_signal - np.mean(output_signal))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
        
        # 计算置信区间 (95%)
        param_std = np.sqrt(np.diag(pcov))
        confidence_intervals = {}
        param_names = list(parameters.keys())
        for i, name in enumerate(param_names):
            ci_half = 1.96 * param_std[i]
            confidence_intervals[name] = {
                'lower': parameters[name] - ci_half,
                'upper': parameters[name] + ci_half
            }
        
        result = {
            'order': order,
            'parameters': parameters,
            'covariance': pcov.tolist(),
            'confidence_intervals': confidence_intervals,
            'fitted_output': fitted_output_original.tolist(),
            'rmse': float(rmse),
            'r_squared': float(r_squared),
            'step_amplitude': float(step_amplitude),
            'steady_state': float(np.mean(output_signal[-max(1, len(output_signal)//10):])),
            'transfer_function': tf_str
        }
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"系统辨识失败: {str(e)}")


def print_identification_result(result):
    """
    打印辨识结果
    
    参数:
        result: identify_system 返回的结果字典
    """
    print("\n" + "="*60)
    print(f"系统辨识结果 ({result['order']}阶系统)")
    print("="*60)
    
    print(f"\n传递函数:")
    print(f"  {result['transfer_function']}")
    
    print(f"\n参数估计:")
    for name, value in result['parameters'].items():
        ci = result['confidence_intervals'][name]
        print(f"  {name:8s} = {value:12.6f}  [95% CI: {ci['lower']:12.6f}, {ci['upper']:12.6f}]")
    
    print(f"\n拟合质量:")
    print(f"  RMSE      = {result['rmse']:.6f}")
    print(f"  R²        = {result['r_squared']:.6f}")
    print(f"  稳态值    = {result['steady_state']:.6f}")
    print(f"  阶跃幅值  = {result['step_amplitude']:.6f}")
    print("="*60 + "\n")


# 示例用法
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # 生成示例数据：一阶系统
    print("示例 1: 一阶系统辨识")
    t_test = np.linspace(0, 5, 500)
    K_true = 2.0
    tau_true = 0.8
    
    # 生成理想阶跃响应并加噪声
    input_step = np.ones_like(t_test)
    input_step[t_test < 1.0] = 0
    output_test = first_order_step_response(t_test, K_true, tau_true, t0=1.0)
    output_test += np.random.normal(0, 0.05, len(output_test))  # 添加噪声
    
    # 辨识
    result1 = identify_system(t_test, input_step, output_test, order=1)
    print_identification_result(result1)
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(t_test, output_test, 'b.', label='实际输出', markersize=3, alpha=0.5)
    plt.plot(t_test, result1['fitted_output'], 'r-', label='辨识模型', linewidth=2)
    plt.xlabel('时间 (s)')
    plt.ylabel('输出')
    plt.title(f'一阶系统辨识 (R² = {result1["r_squared"]:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('identification_example_order1.png', dpi=150, bbox_inches='tight')
    print("已保存图片: identification_example_order1.png")
    
    # 生成示例数据：二阶系统
    print("\n示例 2: 二阶系统辨识")
    K_true2 = 1.5
    wn_true = 5.0
    zeta_true = 0.4
    
    output_test2 = second_order_step_response(t_test, K_true2, wn_true, zeta_true, t0=1.0)
    output_test2 += np.random.normal(0, 0.03, len(output_test2))
    
    result2 = identify_system(t_test, input_step, output_test2, order=2)
    print_identification_result(result2)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_test, output_test2, 'b.', label='实际输出', markersize=3, alpha=0.5)
    plt.plot(t_test, result2['fitted_output'], 'r-', label='辨识模型', linewidth=2)
    plt.xlabel('时间 (s)')
    plt.ylabel('输出')
    plt.title(f'二阶系统辨识 (R² = {result2["r_squared"]:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('identification_example_order2.png', dpi=150, bbox_inches='tight')
    print("已保存图片: identification_example_order2.png\n")
