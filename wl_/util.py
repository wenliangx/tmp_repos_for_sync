from pathlib import Path
import numpy as np
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import odeint
import warnings
from scipy import signal
import matplotlib.pyplot as plt


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

    data = df_num.to_numpy(dtype=float)
    mv_t = data[:, 0] / 2500.0
    mv_y__input = data[:, 1]
    mv_theta = data[:, 3]
    mv_omega = data[:, 5]
    mv_dot_theta = np.gradient(mv_theta, mv_t)
    mv_F__s = 2500       # 采样频率 (Hz)
    mv_F__c = 10       # 截止频率 (Hz)，例如设计一个低通滤波器
    mv_N__Taps = 101   # 滤波器阶数 (通常越大，过渡带越陡峭)
    # 1.1 计算归一化截止频率
    normalized_cutoff = mv_F__c / (mv_F__s / 2)
    # 1.2 设计 FIR 滤波器系数 h (脉冲响应)
    h = signal.firwin(
        numtaps=mv_N__Taps, 
        cutoff=normalized_cutoff, 
        window='hamming', 
        pass_zero='lowpass'
    )
    mv_dot_theta__filtered = signal.lfilter(h, 1.0, mv_dot_theta)
    mv_y__output = data[:, 5]
    return mv_t, mv_y__input, mv_theta, mv_dot_theta__filtered, mv_omega

def parse_filename(fn: str):
    """从文件名解析信息：type、amp、freq、是否闭环(kp)。"""
    s = Path(fn).stem
    info = {
        "filename": Path(fn).name,
        "type": None,
        "amp": None,
        "freq": None,
        "closed_loop": True if "kp" in s.lower() else False,
    }

    parts = s.split(" ")
    for p in parts:
        if p.startswith("type"):
            try:
                info["type"] = int(p.replace("type", ""))
            except:
                pass
        if p.startswith("amp"):
            v = p.replace("amp", "").replace("_", ".")
            try:
                info["amp"] = float(v)
            except:
                pass
        if p.startswith("freq"):
            v = p.replace("freq", "").replace("_", ".")
            try:
                info["freq"] = float(v)
            except:
                pass

    return info

def ori_time_domain_step_extractor(mv_t, mv_y__input, mv_theta, mv_dot_theta__filtered, mv_omega):
    index = []
    mv_dot_y__input = np.gradient(mv_y__input, mv_t)
    mv_t___clip = mv_t.copy()
    mv_y__input___clip = mv_y__input.copy()
    mv_theta___clip = mv_theta.copy()
    mv_dot_theta__filtered___clip = mv_dot_theta__filtered.copy()
    mv_omega___clip = mv_omega.copy()
    base_index = 0
    while len(mv_y__input___clip) > 0:
        try:
            start_index = np.where(mv_dot_y__input > 1e-2)[0][0]
            mv_y__input___clip = mv_y__input___clip[start_index:]
            mv_t___clip = mv_t___clip[start_index:]
            base_index += start_index
            mv_dot_y__input = np.gradient(mv_y__input___clip, mv_t___clip)
            end_index = np.where(mv_dot_y__input < -1e-2)[0][0]
            index.append((base_index, base_index + end_index))
            mv_y__input___clip = mv_y__input___clip[end_index:]
            mv_t___clip = mv_t___clip[end_index:]
            base_index += end_index
            mv_dot_y__input = np.gradient(mv_y__input___clip, mv_t___clip)
        except Exception:
            print(Exception)
            break
    data = []
    for start, end in index:
        if data == []:
            data = [mv_t[start:end] - mv_t[start], mv_y__input[start:end], mv_theta[start:end] - mv_theta[start], mv_dot_theta__filtered[start:end], mv_omega[start:end]]

        else:
            data = [np.concatenate((data[0], mv_t[start:end] - mv_t[start])),
                    np.concatenate((data[1], mv_y__input[start:end])),
                    np.concatenate((data[2], mv_theta[start:end] - mv_theta[start])),
                    np.concatenate((data[3], mv_dot_theta__filtered[start:end])),
                    np.concatenate((data[4], mv_omega[start:end]))]
        
    return data


def time_domain_step_extractor(mv_t, mv_y__input, mv_theta, mv_dot_theta__filtered, mv_omega):
    index = []
    mv_dot_y__input = np.gradient(mv_y__input, mv_t)
    mv_t___clip = mv_t.copy()
    mv_y__input___clip = mv_y__input.copy()
    mv_theta___clip = mv_theta.copy()
    mv_dot_theta__filtered___clip = mv_dot_theta__filtered.copy()
    mv_omega___clip = mv_omega.copy()
    base_index = 0
    while len(mv_y__input___clip) > 0:
        try:
            start_index = np.where(mv_dot_y__input > 1e-2)[0][0]
            mv_y__input___clip = mv_y__input___clip[start_index:]
            mv_t___clip = mv_t___clip[start_index:]
            base_index += start_index
            mv_dot_y__input = np.gradient(mv_y__input___clip, mv_t___clip)
            end_index = np.where(mv_dot_y__input < -1e-2)[0][0]
            index.append((base_index, base_index + end_index))
            mv_y__input___clip = mv_y__input___clip[end_index:]
            mv_t___clip = mv_t___clip[end_index:]
            base_index += end_index
            mv_dot_y__input = np.gradient(mv_y__input___clip, mv_t___clip)
        except Exception:
            print(Exception)
            break
    data = []
    for start, end in index:
        if data == []:
            data = [mv_t[start:end] - mv_t[start], mv_y__input[start:end], (mv_theta[start:end] - mv_theta[start])/mv_y__input[end-1], (mv_dot_theta__filtered[start:end])/mv_y__input[end-1], (mv_omega[start:end])/mv_y__input[end-1]]

        else:
            data = [np.concatenate((data[0], mv_t[start:end] - mv_t[start])),
                    np.concatenate((data[1], mv_y__input[start:end])),
                    np.concatenate((data[2], (mv_theta[start:end] - mv_theta[start])/mv_y__input[end-1])),
                    np.concatenate((data[3], (mv_dot_theta__filtered[start:end])/mv_y__input[end-1])),
                    np.concatenate((data[4], (mv_omega[start:end])/mv_y__input[end-1]))]
        
    return data

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

def open_speed_response(t, K, wn, zeta):
    zero_index = np.where(t == 0.0)[0]
    y = None
    num = [K * wn**2]
    den = [1, 2*zeta*wn, wn**2]
    sys = signal.lti(num, den)
    for i in range(len(zero_index)):
        end_index = zero_index[i+1] if i+1 < len(zero_index) else len(t)
        T = np.linspace(0, t[end_index-1], end_index - zero_index[i])
        _, y_out = signal.step(sys, T=T)
        if y is None:
            y = y_out
        else:
            y = np.concatenate((y, y_out))
    return y


def close_angle_response(t, K, wn, zeta):
    zero_index = np.where(t == 0.0)[0]
    y = None
    Kp = 0.3
    num = [K * Kp * wn**2]
    den = [1, 2*zeta*wn, wn**2, K * Kp * wn**2]
    sys = signal.lti(num, den)
    for i in range(len(zero_index)):
        end_index = zero_index[i+1] if i+1 < len(zero_index) else len(t)
        T = np.linspace(0, t[end_index-1], end_index - zero_index[i])
        _, y_out = signal.step(sys, T=T)
        if y is None:
            y = y_out
        else:
            y = np.concatenate((y, y_out))
    return y


def open_angle_response(t, K, wn, zeta):
    zero_index = np.where(t == 0.0)[0]
    y = None
    num = [K * wn**2]
    den = [1, 2*zeta*wn, wn**2, 0]
    sys = signal.lti(num, den)
    for i in range(len(zero_index)):
        end_index = zero_index[i+1] if i+1 < len(zero_index) else len(t)
        T = np.linspace(0, t[end_index-1], end_index - zero_index[i])
        _, y_out = signal.step(sys, T=T)
        if y is None:
            y = y_out
        else:
            y = np.concatenate((y, y_out))
    return y

def third_order_step_response(t, K, tau1, tau2, tau3, t0=0.0):
    """
    三阶系统阶跃响应: G(s) = K / [(tau1*s + 1)(tau2*s^2 + tau3*s + 1)]
    
    参数:
        t: 时间数组
        K: 增益
        tau1: 一阶项时间常数
        tau2: 二阶项的 s^2 系数
        tau3: 二阶项的 s 系数
        t0: 初始时间偏移
    
    返回:
        y: 输出响应
    
    推导过程:
        传递函数: G(s) = K / [(tau1*s + 1)(tau2*s^2 + tau3*s + 1)]
        
        对于单位阶跃输入 1/s，输出为:
        Y(s) = K / [s(tau1*s + 1)(tau2*s^2 + tau3*s + 1)]
        
        二阶部分判别式: Delta = tau3^2 - 4*tau2
        
        情况1: Delta < 0 (共轭复根)
            二阶部分可改写为: tau2*s^2 + tau3*s + 1 = tau2*(s^2 + 2*zeta*wn*s + wn^2)
            其中: wn = 1/sqrt(tau2), zeta = tau3/(2*sqrt(tau2))
            
            时域响应包含振荡衰减项:
            y(t) = K*[1 - A1*e^(-t/tau1) - e^(-zeta*wn*t)*(B*cos(wd*t) + C*sin(wd*t))]
            wd = wn*sqrt(1 - zeta^2) (阻尼自然频率)
            
        情况2: Delta >= 0 (实根)
            可分解为两个一阶项，使用原来的三个指数项形式
    """
    t_shifted = t - t0
    T = np.linspace(0, t_shifted[-1], len(t_shifted))
    num = [K]
    den = [tau1 * tau2, tau2 + tau1 * tau3, tau1 + tau3, 1]
    sys = signal.lti(num, den)
    _, y_out = signal.step(sys, T=T)
    return y_out

    y = np.zeros_like(t)
    mask = t_shifted >= 0
    
    # 判别式
    Delta = tau3**2 - 4*tau2
    
    if Delta < 0:  # 共轭复根情况
        # 二阶部分参数
        wn = 1.0 / np.sqrt(tau2)  # 自然频率
        zeta = tau3 / (2.0 * np.sqrt(tau2))  # 阻尼比
        wd = wn * np.sqrt(1 - zeta**2)  # 阻尼自然频率
        
        # 一阶部分
        p1 = -1.0 / tau1  # 实极点
        
        # 部分分式分解系数
        # Y(s) = A0/s + A1/(tau1*s + 1) + (B*s + C)/(tau2*s^2 + tau3*s + 1)
        
        # A0 = K (稳态增益)
        A0 = K
        
        # A1 = -K*tau1 / (tau2/tau1^2 + tau3/tau1 + 1)
        denom = tau2/(tau1**2) + tau3/tau1 + 1.0
        A1 = -K * tau1 / denom
        
        # B 和 C 的计算
        # 通过留数定理或代数方法求解
        # B*s + C = K - A0*(...) - A1*(...)
        # 简化计算: B = -K*tau2/tau1 / denom, C = -K*(tau3 - tau2/tau1) / denom
        B = -K * tau2 / (tau1 * denom)
        C = -K * (tau3 - tau2/tau1) / denom
        
        # 转换为振幅-相位形式更稳定
        # (B*s + C)/(tau2*s^2 + tau3*s + 1) 的反拉普拉斯变换
        # 标准形式: [(B*tau2)*s + C]/(s^2 + 2*zeta*wn*s + wn^2) / tau2
        
        # 重新计算以确保数值稳定性
        # 使用标准二阶系统部分分式
        B_std = (B * tau2 + C * tau3) / tau2
        C_std = C / tau2
        
        # 时域响应
        exp_decay = np.exp(-zeta * wn * t_shifted[mask])
        cos_term = np.cos(wd * t_shifted[mask])
        sin_term = np.sin(wd * t_shifted[mask])
        
        # 组合各项
        y[mask] = (A0 + 
                   A1 * np.exp(p1 * t_shifted[mask]) +
                   exp_decay * ((B_std - C_std * zeta * wn / wd) * cos_term + 
                                (C_std / wd + B_std * zeta * wn / wd) * sin_term))
        
    else:  # 实根情况 (Delta >= 0)
        # 将二阶项分解为两个一阶项
        sqrt_delta = np.sqrt(Delta)
        # 二阶方程的根: tau2*s^2 + tau3*s + 1 = 0
        # s = (-tau3 ± sqrt(Delta)) / (2*tau2)
        r1 = (-tau3 + sqrt_delta) / (2 * tau2)
        r2 = (-tau3 - sqrt_delta) / (2 * tau2)
        
        # 转换为时间常数形式: (s - r1)(s - r2) = 0
        # 等价于: (1/(−r1))*(s + 1/tau_a) * (1/(−r2))*(s + 1/tau_b) 形式
        tau_a = -1.0 / r1 if abs(r1) > 1e-10 else 1e10
        tau_b = -1.0 / r2 if abs(r2) > 1e-10 else 1e10
        
        # 确保时间常数不相等
        eps = 1e-9
        if abs(tau1 - tau_a) < eps:
            tau_a += eps
        if abs(tau1 - tau_b) < eps:
            tau_b += 2*eps
        if abs(tau_a - tau_b) < eps:
            tau_b += eps
        
        # 部分分式分解系数
        A1 = tau1 / ((tau1 - tau_a) * (tau1 - tau_b))
        A2 = tau_a / ((tau_a - tau1) * (tau_a - tau_b))
        A3 = tau_b / ((tau_b - tau1) * (tau_b - tau_a))
        
        # 时域表达式
        y[mask] = K * (1.0 - 
                       A1 * np.exp(-t_shifted[mask] / tau1) -
                       A2 * np.exp(-t_shifted[mask] / tau_a) -
                       A3 * np.exp(-t_shifted[mask] / tau_b))
    
    return y

def identify_open_speed(time, input_signal, output_signal):
    time = np.asarray(time).flatten()
    input_signal = np.asarray(input_signal).flatten()
    output_signal = np.asarray(output_signal).flatten()
    
    if len(time) != len(input_signal) or len(time) != len(output_signal):
        raise ValueError("时间、输入、输出数组长度必须相同")
    try:
        K_init = 30
        wn_init = 60
        zeta_init = 0.3
        
        bounds = ([20, 0.01, 0.01],
                 [40, 100, 2.0])
        
        popt, pcov = curve_fit(
            open_speed_response,
            time,
            output_signal,
            p0=[K_init, wn_init, zeta_init],
            bounds=bounds,
            maxfev=100000
        )
        
        K_fit, wn_fit, zeta_fit = popt
        fitted_output = open_speed_response(time, K_fit, wn_fit, zeta_fit)
        
        K_fit_original = K_fit
        fitted_output_original = fitted_output
        
        parameters = {
            'K': K_fit_original,
            'wn': wn_fit,
            'zeta': zeta_fit
        }

        num = [K_fit_original * wn_fit**2]
        den = [1, 2*zeta_fit*wn_fit, wn_fit**2]
        
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
            'parameters': parameters,
            'covariance': pcov.tolist(),
            'confidence_intervals': confidence_intervals,
            'fitted_output': fitted_output_original.tolist(),
            'rmse': float(rmse),
            'r_squared': float(r_squared),
            'steady_state': float(np.mean(output_signal[-max(1, len(output_signal)//10):])),
            'transfer_function': tf_str,
            'num': num,
            'den': den
        }
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"系统辨识失败: {str(e)}")
    
def identify_close_angle(time, input_signal, output_signal):
    time = np.asarray(time).flatten()
    input_signal = np.asarray(input_signal).flatten()
    output_signal = np.asarray(output_signal).flatten()
    
    if len(time) != len(input_signal) or len(time) != len(output_signal):
        raise ValueError("时间、输入、输出数组长度必须相同")
    
    try:
        K_init = 30
        wn_init = 60
        zeta_init = 0.3
        
        bounds = ([20, 0.01, 0.01],
                 [40, 100, 2.0])

        popt, pcov = curve_fit(
            close_angle_response,
            time,
            output_signal,
            p0=[K_init, wn_init, zeta_init],
            bounds=bounds,
            maxfev=100000
        )
        Kp_fit = 0.3
        K_fit, wn_fit, zeta_fit = popt
        
        fitted_output = close_angle_response(
            time, K_fit, wn_fit, zeta_fit
        )
        
        # 恢复原始尺度
        K_fit_original = K_fit
        fitted_output_original = fitted_output
        
        parameters = {
            'K': K_fit_original,
            'wn': wn_fit,
            'zeta': zeta_fit
        }

        num = [K_fit_original * Kp_fit * wn_fit**2]
        den = [1, 
               2 * zeta_fit * wn_fit,
               wn_fit**2,
               K_fit_original * Kp_fit * wn_fit**2]
        tf_str = (f"G(s) = {K_fit_original: .4f} * {Kp_fit:.4f} * {wn_fit:.4f}^2 / "
                 f"[s^3 + 2 * {zeta_fit:.4f} * {wn_fit:.4f}*s^2 + {wn_fit:.4f}^2*s + {K_fit_original: .4f} * {Kp_fit:.4f} * {wn_fit:.4f}^2]")
        
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
            'parameters': parameters,
            'covariance': pcov.tolist(),
            'confidence_intervals': confidence_intervals,
            'fitted_output': fitted_output_original.tolist(),
            'rmse': float(rmse),
            'r_squared': float(r_squared),
            'steady_state': float(np.mean(output_signal[-max(1, len(output_signal)//10):])),
            'transfer_function': tf_str,
            'num': num,
            'den': den
        }
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"系统辨识失败: {str(e)}")
    
def identify_open_angle(time, input_signal, output_signal):
    time = np.asarray(time).flatten()
    input_signal = np.asarray(input_signal).flatten()
    output_signal = np.asarray(output_signal).flatten()
    
    if len(time) != len(input_signal) or len(time) != len(output_signal):
        raise ValueError("时间、输入、输出数组长度必须相同")
    
    step_end = np.max(input_signal)
    step_amplitude = step_end
    
    try:
        K_init = 30
        wn_init = 60
        zeta_init = 0.3
        
        bounds = ([20, 0.01, 0.01],
                 [40, 100, 2.0])
        
        popt, pcov = curve_fit(
            open_angle_response,
            time,
            output_signal,
            p0=[K_init, wn_init, zeta_init],
            bounds=bounds,
            maxfev=100000
        )
        
        K_fit, wn_fit, zeta_fit = popt
        
        # 排序时间常数
        fitted_output = open_angle_response(
            time, K_fit, wn_fit, zeta_fit
        )
    
        # 恢复原始尺度
        K_fit_original = K_fit
        fitted_output_original = fitted_output
        
        parameters = {
            'K': K_fit_original,
            'wn': wn_fit,
            'zeta': zeta_fit
        }
        num = [K_fit_original * wn_fit**2]
        den = [1, 
               2 * zeta_fit * wn_fit,
               wn_fit**2,
               0]
        tf_str = (f"G(s) = {K_fit_original:.4f} * {wn_fit:.4f}^2 / "
                 f"s[s^2 + 2*{zeta_fit:.4f} * {wn_fit:.4f}*s + {wn_fit:.4f}^2]")
        
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
            'parameters': parameters,
            'covariance': pcov.tolist(),
            'confidence_intervals': confidence_intervals,
            'fitted_output': fitted_output_original.tolist(),
            'rmse': float(rmse),
            'r_squared': float(r_squared),
            'steady_state': float(np.mean(output_signal[-max(1, len(output_signal)//10):])),
            'transfer_function': tf_str,
            'num': num,
            'den': den
        }
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"系统辨识失败: {str(e)}")

def identify_system(time, input_signal, output_signal, order=1):
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
                'parameters': 参数字典 (二阶: {'K', 'wn', 'zeta'}, 三阶: {'K', 'tau1', 'tau2', 'tau3'}),
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
    
    if order not in [2, 3]:
        raise ValueError("系统阶数必须为 2 或 3")
    
    # 检测阶跃幅值（取输入信号的稳态值与初始值之差）
    step_start = np.mean(input_signal[:max(1, len(input_signal)//100)])
    step_end = np.mean(input_signal[-max(1, len(input_signal)//100):])
    step_amplitude = step_end - step_start
    
    if abs(step_amplitude) < 1e-6:
        raise ValueError("输入信号不是有效的阶跃信号（幅值变化太小）")
    
    # 归一化：将输出按输入阶跃幅值缩放
    output_normalized = output_signal.copy()
    if abs(step_amplitude) > 1e-6:
        output_normalized = output_signal / abs(step_amplitude)

    mv_t__0 = time[0]
    # mv_t__0 = time[np.where(np.abs(output_signal) > 1e1)[0][0]]  # 估计阶跃开始时间点
    
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
        if order == 2:
            # 二阶系统辨识
            # 初始猜测
            K_init = steady_state if abs(steady_state) > 1e-6 else 1.0
            wn_init = 2 * np.pi / ((time[-1] - time[0]) * 0.2)  # 假设周期约为20%时间跨度
            zeta_init = 0.8  # 常见阻尼比
            
            
            def fit_func2(t, K, wn, zeta):
                return second_order_step_response(t, K, wn, zeta, mv_t__0)
            
            # 参数边界: K>0, wn>0, 0<zeta<5, t0在合理范围
            bounds = ([0, 0.1, 0],
                     [np.inf, 1000, 5])
            
            popt, pcov = curve_fit(
                fit_func2,
                time,
                output_normalized,
                p0=[K_init, wn_init, zeta_init],
                bounds=bounds,
                maxfev=50000
            )
            
            K_fit, wn_fit, zeta_fit = popt
            fitted_output = second_order_step_response(time, K_fit, wn_fit, zeta_fit, time[100])
            
            K_fit_original = K_fit * abs(step_amplitude)
            fitted_output_original = fitted_output * abs(step_amplitude)
            
            parameters = {
                'K': K_fit_original,
                'wn': wn_fit,
                'zeta': zeta_fit
            }

            num = [K_fit_original * wn_fit**2]
            den = [1, 2*zeta_fit*wn_fit, wn_fit**2]
            
            tf_str = f"G(s) = {K_fit_original:.4f} * {wn_fit:.4f}^2 / (s^2 + 2*{zeta_fit:.4f}*{wn_fit:.4f}*s + {wn_fit:.4f}^2)"
            
        else:  # order == 3
            # 三阶系统辨识
            K_init = 1.0
            
            # 初始时间常数猜测：从快到慢分布
            time_span = time[-1] - time[0]
            tau1_init = 0.77
            tau2_init = 0.025
            tau3_init = 0.0065 
            
            def fit_func3(t, K, tau1, tau2, tau3):
                # # 确保时间常数按从小到大排序
                # taus = np.sort([abs(tau1), abs(tau2), abs(tau3)])
                # # 避免时间常数太接近
                # if np.min(np.diff(taus)) < 1e-6:
                #     return np.ones_like(t) * 1e10
                
                return third_order_step_response(t, K, tau1, tau2, tau3, mv_t__0)
            
            # 参数边界
            bounds = ([0, 1e-8, 1e-8, 1e-8],
                     [1.2, 1.0, 0.1, 0.1])
            
            popt, pcov = curve_fit(
                fit_func3,
                time,
                output_normalized,
                p0=[K_init, tau1_init, tau2_init, tau3_init],
                bounds=bounds,
                maxfev=300000
            )
            
            K_fit, tau1_fit, tau2_fit, tau3_fit = popt
            
            # 排序时间常数
            taus_sorted = [tau1_fit, tau2_fit, tau3_fit]
            fitted_output = third_order_step_response(
                time, K_fit, taus_sorted[0], taus_sorted[1], taus_sorted[2], mv_t__0
            )
            
            # 恢复原始尺度
            K_fit_original = K_fit * abs(step_amplitude)
            fitted_output_original = fitted_output * abs(step_amplitude)
            
            parameters = {
                'K': K_fit_original,
                'tau1': taus_sorted[0],
                'tau2': taus_sorted[1],
                'tau3': taus_sorted[2]
            }

            # tmp_k = taus_sorted[0] * taus_sorted[1] * taus_sorted[2]
            # if tmp_k == 0:
            #     tmp_k = 1e-12  # 防止除零错误

            # num = [K_fit_original / tmp_k]
            # den = [1, 
            #        (taus_sorted[0]*taus_sorted[1] + taus_sorted[1]*taus_sorted[2] + taus_sorted[0]*taus_sorted[2]) / tmp_k,
            #        sum(taus_sorted) / tmp_k,
            #         1 / tmp_k]
            tmp_k = taus_sorted[0] * taus_sorted[1]
            if tmp_k == 0:
                tmp_k = 1e-12  # 防止除零错误

            num = [K_fit_original / tmp_k]
            den = [1, 
                   (taus_sorted[0]*taus_sorted[2] + taus_sorted[1]) / tmp_k,
                   (taus_sorted[0] + taus_sorted[2]) / tmp_k,
                    1 / tmp_k]
            tf_str = (f"G(s) = {K_fit_original:.4f} / "
                     f"[({taus_sorted[0]:.4f}*s + 1)({taus_sorted[1]:.4f}*s^2 + {taus_sorted[2]:.4f}*s + 1)]")
        
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
            'transfer_function': tf_str,
            'num': num,
            'den': den
        }
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"系统辨识失败: {str(e)}")
    
def sine_fit_amplitude_phase(t, x, freq=None):
    """对 x(t) 在已知或未知频率下拟合正弦，返回 (amp, phase(rad), freq)

    如果 freq 为 None，则用 FFT 估计主要频率。
    """
    N = len(t)
    dt = (t[1] - t[0]) if N > 1 else 1.0
    if freq is None:
        # 使用 FFT 估计频率
        X = np.fft.rfft(x - np.mean(x))
        freqs = np.fft.rfftfreq(N, d=dt)
        idx = np.argmax(np.abs(X[1:])) + 1
        freq = float(freqs[idx])

    omega = 2 * np.pi * freq
    # 线性最小二乘： x ~ A*sin(omega t) + B*cos(omega t)
    S = np.column_stack([np.sin(omega * t), np.cos(omega * t)])
    coeffs, *_ = np.linalg.lstsq(S, x, rcond=None)
    A, B = coeffs
    amp = float(np.hypot(A, B))
    phase = float(np.arctan2(B, A))  # 使得 A*sin + B*cos = amp*sin(omega t + phase)
    return amp, phase, freq

def frequency_characteristics(t, input_sig, output_sig, known_freq=None):
    res = {}
    # 估计输入和输出的幅值与相位
    in_amp, in_phase, freq = sine_fit_amplitude_phase(t, input_sig, freq=known_freq)
    out_amp, out_phase, _ = sine_fit_amplitude_phase(t, output_sig, freq=freq)

    gain = out_amp / (in_amp if in_amp != 0 else 1e-12)
    phase_lag = out_phase - in_phase
    # 转为度数并把相位限制在 [-180,180]
    phase_deg = np.degrees(phase_lag)
    phase_deg = (phase_deg + 180) % 360 - 180
    phase_rad = np.radians(phase_deg)

    res["freq_hz"] = float(freq)
    res["input_amp"] = float(in_amp)
    res["output_amp"] = float(out_amp)
    res["gain"] = float(gain)
    res["phase_rad"] = float(phase_rad)
    return res

def plot_bode(num, den, filename, point_data=None):
    """
    Plot Bode diagram for a second-order system
    
    Parameters:
        K: Gain
        wn: Natural frequency (rad/s)
        zeta: Damping ratio
        sine_data: List of sine identification data points (optional)
        filename: Output filename for the plot
    """
    # Create transfer function
    # G(s) = K*wn^2 / (s^2 + 2*zeta*wn*s + wn^2)
    
    sys = signal.TransferFunction(num, den)
    
    # Generate frequency range (automatic)
    w = np.logspace(-2, 3, 1000)  # From 0.01 to 1000 rad/s
    
    # Calculate Bode plot
    w, mag, phase = signal.bode(sys, w)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Magnitude plot
    ax1.semilogx(w, mag, 'b-', linewidth=2, label='Step Response ID')
    
    # Plot sine identification points if available
    if point_data and len(point_data) > 0:
        sine_freqs = [d[0] for d in point_data]
        sine_mags = [d[1] for d in point_data]
        ax1.semilogx(sine_freqs, sine_mags, 'ro', markersize=8, 
                    markeredgewidth=1.5, markerfacecolor='none', 
                    label='Sine Response ID', zorder=5)
    
    ax1.set_ylabel('Magnitude (dB)', fontsize=12)
    ax1.set_title(f'Bode Diagram of Identified System: {num}/{den}', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, which='both', alpha=0.3)
    ax2.set_xlim([1e-2, 1e3])
    ax1.legend(loc='best', fontsize=10)
    
    # Phase plot
    ax2.semilogx(w, phase, 'b-', linewidth=2, label='Step Response ID')
    
    # Plot sine identification phase points if available
    if point_data and len(point_data) > 0:
        sine_freqs = [d[0] for d in point_data]
        sine_phases = [d[2] for d in point_data]
        ax2.semilogx(sine_freqs, sine_phases, 'ro', markersize=8,
                    markeredgewidth=1.5, markerfacecolor='none',
                    label='Sine Response ID', zorder=5)
    ax2.set_xlabel('Frequency (rad/s)', fontsize=12)
    ax2.set_ylabel('Phase (deg)', fontsize=12)
    ax2.grid(True, which='both', alpha=0.3)
    ax2.set_xlim([1e-2, 1e3])
    # ax2.set_ylim([-180, 0])
    ax2.legend(loc='best', fontsize=10)
    
    # Add system parameters as text
    param_text = f'System Parameters:\n'
    param_text += f'num = {num}\n'
    param_text += f'den = {den}\n'
    
    ax1.text(0.02, 0.05, param_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Bode plot saved to: {filename}")
    plt.close()
    
    return fig, (ax1, ax2)

def plot_angle_velocity(time, angle, raw_velocity, filtered_velocity, title, filename):
    """
    Plot angle and velocity (raw and filtered) over time
    
    Parameters:
        time: Time array (numpy array or list)
        angle: Angle data (numpy array or list)
        raw_velocity: Raw angular velocity data (numpy array or list)
        filtered_velocity: Filtered angular velocity data (numpy array or list)
        title: Plot title (string)
        filename: Output filename for the plot (string)
    
    Returns:
        fig: Figure object
        (ax1, ax2): Tuple of axes objects
    """
    # Convert to numpy arrays
    time = np.asarray(time)
    angle = np.asarray(angle)
    raw_velocity = np.asarray(raw_velocity)
    filtered_velocity = np.asarray(filtered_velocity)
    
    # Create figure with two subplots (stacked vertically)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Angle vs Time
    ax1.plot(time, angle, 'b-', linewidth=1.5, label='Angle')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Angle (rad)', fontsize=12)
    ax1.set_title(f'{title} - Angle', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)
    
    # Plot 2: Angular Velocity vs Time
    ax2.plot(time, raw_velocity, 'r-', linewidth=1.0, alpha=0.5, label='Raw Velocity')
    ax2.plot(time, filtered_velocity, 'g-', linewidth=1.5, label='Filtered Velocity')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Angular Velocity (rad/s)', fontsize=12)
    ax2.set_title(f'{title} - Angular Velocity', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)
    
    # Add overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {filename}")
    plt.close()
    
    return fig, (ax1, ax2)