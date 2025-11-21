#!/usr/bin/env python3
"""
analyze_motor.py

读取指定的 .xlsx 文件（使用 `read_xlsx.read_xlsx_to_numpy`），
提取时间戳（第135列）、目标角度（第2列）、反馈角度（第4列）、反馈速度（第6列），
计算时域与频域的静态与动态特性，并打印/保存结果。

用法示例:
  python analyze_motor.py 14/your_file.xlsx --out results.json

支持批量文件处理。
"""

from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from read_xlsx import read_xlsx_to_numpy
from scipy import signal


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

def time_domain_step_metrics(t, target, output):
    res = {}
    
    def _analysis(clip_t, clip_target, clip_output):
        N = len(clip_t)
        init_val = np.mean(clip_target[: max(1, N//10)])
        final_val = np.mean(clip_target[- max(1, N//10):])
        delta = final_val - init_val
        res["target_initial"] = float(init_val)
        res["target_final"] = float(final_val)

        # step start: first index where |target - init| > 10%*|delta|
        thr = 0.1 * max(abs(delta), 1e-12)
        step_idxs = np.where(np.abs(clip_target - init_val) > thr)[0]
        if len(step_idxs) == 0:
            res["note"] = "无法检测到明显阶跃"
            return res
        step_start = step_idxs[0]
        t0 = clip_t[step_start]
        res["step_start_time"] = float(t0)

        # steady-state value of output
        ss_val = np.mean(clip_output[- max(1, N//10):])
        res["steady_state_output"] = float(ss_val)
        res["steady_state_error"] = float(final_val - ss_val)

        # overshoot
        peak = np.max(clip_output[step_start:])
        overshoot = (peak - final_val) / (abs(final_val) if abs(final_val) > 1e-12 else 1e-12) * 100.0
        res["overshoot_percent"] = float(overshoot)

        # rise time (10% -> 90%)
        low_level = 0.1 * final_val
        high_level = 0.9 * final_val
        try:
            t_low = clip_t[np.where(abs(clip_output) >= abs(low_level))[0][0]]
            t_high = clip_t[np.where(abs(clip_output) >= abs(high_level))[0][0]]
            res["rise_time"] = float(t_high - t_low)
        except Exception:
            res["rise_time"] = None

        # settling time: last time output leaves within 2% band around final_val
        band = 0.02 * max(abs(final_val), 1e-12)
        outside = np.where(np.abs(clip_output - final_val) > band)[0]
        if len(outside) == 0:
            res["settling_time"] = 0.0
        else:
            last_out = outside[-1]
            if last_out >= N-1:
                res["settling_time"] = None
            else:
                res["settling_time"] = float(clip_t[last_out+1] - t0)

        return res
    
    clip_target = target.copy()
    clip_output = output.copy()
    clip_t = t.copy()
    clip_d_target = np.gradient(clip_target, clip_t)
    while len(clip_target) > 0:
        try:
            start_index = np.where(clip_d_target > 1e-2)[0][0]
            clip_target = clip_target[start_index:]
            clip_output = clip_output[start_index:]
            clip_t = clip_t[start_index:]
            clip_d_target = np.gradient(clip_target, clip_t)
            end_index = np.where(clip_d_target < -1e-2)[0][0]
            res = _analysis(clip_t[:end_index], clip_target[:end_index], clip_output[:end_index])
            print(f"start_t: {clip_t[0]}, end_t: {clip_t[end_index]}")
            clip_target = clip_target[end_index:]
            clip_output = clip_output[end_index:]
            clip_t = clip_t[end_index:]
            clip_d_target = np.gradient(clip_target, clip_t)
        except Exception:
            print(Exception)
            break
    return res
    


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

    res["freq_hz"] = float(freq)
    res["input_amp"] = float(in_amp)
    res["output_amp"] = float(out_amp)
    res["gain"] = float(gain)
    res["phase_deg"] = float(phase_deg)
    return res


def analyze_file(path):
    arr = read_xlsx_to_numpy(path)
    # convert 1-based column indices (user provided) to 0-based
    idx_time = 0
    idx_target = 1
    idx_feedback = 3
    idx_speed = 5

    if arr.shape[1] <= max(idx_time, idx_target, idx_feedback, idx_speed):
        raise ValueError(f"文件列数小于所需列索引，请检查文件 {path}")

    t = arr[:, idx_time].astype(float)
    t = (t - t[0])/2500.0
    target = arr[:, idx_target].astype(float)
    feedback = arr[:, idx_feedback].astype(float)
    speed = arr[:, idx_speed].astype(float)
    d_feedback = np.gradient(feedback, t)
    Fs = 2500       # 采样频率 (Hz)
    Fc = 10         # 截止频率 (Hz)，例如设计一个低通滤波器
    NumTaps = 100   # 滤波器阶数 (通常越大，过渡带越陡峭)
    # 1.1 计算归一化截止频率
    normalized_cutoff = Fc / (Fs / 2)
    # 1.2 设计 FIR 滤波器系数 h (脉冲响应)
    h = signal.firwin(
        numtaps=NumTaps, 
        cutoff=normalized_cutoff, 
        window='hamming', 
        pass_zero='lowpass'
    )
    d_feedback = signal.filtfilt(h, 1.0, d_feedback)
    N = len(d_feedback)
    start_idx = N // 10
    end_idx = N - N // 10

    meta = parse_filename(path)

    results = {
        "meta": meta,
        "time_domain": {},
        "frequency_domain": {},
    }

    if meta.get("closed_loop") == True:
        if meta.get("type") == 0:
            td = time_domain_step_metrics(t[start_idx:end_idx], target[start_idx:end_idx], feedback[start_idx:end_idx])
            results["time_domain"] = td
        elif meta.get("type") == 1:
            known_freq = meta.get("freq")
            fd = frequency_characteristics(t[start_idx:end_idx], target[start_idx:end_idx], feedback[start_idx:end_idx], known_freq=known_freq)
            results["frequency_domain"]["angle"] = fd
            fd_speed = frequency_characteristics(t[start_idx:end_idx], target[start_idx:end_idx], d_feedback[start_idx:end_idx], known_freq=known_freq)
            results["frequency_domain"]["speed"] = fd_speed
    elif meta.get("closed_loop") == False:
        if meta.get("type") == 0:
            td = time_domain_step_metrics(t[start_idx:end_idx], d_feedback[start_idx:end_idx], feedback[start_idx:end_idx])
            results["time_domain"] = td
        elif meta.get("type") == 1:
            known_freq = meta.get("freq")
            fd = frequency_characteristics(t[start_idx:end_idx], target[start_idx:end_idx], feedback[start_idx:end_idx], known_freq=known_freq)
            results["frequency_domain"]["angle"] = fd
            fd_speed = frequency_characteristics(t[start_idx:end_idx], target[start_idx:end_idx], d_feedback[start_idx:end_idx], known_freq=known_freq)
            results["frequency_domain"]["speed"] = fd_speed

    return results


def save_plots(out_dir, name, t, target, feedback, speed, d_feedback):
    os.makedirs(out_dir, exist_ok=True)
    base = Path(name).stem
    # 时间域图
    plt.figure(figsize=(10, 6))
    plt.plot(t, target, label='target')
    plt.plot(t, feedback, label='feedback')
    plt.plot(t, speed, label='speed')
    plt.plot(t, d_feedback, label='d_feedback')
    plt.xlabel('time (s)')
    plt.ylabel('value')
    plt.title(f'{base} - time domain')
    plt.legend()
    plt.grid(True)
    p1 = Path(out_dir) / f"{base}_time.png"
    plt.tight_layout()
    plt.savefig(p1)
    plt.close()

    return str(p1)


def main():
    parser = argparse.ArgumentParser(description="分析电机时域与频域特性。")
    parser.add_argument("files", nargs="*", help="要分析的 xlsx 文件路径")
    parser.add_argument("--dir", default="14", help="要遍历的目录（若指定，则忽略 files 参数）")
    parser.add_argument("--out", default="results.json", help="可选：保存结果为 JSON 文件")
    parser.add_argument("--plots-dir", default="plots", help="保存图片的目录，默认 'plots'")
    parser.add_argument("--save-plots", default=True, action="store_true", help="若指定，则为每个文件保存图像到 --plots-dir")
    parser.add_argument("--metrics-out", default="metrics.txt", help="导出常用指标的文本文件，默认 'metrics.txt'")
    parser.add_argument("--time-col", type=int, default=1, help="时间戳列（1-based），默认 135")
    parser.add_argument("--target-col", type=int, default=2, help="目标角度列（1-based），默认 2")
    parser.add_argument("--feedback-col", type=int, default=4, help="反馈角度列（1-based），默认 4")
    parser.add_argument("--speed-col", type=int, default=6, help="反馈速度列（1-based），默认 6")

    args = parser.parse_args()

    # 构建要分析的文件列表：优先使用 --dir 遍历所有 .xlsx
    files_to_process = []
    if args.dir:
        d = Path(args.dir)
        if d.exists() and d.is_dir():
            files_to_process = sorted([str(p) for p in d.glob('*.xlsx') if not p.name.startswith('~$')])
        else:
            print(f"目录不存在: {d}")
            return
    else:
        files_to_process = args.files

    all_results = {}
    metrics_lines = []
    for f in files_to_process:
        p = Path(f)
        try:
            res = analyze_file(p)
            all_results[p.name] = res
            print(json.dumps({p.name: res}, indent=2, ensure_ascii=False))

            # 导出常用指标到 metrics_lines
            meta = res.get('meta', {})
            td = res.get('time_domain', {})
            fd_angle = res.get('frequency_domain', {}).get('angle', {})
            fd_speed = res.get('frequency_domain', {}).get('speed', {})
            line = (
                f"{p.name}\tclosed_loop={meta.get('closed_loop')}\ttype={meta.get('type')}\t"
                f"amp={meta.get('amp')}\tfreq={meta.get('freq')}\t"
                f"overshoot%={td.get('overshoot_percent')}\trise_time={td.get('rise_time')}\t"
                f"settling_time={td.get('settling_time')}\tss_error={td.get('steady_state_error')}\t"
                f"gain_angle={fd_angle.get('gain')}\tphase_angle={fd_angle.get('phase_deg')}\t"
                f"gain_speed={fd_speed.get('gain')}\tphase_speed={fd_speed.get('phase_deg')}"
            )
            metrics_lines.append(line)

            # 保存图像
            if args.save_plots:
                tcol = int(args.time_col) - 1
                arr = read_xlsx_to_numpy(p)
                t = (arr[:, tcol] - arr[0, tcol]) / 2500.0
                target = arr[:, int(args.target_col)-1]
                feedback = arr[:, int(args.feedback_col)-1]
                speed = arr[:, int(args.speed_col)-1]
                d_feedback = np.gradient(feedback, t)
                Fs = 2500       # 采样频率 (Hz)
                Fc = 10         # 截止频率 (Hz)，例如设计一个低通滤波器
                NumTaps = 501   # 滤波器阶数 (通常越大，过渡带越陡峭)

                # 1.1 计算归一化截止频率
                normalized_cutoff = Fc / (Fs / 2)

                # 1.2 设计 FIR 滤波器系数 h (脉冲响应)
                h = signal.firwin(
                    numtaps=NumTaps, 
                    cutoff=normalized_cutoff, 
                    window='hamming', 
                    pass_zero='lowpass'
                )
                d_feedback = signal.filtfilt(h, 1.0, d_feedback)
                N = len(d_feedback)
                start_idx = N // 10
                end_idx = N - N // 10
                p1 = save_plots(args.plots_dir, p.name, t[start_idx:end_idx], target[start_idx:end_idx], feedback[start_idx:end_idx], speed[start_idx:end_idx], d_feedback[start_idx:end_idx])
                print(f"已保存图片: {p1}")
        except Exception as e:
            print(f"分析文件 {f} 失败: {e}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(all_results, fh, ensure_ascii=False, indent=2)
        print(f"已保存结果到 {args.out}")

    # 保存 metrics.txt
    if metrics_lines:
        with open(args.metrics_out, 'w', encoding='utf-8') as mh:
            mh.write('# filename\tclosed_loop\ttype\tamp\tfreq\tovershoot%\trise_time\tsettling_time\tss_error\tgain_angle\tphase_angle\tgain_speed\tphase_speed\n')
            for ln in metrics_lines:
                mh.write(ln + '\n')
        print(f"已保存常用指标到 {args.metrics_out}")


if __name__ == "__main__":
    main()
