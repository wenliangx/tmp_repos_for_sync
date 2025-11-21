#!/usr/bin/env python3
"""
plot_bode.py

从 `results.json`（或指定 JSON）中读取频域测量点，分别绘制开环/闭环的角度与速度 Bode 图。
画出原始数据点并在对数频率上用样条拟合平滑曲线。

用法示例：
  python plot_bode.py --results results.json --out-dir bode_plots
"""

from pathlib import Path
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline


def collect_points(results, signal='angle'):
    """从 results dict 中收集 (freq, gain, phase, closed_loop) 点，signal in {'angle','speed'}"""
    pts = []
    for fname, data in results.items():
        meta = data.get('meta', {})
        closed = bool(meta.get('closed_loop'))
        fd = data.get('frequency_domain', {}).get(signal)
        if not fd:
            continue
        f = fd.get('freq_hz')
        g = fd.get('gain')
        ph = fd.get('phase_deg')
        if f is None or g is None or ph is None:
            continue
        pts.append((float(f), float(g), float(ph), closed, fname))
    return pts


def prepare_group(pts):
    """将点按 closed_loop 分组并排序，返回 dict {'closed': (f,g,ph), 'open': ...}"""
    closed = [(f,g,ph,name) for (f,g,ph,closed,name) in pts if closed]
    openp = [(f,g,ph,name) for (f,g,ph,closed,name) in pts if not closed]

    def to_arrays(lst):
        if not lst:
            return np.array([]), np.array([]), np.array([]), []
        lst_sorted = sorted(lst, key=lambda x: x[0])
        f = np.array([x[0] for x in lst_sorted])
        g = np.array([x[1] for x in lst_sorted])
        ph = np.array([x[2] for x in lst_sorted])
        names = [x[3] for x in lst_sorted]
        return f, g, ph, names

    return {'closed': to_arrays(closed), 'open': to_arrays(openp)}


def fit_spline_on_logfreq(freq, y_vals, s=None):
    """在 log10(freq) 上拟合样条并返回 callable(x_freq)->y。"""
    if len(freq) < 3:
        return None
    lx = np.log10(freq)
    # 平滑因子 s 可调整；缺省使用 len(freq)*var*0.01
    if s is None:
        s = max(1e-6, len(lx) * np.var(y_vals) * 0.01)
    sp = UnivariateSpline(lx, y_vals, s=s)

    def fn(xf):
        return sp(np.log10(xf))

    return fn


def bode_plot_grouped(freqs, gains, phases, label, ax_mag, ax_phase, color, marker='o'):
    """在提供的轴上绘制点和样条拟合曲线（如果有）。freqs: 1D array in Hz."""
    if freqs.size == 0:
        return
    # magnitude in dB
    mag_db = 20.0 * np.log10(gains)
    # unwrap phase in radians
    ph_rad = np.deg2rad(phases)
    ph_unwrapped = np.rad2deg(np.unwrap(ph_rad))

    ax_mag.semilogx(freqs, mag_db, marker+color, linestyle='None', label=label+' (data)')
    ax_phase.semilogx(freqs, ph_unwrapped, marker+color, linestyle='None', label=label+' (data)')

    # 拟合样条并绘制平滑曲线
    mag_fn = fit_spline_on_logfreq(freqs, mag_db)
    ph_fn = fit_spline_on_logfreq(freqs, ph_unwrapped)
    fit_info = None
    if mag_fn is not None and ph_fn is not None:
        xf = np.logspace(np.log10(freqs[0]), np.log10(freqs[-1]), 200)
        mag_fit = mag_fn(xf)
        ph_fit = ph_fn(xf)
        ax_mag.semilogx(xf, mag_fit, '-', color=color, alpha=0.8, label=label+' (fit)')
        ax_phase.semilogx(xf, ph_fit, '-', color=color, alpha=0.8, label=label+' (fit)')

        # 评估拟合质量（在测量点上）
        mag_fit_at_pts = mag_fn(freqs)
        ph_fit_at_pts = ph_fn(freqs)
        # RMSE
        rmse_mag = float(np.sqrt(np.mean((mag_db - mag_fit_at_pts) ** 2)))
        rmse_ph = float(np.sqrt(np.mean((ph_unwrapped - ph_fit_at_pts) ** 2)))
        # R^2
        ss_res_mag = np.sum((mag_db - mag_fit_at_pts) ** 2)
        ss_tot_mag = np.sum((mag_db - np.mean(mag_db)) ** 2)
        r2_mag = float(1 - ss_res_mag / ss_tot_mag) if ss_tot_mag > 0 else None
        ss_res_ph = np.sum((ph_unwrapped - ph_fit_at_pts) ** 2)
        ss_tot_ph = np.sum((ph_unwrapped - np.mean(ph_unwrapped)) ** 2)
        r2_ph = float(1 - ss_res_ph / ss_tot_ph) if ss_tot_ph > 0 else None

        fit_info = {
            'fit_type': 'spline_logfreq',
            'spline_s': None,
            # 'xf': xf.tolist(),
            # 'mag_fit_db': mag_fit.tolist(),
            # 'ph_fit_deg': ph_fit.tolist(),
            'rmse_mag_db': rmse_mag,
            'rmse_phase_deg': rmse_ph,
            'r2_mag': r2_mag,
            'r2_phase': r2_ph,
            'n_points': int(len(freqs)),
        }
    else:
        fit_info = {
            'fit_type': 'none',
            'n_points': int(len(freqs)),
        }

    return fit_info


def plot_bode_from_results(results, out_dir='bode_plots'):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # Collect both signals groups once
    pts_angle = collect_points(results, signal='angle')
    pts_speed = collect_points(results, signal='speed')
    groups_angle = prepare_group(pts_angle)
    groups_speed = prepare_group(pts_speed)

    # 1) Per-signal combined plot (both open & closed) - angle and speed
    for signal, groups in (('angle', groups_angle), ('speed', groups_speed)):
        fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        f_c, g_c, ph_c, _ = groups['closed']
        fit_closed = None
        if f_c.size:
            fit_closed = bode_plot_grouped(f_c, g_c, ph_c, 'closed-loop', ax_mag, ax_phase, color='C0')
        f_o, g_o, ph_o, _ = groups['open']
        fit_open = None
        if f_o.size:
            fit_open = bode_plot_grouped(f_o, g_o, ph_o, 'open-loop', ax_mag, ax_phase, color='C1', marker='s')

        ax_mag.set_ylabel('Magnitude (dB)')
        ax_mag.grid(True, which='both', ls='--', alpha=0.4)
        ax_phase.set_ylabel('Phase (deg)')
        ax_phase.set_xlabel('Frequency (Hz)')
        ax_phase.grid(True, which='both', ls='--', alpha=0.4)
        ax_mag.legend()
        ax_phase.legend()

        outp = Path(out_dir) / f'bode_{signal}.png'
        plt.tight_layout()
        plt.savefig(outp)
        plt.close(fig)
        print(f'Saved {outp}')

        fit_out = {
            'signal': signal,
            'closed': fit_closed,
            'open': fit_open,
        }
        with open(Path(out_dir) / f'bode_{signal}_fit.json', 'w', encoding='utf-8') as fh:
            json.dump(fit_out, fh, ensure_ascii=False, indent=2)
        print(f'Saved fit JSON: {Path(out_dir) / f"bode_{signal}_fit.json"}')

    # 2) Per-loop single-signal plots and both-signals per-loop plots
    for loop in ('open', 'closed'):
        # single-signal plots
        for signal, groups in (('angle', groups_angle), ('speed', groups_speed)):
            f, g, ph, _ = groups[loop]
            if f.size:
                fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
                fit_info = bode_plot_grouped(f, g, ph, f'{loop}-loop {signal}', ax_mag, ax_phase, color='C0' if loop=='closed' else 'C1')
                ax_mag.set_ylabel('Magnitude (dB)')
                ax_phase.set_ylabel('Phase (deg)')
                ax_phase.set_xlabel('Frequency (Hz)')
                ax_mag.grid(True, which='both', ls='--', alpha=0.4)
                ax_phase.grid(True, which='both', ls='--', alpha=0.4)
                ax_mag.legend()
                ax_phase.legend()
                outp = Path(out_dir) / f'bode_{loop}_{signal}.png'
                plt.tight_layout()
                plt.savefig(outp)
                plt.close(fig)
                print(f'Saved {outp}')

        # both-signals plot for this loop (angle + speed together)
        # collect data for both signals
        f_a, g_a, ph_a, _ = groups_angle[loop]
        f_s, g_s, ph_s, _ = groups_speed[loop]
        if (f_a.size or f_s.size):
            fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
            if f_a.size:
                bode_plot_grouped(f_a, g_a, ph_a, f'{loop}-loop angle', ax_mag, ax_phase, color='C0')
            if f_s.size:
                bode_plot_grouped(f_s, g_s, ph_s, f'{loop}-loop speed', ax_mag, ax_phase, color='C1', marker='s')

            ax_mag.set_ylabel('Magnitude (dB)')
            ax_phase.set_ylabel('Phase (deg)')
            ax_phase.set_xlabel('Frequency (Hz)')
            ax_mag.grid(True, which='both', ls='--', alpha=0.4)
            ax_phase.grid(True, which='both', ls='--', alpha=0.4)
            ax_mag.legend()
            ax_phase.legend()
            outp = Path(out_dir) / f'bode_{loop}_both.png'
            plt.tight_layout()
            plt.savefig(outp)
            plt.close(fig)
            print(f'Saved {outp}')


def main():
    parser = argparse.ArgumentParser(description='从 results.json 绘制 Bode 图（点 + spline 拟合）')
    parser.add_argument('--results', default='results.json', help='分析结果 JSON 文件（默认 results.json）')
    parser.add_argument('--out-dir', default='bode_plots', help='输出图片目录')
    args = parser.parse_args()

    p = Path(args.results)
    if not p.exists():
        raise FileNotFoundError(f'Results file not found: {p}')
    with open(p, 'r', encoding='utf-8') as fh:
        results = json.load(fh)

    plot_bode_from_results(results, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
