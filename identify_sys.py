from wl_.util import *
from scipy import signal
from matplotlib import pyplot as plt
import numpy as np
import json


def _process_all_file_data(files):
    mv_D__step___open = []
    mv_D__step___close = []
    mv_D__sine___open = {'angle': [], 'speed': []}
    mv_D__sine___close = {'angle': [], 'speed': []}
    dir = 'result'
    if not Path(dir).exists():
        Path(dir).mkdir(parents=True, exist_ok=True)
    for fn in files:
        try:
            mv_t, mv_y__input, mv_theta, mv_dot_theta__filtered, mv_omega = read_xlsx_to_numpy(fn)
            info = parse_filename(fn)
            name = ''
            if info['type'] == 0:
                name += 'step_'
            elif info['type'] == 1:
                name += f"sine_{info['freq']}Hz_"
            name += f"amp_{info['amp']}_freq_{info['freq']}_"
            if info['closed_loop']:
                name += 'closed'
            else:
                name += 'open'
            plot_angle_velocity(
                time=mv_t,
                angle=mv_theta,
                raw_velocity=mv_omega,
                filtered_velocity=mv_dot_theta__filtered,
                title=name,
                filename="result/"+name+".png")
            if info['type'] == 0 and info['closed_loop'] == False:
                data = time_domain_step_extractor(mv_t, mv_y__input, mv_theta, mv_dot_theta__filtered, mv_omega)
                if mv_D__step___open == []:
                    mv_D__step___open = data
                else:
                    mv_t__all, mv_u__all, mv_theta__all, mv_dot_theta__all, mv_omega__all = mv_D__step___open
                    mv_t__new, mv_u__new, mv_theta__new, mv_dot_theta__new, mv_omega__new = data
                    mv_D__step___open = (np.concatenate((mv_t__all, mv_t__new)),
                            np.concatenate((mv_u__all, mv_u__new)),
                            np.concatenate((mv_theta__all, mv_theta__new)),
                            np.concatenate((mv_dot_theta__all, mv_dot_theta__new)),
                            np.concatenate((mv_omega__all, mv_omega__new)))
            elif info['type'] == 0 and info['closed_loop'] == True:
                data = time_domain_step_extractor(mv_t, mv_y__input, mv_theta, mv_dot_theta__filtered, mv_omega)
                if mv_D__step___close == []:
                    mv_D__step___close = data
                else:
                    mv_t__all, mv_u__all, mv_theta__all, mv_dot_theta__all, mv_omega__all = mv_D__step___close
                    mv_t__new, mv_u__new, mv_theta__new, mv_dot_theta__new, mv_omega__new = data
                    mv_D__step___close = (np.concatenate((mv_t__all, mv_t__new)),
                             np.concatenate((mv_u__all, mv_u__new)),
                             np.concatenate((mv_theta__all, mv_theta__new)),
                             np.concatenate((mv_dot_theta__all, mv_dot_theta__new)),
                             np.concatenate((mv_omega__all, mv_omega__new)))
            elif info['type'] == 1 and info['closed_loop'] == False:
                res = frequency_characteristics(mv_t, mv_y__input, mv_dot_theta__filtered, known_freq=info['freq'])
                mv_D__sine___open['speed'].append((info['freq'] * 2 * np.pi, 20 * np.log10(res['gain']) if res['gain'] > 0 else -np.inf, np.degrees(res['phase_rad'])))
                res = frequency_characteristics(mv_t, mv_y__input, mv_theta, known_freq=info['freq'])
                mv_D__sine___open['angle'].append((info['freq'] * 2 * np.pi, 20 * np.log10(res['gain']) if res['gain'] > 0 else -np.inf, np.degrees(res['phase_rad'])))                
            elif info['type'] == 1 and info['closed_loop'] == True:
                res = frequency_characteristics(mv_t, mv_y__input, mv_theta, known_freq=info['freq'])
                mv_D__sine___close['angle'].append((info['freq'] * 2 * np.pi, 20 * np.log10(res['gain']) if res['gain'] > 0 else -np.inf, np.degrees(res['phase_rad'])))
        except Exception as e:
            print(f"处理文件 {fn} 时出错: {e}")
    return mv_D__step___open, mv_D__step___close, mv_D__sine___open, mv_D__sine___close


def main():
    files_to_process = []
    dir = './14'
    d = Path(dir)
    if d.exists() and d.is_dir():
        files_to_process = sorted([str(p) for p in d.glob('*.xlsx') if not p.name.startswith('~$')])
    else:
        print(f"目录不存在: {d}")
        return

    mv_D__step___open, mv_D__step___close, mv_D__sine___open, mv_D__sine___close = _process_all_file_data(files_to_process)

    dir = 'result'
    if not Path(dir).exists():
        Path(dir).mkdir(parents=True, exist_ok=True)

    mv_t__step___open, mv_u__step___open, mv_theta__step___open, mv_dot_theta__step___open, mv_omega__step___open  = mv_D__step___open
    mv_t__step___close, mv_u__step___close, mv_theta__step___close, mv_dot_theta__step___close, mv_omega__step___close  = mv_D__step___close

    data = [(mv_t__step___open, mv_u__step___open, mv_dot_theta__step___open),
            (mv_t__step___open, mv_u__step___open, mv_theta__step___open),
            (mv_t__step___close, mv_u__step___close, mv_theta__step___close)]
    func = [identify_open_speed, identify_open_angle, identify_close_angle]
    point = [mv_D__sine___open['speed'], mv_D__sine___open['angle'], mv_D__sine___close['angle']]
    output_files = ['result/open_step_speed.png',
                    'result/open_step_angle.png',
                    'result/close_step_angle.png']
    res = []

    for i in range(len(data)):
        mv_t__all, mv_u__all, mv_y__all = data[i]
        res_tmp = func[i](mv_t__all, mv_u__all, mv_y__all)
        res.append(res_tmp)

    avr_K = np.mean([r['parameters']['K'] for r in res])
    avr_wn = np.mean([r['parameters']['wn'] for r in res])
    avr_zeta = np.mean([r['parameters']['zeta'] for r in res])
    print(f"各阶跃响应辨识参数均值: K={avr_K}, wn={avr_wn}, zeta={avr_zeta}")
    
    Kp = 0.3
    nums = [[avr_K * avr_wn**2],
            [avr_K * avr_wn**2],
            [Kp * avr_K * avr_wn**2]]
    dens = [[1, 2*avr_zeta*avr_wn, avr_wn**2],
            [1, 2*avr_zeta*avr_wn, avr_wn**2, 0],
            [1, 2*avr_zeta*avr_wn, avr_wn**2, Kp * avr_K * avr_wn**2]]
    for i in range(len(res)):
        plot_bode(nums[i], dens[i], point_data=point[i], filename=output_files[i])
    
    json_data = [
        {
            "open_speed":
            {
                "num": res[0]['num'],
                "den": res[0]['den'],
                "rmse": res[0]['rmse'],
            },
            "open_angle":
            {
                "num": res[1]['num'],
                "den": res[1]['den'],
                "rmse": res[1]['rmse'],
            },
            "close_angle":
            {
                "num": res[2]['num'],
                "den": res[2]['den'],
                "rmse": res[2]['rmse'],
            },
            "average_parameters":
            {
                "K": avr_K,
                "wn": avr_wn,
                "zeta": avr_zeta,
            }
        }
    ]
    with open('result/identified_parameters.json', 'w') as f:
        json.dump(json_data, f, indent=4)
        


if __name__ == "__main__":
    main()