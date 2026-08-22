import os
import argparse
import pickle
import numpy as np
from pathlib import Path

def summarize_trajectory_stats(root_directory):
    all_dx = []
    all_dy = []
    
    file_paths = list(Path(root_directory).rglob('traj_data.pkl'))
    print(f"找到 {len(file_paths)} 个文件。开始处理...")

    for path in file_paths:
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                
                positions = np.array(data['position']) 
                
                if len(positions) < 2:
                    continue
                
                diffs = np.diff(positions, axis=0)
                
                all_dx.extend(diffs[:, 0])
                all_dy.extend(diffs[:, 1])
                
        except Exception as e:
            print(f"读取文件 {path} 时出错: {e}")

    all_dx = np.array(all_dx)
    all_dy = np.array(all_dy)

    if len(all_dx) == 0:
        print("未找到有效轨迹数据。")
        return None

    def get_stats(data):
        return {
            'Max': np.max(data),
            'Min': np.min(data),
            'Mean': np.mean(data),
            'Std': np.std(data),
            'Var': np.var(data),
            'Count': len(data)
        }

    stats = {
        'dx': get_stats(all_dx),
        'dy': get_stats(all_dy)
    }

    return stats



parser = argparse.ArgumentParser(description="Visual Navigation Transformer")
parser.add_argument(
    "--data",
    "-d",
    default="go_stanford",
    type=str,
    help="[go_stanford, recon, scand, sacson, tartan_drive]",
)
args = parser.parse_args()

root_dir = f'/data/Datasets/GNM_Datasets/process_data/{args.data}'  # 替换为你的目标文件夹路径
results = summarize_trajectory_stats(root_dir)

if results:
    print("\n--- 统计特征结果 ---")
    for axis in ['dx', 'dy']:
        print(f"\n[{axis} 的统计信息]:")
        for key, value in results[axis].items():
            print(f"  {key:8s}: {value:.6f}")