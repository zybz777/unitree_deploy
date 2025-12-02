import os

import numpy as np
import pandas as pd


class Go2Logger:
    def __init__(self, log_enable=True, log_type="flat", log_dir="logs_csv", file_name="obs.csv"):
        self.log_enable = log_enable
        if self.log_enable:
            save_dir = os.path.join(log_dir, log_type)
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, file_name)
            if os.path.exists(path):
                os.remove(path)
            self.df_path = os.path.join(save_dir, file_name)
        else:
            self.df_path = None

    def log(self, obs: dict, step: int):
        if not self.log_enable:
            return
        # 展开字典
        flat_dict = {}
        for key, val in obs.items():
            val = np.array(val).flatten()
            for i in range(len(val)):
                flat_dict[f"{key}_{i}"] = val[i]

        # 构建 DataFrame
        df = pd.DataFrame([flat_dict])

        df.insert(0, "step", step)  # 在最前面加一列 step
        df.to_csv(self.df_path, mode="a", header=not os.path.exists(self.df_path), index=False)
