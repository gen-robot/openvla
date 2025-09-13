### 安装
pip install -e . 即可，需要在同一个环境安装maniskill，建议新开一个环境，先安装openvla，再安装maniskill
运行时可能会出现import error，缺啥装啥...

### 测试

使用lora_co_training_adaptive.sh脚本。参数为：

vla_path - openvla-7b的路径
task_name - 任务名。目前已经有的是pick_to_plate, open_drawer
shuffle_buffer_size - 保证 shuffle_buffer_size % (grad_accumulation_steps * batch_size) = 0, 可以写 80_000
num_actions_chunk - 保持 16
env_id - Maniskill 的任务名，写 "PandaPutOnPlateInScene25Simple-v1" 或 "PandaOpenDrawer-v1"
adaptive_ratio - 是否开启 adaptive_ratio 算法，对于 fixed ratio 的算法写 false
fixed_ratio_value - ratio 参数