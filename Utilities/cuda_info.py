import psutil
import torch

###########################################################################
# print_gpu
###########################################################################
def print_gpu() -> str:
    total_cuda = torch.cuda.device_count()
    total_cpus = torch.cpu.device_count()

    rval = f"""
---
## CUDA (GPUs &amp; CPUs)

The default device is `{torch.get_default_device()}`.<br>
There are a total of `{total_cuda}` GPU(s) available.<br>
There are a total of `{total_cpus}` CPU(s) available.<br>
Currently selected GPU device is `{torch.cuda.current_device()}`.<br>
Currently selected CPU device is `{torch.cpu.current_device()}`.

### GPUs

| <span title='GPU Id'>id</span> | Name | Major | Minor | Processor Count | Total Memory | Integrated | Multi GPU Board | Max Threads per Processor | Arch Name | <span title='The average temperature of the GPU sensor in Degrees C (Centigrades)'>Temp.</span> | <span title='The clock speed of the GPU SM in Hz Hertz over the past sample period as given by `nvidia-smi`'>Clock</span> | <span title='The average power draw of the GPU sensor in mW (MilliWatts) over the past sample period as given by `nvidia-smi` for Fermi or newer fully supported devices'>Power</span> |
| ------------------------------ | ---- | ----- | ----- | --------------- | ------------ | ---------- | --------------- | ------------------------- | --------- | ----- | ----- | ----- |
"""
    for index in range(total_cuda):
        prop = torch.cuda.get_device_properties(index)
        rval += f"|{index}|{prop.name}|{prop.major}|{prop.minor}|{prop.multi_processor_count}|{prop.total_memory}|"
        rval += f"{prop.is_integrated}|{prop.is_multi_gpu_board}|{prop.max_threads_per_multi_processor}|{prop.gcnArchName}|{torch.cuda.temperature(index)}&deg;C|"
        rval += (
            f"{torch.cuda.clock_rate(index)}hz|{torch.cuda.power_draw(index)}mW|\n"
        )

    cpufreq = psutil.cpu_freq()
    rval += f"""

### CPUs

| Physical Cores | Total Cores | Max Freq | Min Freq | Curr Freq |
| -------------- | ----------- | -------- | -------- | --------- |
| {psutil.cpu_count(logical=False)} | {psutil.cpu_count(logical=True)} | {cpufreq.max:.2f}Mhz | {cpufreq.min:.2f}Mhz |

#### CPU Usage / Core


"""
    for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
        rval += f"* Core {i} : {percentage}%\n"
        
    rval += f"Total CPU Usage : {psutil.cpu_percent()}%\n"
    return rval


###########################################################################
# get_gpu
###########################################################################
def get_gpu():
    return 0 if torch.cuda.is_available() else -1