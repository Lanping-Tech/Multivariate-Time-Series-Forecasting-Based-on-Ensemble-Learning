import pandas as pd
import numpy as np

def read_data_from_file(file_path, window_size=4, stride=1):
    o_data = []
    df = pd.read_excel(file_path)
    df[['AllocNodes', 'AllocCPUs', 'req_mem_per_cpu', 'Timelimit', 'CPUTimeRAW(s)', 'req_CPU', 'req_mem']] = df[['AllocNodes', 'AllocCPUs', 'req_mem_per_cpu', 'Timelimit', 'CPUTimeRAW(s)', 'req_CPU', 'req_mem']].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    job_names = df['Job_name'].unique()
    for job_name in job_names:
        o_data.append(df[df['Job_name'] == job_name].T.tail(7).T.values)

    data = []
    label = []
    for o_d in o_data:
        frequency = (o_d.shape[0] - window_size) // stride
        start_index = 0
        for i in range(frequency):
            stop_index = start_index + window_size
            sub_o_d = o_d[start_index:stop_index]
            data.append(sub_o_d)
            label.append(o_d[stop_index, -3:])
            start_index += stride
            
    data = np.array(data).astype('float32')
    label = np.array(label).astype('float32')
    return data, label

