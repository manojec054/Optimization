import torch
import time
import numpy as np
import tvm.relay as relay
import tvm
import argparse
import pandas as pd
import torch.cuda.profiler as profiler
# nvprof --profile-from-start off -o test.nvvp -f --print-gpu-summary  python pytorch_benchmark_explore.py
# nvvp -vm /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java
# nvprof --profile-from-start off --csv --log-file test.csv  python pytorch_benchmark_explore.py 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def matrix_cal(shape):
    N=shape

    print("#" * 50)
    print("Square Mat Shape = ", N)
    print("#" * 50)

    x = torch.randn((N, N), requires_grad=True).to(device)
    y = torch.randn((N, N), requires_grad=True).to(device)

    #warmup
    for itr in range(100):
        z = torch.matmul(x,y)

    #with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for itr in range(10):
        with torch.autograd.profiler.emit_nvtx() as pfvtx:
            profiler.start()
            z = torch.matmul(x,y)
        torch.cuda.synchronize()
        profiler.stop()


    torch_profile = []
    cuda_event = []
    time_api = []
    time_sync_api = []

    for itr in range(1000):
        ## Using CUDA Profile
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            z = torch.matmul(x,y)
        torch.cuda.synchronize() # wait for previous GPU operations but anyway not considered for calc
        gpu_time = sum(evt.self_cuda_time_total for evt in prof.function_events)
        torch_profile.append(gpu_time / 1000)


        ## Using CUDA events
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        z = torch.matmul(x,y)
        end.record()
        #torch.cuda.synchronize() 
        end.synchronize()
        cuda_event.append(start.elapsed_time(end))


        ## Using time API
        start = time.time()
        z = torch.matmul(x,y)
        end = time.time()
        time_api.append((end-start)*1000)
        torch.cuda.synchronize() # wait for previous GPU operations but dont consider for calc


        ## Using time API and Sync
        start = time.time()
        z = torch.matmul(x,y)
        torch.cuda.synchronize()
        end = time.time()
        time_sync_api.append((end-start)*1000)

        tvm.ma


    print(f"Using TORCH Profile = {np.array(torch_profile).mean()}ms")
    print(f"Using CUDA Events   = {np.array(cuda_event).mean()}ms")
    print(f"Using TIME API      = {np.array(time_api).mean()}ms")
    print(f"Using TIME SYNC API = {np.array(time_sync_api).mean()}ms")

    df = pd.DataFrame()
    df["torch_profile"] = torch_profile
    df["cuda_event"] = cuda_event
    df["time_api"] = time_api
    df["time_sync"] = time_sync_api
    df.to_csv("inference_benchmark.csv", index=False)

## RESULTS ## 50x50 shape
# cuda:0
# Using TORCH Profile = 0.0522901119158268ms
# Using CUDA Events   = 0.03761523292819038ms
# Using TIME API      = 0.018018245697021484ms
# Using TIME SYNC API = 0.022334575653076172ms

# 3200 matrix
# Using TORCH Profile = 79.95871226556491ms
# Using CUDA Events   = 39.38264159011841ms
# Using TIME API      = 0.04071927070617676ms
# Using TIME SYNC API = 39.40298104286194ms

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape', default=16,
                        help="Set the batch size used in inference", type=int)

    args = parser.parse_args()
    matrix_cal(args.shape)
    print("\n\n\n")