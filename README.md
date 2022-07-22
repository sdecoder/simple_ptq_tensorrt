
trtexec --loadEngine=resnet50.int8.engine --batch=8192 --streams=8 --verbose --avgRuns=10
[07/22/2022-23:37:09] [I] === Performance summary ===
[07/22/2022-23:37:09] [I] Throughput: 1.29477e+08 qps
[07/22/2022-23:37:09] [I] Latency: min = 0.360107 ms, max = 0.677246 ms, mean = 0.521576 ms, median = 0.522217 ms, percentile(99%) = 0.615723 ms
[07/22/2022-23:37:09] [I] Enqueue Time: min = 0.0136719 ms, max = 0.0453186 ms, mean = 0.0148635 ms, median = 0.0142822 ms, percentile(99%) = 0.0231934 ms
[07/22/2022-23:37:09] [I] H2D Latency: min = 0.00708008 ms, max = 0.0322266 ms, mean = 0.0125107 ms, median = 0.0122681 ms, percentile(99%) = 0.0178833 ms
[07/22/2022-23:37:09] [I] GPU Compute Time: min = 0.338623 ms, max = 0.656982 ms, mean = 0.499129 ms, median = 0.499756 ms, percentile(99%) = 0.593628 ms
[07/22/2022-23:37:09] [I] D2H Latency: min = 0.00509644 ms, max = 0.0241699 ms, mean = 0.00993502 ms, median = 0.00976562 ms, percentile(99%) = 0.0151367 ms
[07/22/2022-23:37:09] [I] Total Host Walltime: 3.00126 s
[07/22/2022-23:37:09] [I] Total GPU Compute Time: 23.6767 s
[07/22/2022-23:37:09] [W] * GPU compute time is unstable, with coefficient of variance = 7.05215%.
[07/22/2022-23:37:09] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[07/22/2022-23:37:09] [I] Explanations of the performance metrics are printed in the verbose logs.


trtexec --loadEngine=resnet50.fp16.engine --batch=8192 --streams=8 --verbose --avgRuns=10
[07/22/2022-23:56:06] [I] === Performance summary ===
[07/22/2022-23:56:06] [I] Throughput: 1.24304e+08 qps
[07/22/2022-23:56:06] [I] Latency: min = 0.118164 ms, max = 1.01929 ms, mean = 0.539201 ms, median = 0.538818 ms, percentile(99%) = 0.639679 ms
[07/22/2022-23:56:06] [I] Enqueue Time: min = 0.0134277 ms, max = 0.133057 ms, mean = 0.0165005 ms, median = 0.0148926 ms, percentile(99%) = 0.0349121 ms
[07/22/2022-23:56:06] [I] H2D Latency: min = 0.00549316 ms, max = 0.395752 ms, mean = 0.0115327 ms, median = 0.0113525 ms, percentile(99%) = 0.0170898 ms
[07/22/2022-23:56:06] [I] GPU Compute Time: min = 0.105591 ms, max = 0.750916 ms, mean = 0.517884 ms, median = 0.517578 ms, percentile(99%) = 0.618896 ms
[07/22/2022-23:56:06] [I] D2H Latency: min = 0.00500488 ms, max = 0.0252075 ms, mean = 0.00978184 ms, median = 0.00965881 ms, percentile(99%) = 0.0144043 ms
[07/22/2022-23:56:06] [I] Total Host Walltime: 3.00188 s
[07/22/2022-23:56:06] [I] Total GPU Compute Time: 23.5896 s
[07/22/2022-23:56:06] [W] * GPU compute time is unstable, with coefficient of variance = 7.3408%.
[07/22/2022-23:56:06] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[07/22/2022-23:56:06] [I] Explanations of the performance metrics are printed in the verbose logs.

trtexec --loadEngine=resnet50.tf32.engine --batch=8192 --streams=8 --verbose --avgRuns=10
[07/22/2022-23:59:12] [I] === Performance summary ===
[07/22/2022-23:59:12] [I] Throughput: 1.25112e+08 qps
[07/22/2022-23:59:12] [I] Latency: min = 0.391785 ms, max = 0.707275 ms, mean = 0.537911 ms, median = 0.5354 ms, percentile(99%) = 0.636963 ms
[07/22/2022-23:59:12] [I] Enqueue Time: min = 0.0134277 ms, max = 0.0513916 ms, mean = 0.0146272 ms, median = 0.0141602 ms, percentile(99%) = 0.0234375 ms
[07/22/2022-23:59:12] [I] H2D Latency: min = 0.00683594 ms, max = 0.0310059 ms, mean = 0.011475 ms, median = 0.0112305 ms, percentile(99%) = 0.0170898 ms
[07/22/2022-23:59:12] [I] GPU Compute Time: min = 0.371033 ms, max = 0.688965 ms, mean = 0.516601 ms, median = 0.514069 ms, percentile(99%) = 0.61499 ms
[07/22/2022-23:59:12] [I] D2H Latency: min = 0.00488281 ms, max = 0.0249023 ms, mean = 0.00983961 ms, median = 0.00976562 ms, percentile(99%) = 0.0146484 ms
[07/22/2022-23:59:12] [I] Total Host Walltime: 3.00147 s
[07/22/2022-23:59:12] [I] Total GPU Compute Time: 23.681 s
[07/22/2022-23:59:12] [W] * GPU compute time is unstable, with coefficient of variance = 5.8492%.
[07/22/2022-23:59:12] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[07/22/2022-23:59:12] [I] Explanations of the performance metrics are printed in the verbose logs.

trtexec --loadEngine=resnet50.fp32.engine --batch=8192 --streams=8 --verbose --avgRuns=10
[07/23/2022-00:01:31] [I] === Performance summary ===
[07/23/2022-00:01:31] [I] Throughput: 1.24417e+08 qps
[07/23/2022-00:01:31] [I] Latency: min = 0.395233 ms, max = 1.01245 ms, mean = 0.541006 ms, median = 0.540039 ms, percentile(99%) = 0.636841 ms
[07/23/2022-00:01:31] [I] Enqueue Time: min = 0.0134277 ms, max = 0.052887 ms, mean = 0.0152779 ms, median = 0.0143433 ms, percentile(99%) = 0.0285645 ms
[07/23/2022-00:01:31] [I] H2D Latency: min = 0.00683594 ms, max = 0.377075 ms, mean = 0.0115587 ms, median = 0.0114136 ms, percentile(99%) = 0.0166626 ms
[07/23/2022-00:01:31] [I] GPU Compute Time: min = 0.373535 ms, max = 0.713257 ms, mean = 0.519689 ms, median = 0.518799 ms, percentile(99%) = 0.615234 ms
[07/23/2022-00:01:31] [I] D2H Latency: min = 0.00488281 ms, max = 0.0258789 ms, mean = 0.00976126 ms, median = 0.00964355 ms, percentile(99%) = 0.0147705 ms
[07/23/2022-00:01:31] [I] Total Host Walltime: 3.00139 s
[07/23/2022-00:01:31] [I] Total GPU Compute Time: 23.6895 s
[07/23/2022-00:01:31] [W] * GPU compute time is unstable, with coefficient of variance = 5.72335%.
[07/23/2022-00:01:31] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[07/23/2022-00:01:31] [I] Explanations of the performance metrics are printed in the verbose logs.
