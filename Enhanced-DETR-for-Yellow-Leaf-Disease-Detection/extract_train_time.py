import os
import datetime

file_paths = {
              "effi-v2S_ddetr": None,
}

file_paths = {k: os.path.join(".", "weight", k, "train.log")
              for k in file_paths}

"""
Epoch: [0] Total time: 3:40:15 (0.4469 s / it)
Epoch: [1] Total time: 3:42:24 (0.4513 s / it)
Epoch: [2] Total time: 3:45:43 (0.4580 s / it)
Epoch: [3] Total time: 3:48:07 (0.4629 s / it)
Epoch: [4] Total time: 3:48:14 (0.4631 s / it)
"""


for name, path in file_paths.items():
    times = []
    with open(path, 'r') as f:
        for line in f:
            if 'Epoch:' in line and 'Total time:' in line:
                time_str = line.strip().split('Total time: ')[1].split(' (')[0]
                h, m, s = map(int, time_str.split(':'))
                total_seconds = datetime.timedelta(
                    hours=h, minutes=m, seconds=s).total_seconds()
                times.append(total_seconds)

    average_time = sum(times) / len(times)
    # print(
    #     f"Average total time: {name:14} --- {str(datetime.timedelta(seconds=average_time)).split('.')[0]}")

    total_time = sum(times)
    print(f"Total time: {name:14} --- {total_time / 86400:.2f} days")


"""
Result:
Average total time: res-50_ddetr   --- 2:36:34
Average total time: effi-v2S_ddetr --- 3:44:31
Average total time: mb-v3L_ddetr   --- 2:01:20
Average total time: swin-T_ddetr   --- 2:23:02
"""

"""
Result:
Total time: res-50_ddetr   --- 5.44 days
Total time: effi-v2S_ddetr --- 7.80 days
Total time: mb-v3L_ddetr   --- 4.21 days
Total time: swin-T_ddetr   --- 4.97 days
"""
