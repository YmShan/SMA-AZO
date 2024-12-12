from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets import play_frame
import multiprocessing
import torch
import numpy as np
import random

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Instantiate the DVS128Gesture dataset
    datasets = DVS128Gesture('../datasets/DVS128Gesture', train=False, data_type='frame', split_by='number',
                             frames_number=16)

    # Create a process pool to handle all iterations
    pool = multiprocessing.Pool(processes=20)

    for i in range(len(datasets)):
        frame, label = datasets[i]

        # Use the process pool to process each iteration in parallel
        pool.apply_async(play_frame, args=(frame, './dvs128_visualization_results/' + str(i) + '_'))

    # Close the process pool and wait for all processes to complete
    pool.close()
    pool.join()

    print('ok')
