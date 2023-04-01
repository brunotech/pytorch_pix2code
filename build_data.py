import os
import numpy as np
from shutil import copyfile

# Raw html data in /dataset/unprocessed/.
# Have to create training and evaluation directories manually.

input_path = './dataset/unprocessed/'
output_path = './dataset/'
eval_split_percent = 0.10

# List of every datapoint filename
paths = []
for f in os.listdir(input_path):
    if f.find('.gui') != -1:
        file_name = f[:f.find('.gui')]
        if os.path.isfile(f'{input_path}/{file_name}.png'):
            paths.append(file_name)

# Split the data in training and evaluation set
eval_sample_number = int(len(paths) * eval_split_percent)
np.random.shuffle(paths)
eval_set = paths[:eval_sample_number]
train_set = paths[eval_sample_number:]

for path in eval_set:
    copyfile(
        f'{input_path}/{path}.png',
        f'{os.path.dirname(output_path)}/evaluation/{path}.png',
    )
    copyfile(
        f'{input_path}/{path}.gui',
        f'{os.path.dirname(output_path)}/evaluation/{path}.gui',
    )

for path in train_set:
    copyfile(
        f'{input_path}/{path}.png',
        f'{os.path.dirname(output_path)}/training/{path}.png',
    )
    copyfile(
        f'{input_path}/{path}.gui',
        f'{os.path.dirname(output_path)}/training/{path}.gui',
    )