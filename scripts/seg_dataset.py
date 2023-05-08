import os
import random


def organize_input(base, args):
    input_result = []
    output_result = []
    for k, src_name in args:
        input_path = os.path.join(base, src_name)
        src_file_base_name = src_name.split('.')[0]
        src_file_ext = src_name.split('.')[1]
        out_file_base_name = src_file_base_name + '_out_' + k + '.' + src_file_ext
        output_path = os.path.join(base, out_file_base_name)

        input_result.append(input_path)
        output_result.append(output_path)

    return input_result, output_result


def func(input_set, output_set, sample_num):
    zipped = zip(input_set, output_set)

    for in_path, out_path in zipped:
        do_sampling(in_path, out_path, sample_num)
        print("output:" + out_path + "\n" + "input:" + in_path + "\n")


def do_sampling(input_file_path, output_file_path, sample_num):
    input_file = open(input_file_path, "r")
    output_file = open(output_file_path, "w")
    samples = {}
    for line in input_file:
        parts = line.strip().split()
        image_path = parts[0]
        mp4_folder = image_path.split('/')[2]
        if mp4_folder not in samples:
            samples[mp4_folder] = []
        samples[mp4_folder].append(line)

    for mp4_folder in random.choices(list(samples.keys()), k=sample_num):
        for line in samples[mp4_folder]:
            output_file.write(line)

    input_file.close()
    output_file.close()


n = 100
base_path = "/root/autodl-tmp/dataset/CULane/list"
src = ['train_gt.txt', 'test.txt']

input_set, output_set = organize_input(base_path, src)
func(input_set, output_set, n)

