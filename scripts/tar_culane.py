import os
import tarfile
import glob
import argparse

def extract_and_remove_tar_gz(file_path):
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall()
    os.remove(file_path)

def main(tar_path):
    tar_gz_files = glob.glob(os.path.join(tar_path, "*.tar.gz"))

    for file_path in tar_gz_files:
        extract_and_remove_tar_gz(file_path)
        print(f"解压并删除文件：{file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tar_path', type=str, required=True, help='Path to the directory containing tar.gz files')
    args = parser.parse_args()

    main(args.tar_path)
