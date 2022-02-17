import itk
import numpy as np

DATA_ROOT_DIR = "/home/ll610/Onepiece/code_cs138/github/CoNIC-1/local_test/npy"
OUT_DIR = "/home/ll610/Onepiece/code_cs138/github/CoNIC-1/local_test/mha"

arr = np.load(f"{DATA_ROOT_DIR}/images.npy")
dump_itk = itk.image_from_array(arr)
itk.imwrite(dump_itk, f"{OUT_DIR}/images.mha")
dump_np = itk.imread(f"{OUT_DIR}/images.mha")
dump_np = np.array(dump_np)
# content check
assert np.sum(np.abs(dump_np - arr)) == 0