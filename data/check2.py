# superpixels.py
import subprocess

def get_unpickled_data(data_dir, name):
    # Use the full path to the Python executable of venvDisC4
    python_executable_path = '/mnt/nas/pkulkarni/debiasGNN_DisC/venvDisC4/bin/python'
    
    result = subprocess.run([python_executable_path, 'unpickle_data.py', data_dir, name], capture_output=True, text=True)
    output = result.stdout.splitlines()
    return output

data_dir = "/mnt/nas/pkulkarni/debiasGNN_DisC/Disc_source_code/data/"
name = "MNIST_75sp_0.9"
print("calling function")
data = get_unpickled_data(data_dir, name)
print("done calling")
train = data[0]
val = data[1]
biased_test = data[2]
unbiased_test = data[3]