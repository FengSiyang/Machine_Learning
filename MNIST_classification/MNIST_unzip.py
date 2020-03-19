"""
Down load MNIST datasets and unzip them
"""
import os
import os.path        # os file path
import urllib.request # url request
import gzip           # zip file
import shutil         # shell (sh) + tool (util)



if not os.path.exists('mnist'):
    os.mkdir('mnist')



def download_and_gzip(name):
    if not os.path.exists(name + '.gz'):
        urllib.request.urlretrieve('http://yann.lecun.com/exdb/' + name + '.gz', name + '.gz')
    if not os.path.exists(name):
        with gzip.open(name + '.gz', 'rb') as f_in, open(name, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)




def main():
    download_and_gzip('mnist/train-images-idx3-ubyte')
    download_and_gzip('mnist/train-labels-idx1-ubyte')
    download_and_gzip('mnist/t10k-images-idx3-ubyte')
    download_and_gzip('mnist/t10k-labels-idx1-ubyte')



if __name__ == "__main__":
    main()