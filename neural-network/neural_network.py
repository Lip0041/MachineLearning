import pandas as pd
from scipy.io import loadmat


def main():
    data = loadmat('ex3data1.mat')
    print(data)


if __name__ == '__main__':
    main()