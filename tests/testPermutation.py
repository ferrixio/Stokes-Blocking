from sys import path
from os import getcwd
path.insert(0, getcwd()+"\\src")    # adds src to the search list

from PermutationB import *

if __name__ == "__main__":

    B = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    print(B)