import warnings
warnings.filterwarnings("ignore")
import re
from Code.projs.transformer.Dataset import *
import os

if __name__ == "__main__":
    base_path = "../../data"
    proj_folder = "text_translator/fra-eng"
    data_file = "fra.txt"

    fname = os.path.join(base_path, proj_folder, data_file)
    print(fname)

    