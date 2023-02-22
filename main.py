import os
import warnings
warnings.filterwarnings("ignore")
from Code.projs.vit._jobs import train_job

if __name__ == "__main__":
    train_job()