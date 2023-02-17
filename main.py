import os
import warnings
warnings.filterwarnings("ignore")
from Code.projs.transformer._jobs import infer_job, train_job

if __name__ == "__main__":
    train_job()