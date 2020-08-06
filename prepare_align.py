import os
from data import korean
import hparams as hp

def main():
    in_dir = hp.data_path
    if hp.dataset == "korean":
        korean.prepare_align(in_dir)
    
if __name__ == "__main__":
    main()
