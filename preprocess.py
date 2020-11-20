import os
from data import kss
import hparams as hp

def write_metadata(train, val, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in train:
            f.write(m + '\n')
    with open(os.path.join(out_dir, 'val.txt'), 'w', encoding='utf-8') as f:
        for m in val:
            f.write(m + '\n')

def main():
    in_dir = hp.data_path
    out_dir = hp.preprocessed_path
    meta = hp.meta_name
    textgrid_name = hp.textgrid_name

    mel_out_dir = os.path.join(out_dir, "mel")
    if not os.path.exists(mel_out_dir):
        os.makedirs(mel_out_dir, exist_ok=True)

    ali_out_dir = os.path.join(out_dir, "alignment")
    if not os.path.exists(ali_out_dir):
        os.makedirs(ali_out_dir, exist_ok=True)

    f0_out_dir = os.path.join(out_dir, "f0")
    if not os.path.exists(f0_out_dir):
        os.makedirs(f0_out_dir, exist_ok=True)

    energy_out_dir = os.path.join(out_dir, "energy")
    if not os.path.exists(energy_out_dir):
        os.makedirs(energy_out_dir, exist_ok=True)

    if os.path.isfile(textgrid_name):
        os.system('mv ./{} {}'.format(textgrid_name, out_dir))

    if not os.path.exists(os.path.join(out_dir, textgrid_name.replace(".zip", ""))):
        os.system('unzip {} -d {}'.format(os.path.join(out_dir, textgrid_name), out_dir))


    if "kss" in hp.dataset:
        # kss version 1.3
        if "v.1.3" in meta:
            if not os.path.exists(os.path.join(in_dir, "wavs_bak")):
                os.system("mv {} {}".format(os.path.join(in_dir, "wavs"), os.path.join(in_dir, "wavs_bak")))        
                os.makedirs(os.path.join(in_dir, "wavs"))

        # kss version 1.4
        if "v.1.4" in meta:
            if not os.path.exists(os.path.join(in_dir, "wavs_bak")):
                os.makedirs(os.path.join(in_dir, "wavs"))
                os.system("mv {} {}".format(os.path.join(in_dir, "../", meta), os.path.join(in_dir)))
                for i in range(1, 5) : os.system("mv {} {}".format(os.path.join(in_dir, str(i)), os.path.join(in_dir, "wavs")))
                os.system("mv {} {}".format(os.path.join(in_dir, "wavs"), os.path.join(in_dir, "wavs_bak")))
                os.makedirs(os.path.join(in_dir, "wavs"))

        train, val = kss.build_from_path(in_dir, out_dir, meta)

    write_metadata(train, val, out_dir)
    
if __name__ == "__main__":
    main()
