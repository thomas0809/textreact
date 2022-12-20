import gdown
import os
import subprocess


def cmd(command):
    subp = subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding="utf-8")
    subp.wait()
    if subp.poll() == 0:
        print(subp.communicate()[1])
    else:
        print(f"{command} Failure!")

dirname_path = os.path.abspath(os.path.dirname(__file__))
work_dir = os.path.abspath(os.path.join(dirname_path, '..'))
file_list = [
    ('https://drive.google.com/uc?id=1da6JD0CPC4dCuWSy0cOn5g9uby7jL9PI', os.path.join(work_dir, 'dataset/source_dataset', 'USPTO_condition_final.zip')),
    ('https://drive.google.com/uc?id=1gFV2KdVKaLCTeb3nrzopyYHXbM0G_cr_', os.path.join(work_dir, 'outputs', 'Parrot_train_in_USPTO_Condition_enhance.zip')),
    ('https://drive.google.com/uc?id=1L7GnmESYwU7IFGnhMHD2qQH38Z2-kY1c', os.path.join(work_dir, 'outputs', 'Parrot_train_in_Reaxy_TotalSyn_Condition.zip')),    
    ('https://drive.google.com/uc?id=1uEqpkF4tPTlLIPdTyWJdXows7hKQbAAc', os.path.join(work_dir, 'dataset', 'pretrain_data.zip')),
    ('https://drive.google.com/uc?id=1hS-mHXJWF_NN4rA-UlrB0OGvSZt6ti_I', os.path.join(work_dir, 'outputs', 'Parrot_train_in_USPTO_Suzuki_Condition.zip'))
]


print('Downloading files...')

for url, save_path in file_list:
    if not os.path.exists(save_path):
        gdown.download(url, save_path, quiet=False)
        assert os.path.exists(save_path)
        
    else:
        print(f"{save_path} exists, skip downloading")
    cmd('unzip -o {} -d {}'.format(save_path, os.path.dirname(save_path)))

        
