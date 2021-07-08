import subprocess
import time
s1 = subprocess.Popen('python processor.py --train_type jointly --gpus 3 --dataset UCLA --use 1.0 --scale 0.05',shell=True)
s1.wait()