import subprocess
import sys

def start():
    print("starting ...")
#     subprocess.Popen(['python3', ' -m', 'clip_server'])
    subprocess.call(['which python'], shell=True)
    subprocess.Popen(['python -m clip_server'], shell=True)
    
    print("server started ...")

    
if __name__ == '__main__':
    globals()[sys.argv[1]]()