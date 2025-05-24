import argparse
from objects.CameraObject import CameraObject
import os
import warnings
warnings.filterwarnings("ignore")
import shutil
    


def check_arguments():
    
    parser = argparse.ArgumentParser(description='Processing logs.')
    parser.add_argument('--logs', type=str, help='The path to the logs directory')

    # Parse the arguments
    args = parser.parse_args()
    if args.logs:
        log_path = args.logs

        # Check if the path exists and is a directory
        if os.path.exists(log_path) and os.path.isdir(log_path):
            print(f'Log path {log_path} exists and is a directory. Creating Backup...')
            
            logs_backup_path = log_path + '_backup'
            if os.path.exists(logs_backup_path):
                print(f'Log backup path {logs_backup_path} exists. Deleting...')
                shutil.rmtree(logs_backup_path)
            # copying the logs safely first
            shutil.copytree(log_path, log_path + '_backup')
            # Delete the directory
            shutil.rmtree(log_path)
            
            print(f'Log path {log_path} deleted.')
        else:
            print(f'Log path {log_path} does not exist or is not a directory.')
    else:
        print('No log path provided.')

if __name__ == '__main__':
    check_arguments()
    camera_object=CameraObject()    
    camera_object.start()
