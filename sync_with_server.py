import paramiko
import argparse
import os
from scp import SCPClient
import time

parser = argparse.ArgumentParser(description='Live sync with Leonhard')
parser.add_argument('model_name', metavar='Model Name', type=str,
                    help='Model name to use for the checkpoint')
# this can also be changed
parser.add_argument('username', metavar='SSH username', type=str,
                    help='Leonhard username')
parser.add_argument('project_path', metavar='Project path directory', type=str,
                    help='Path for the directory of the project')
# password, we can change this if we want to put a default password
parser.add_argument('--password', '-p', type=str, help='Password for ssh auth if no keys are configured',
                    default=None)

parser.add_argument('--sync_rate', '-s', type=int, help='Syncing rate (seconds)',
                    default=30)

args = parser.parse_args()


HOSTNAME = 'login.leonhard.ethz.ch'

ssh_client = paramiko.SSHClient()
ssh_client.load_system_host_keys()
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# if no password is configured then use the key
if args.password is None:
    ssh_client.connect(hostname=HOSTNAME, username=args.username, compress=True)
else:
    ssh_client.connect(hostname=HOSTNAME, username=args.username, password=args.password, compress=True)


output_model_dir = os.path.join(args.project_path, 'output', args.model_name)
scp = SCPClient(ssh_client.get_transport())

# if no specified mirror directory then use default
mirror_dir = os.path.join(os.path.curdir, 'mirrored')
if not os.path.exists(mirror_dir):
    os.makedirs(mirror_dir)

def get_seconds():
    return int(round(time.time()))

while True:
    print('Syncing...')
    start_time = get_seconds()
    scp.get(output_model_dir, mirror_dir, recursive=True, preserve_times=True)
    end_time = get_seconds()
    print(f'...finished syncing in {end_time - start_time} s')
    time.sleep(args.sync_rate)