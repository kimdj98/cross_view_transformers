import os

# in case when /tmp file is full
os.system('find /tmp -type d -name "*wandb*" -exec rm -rf {} \;')

