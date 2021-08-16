#!/bin/bash -l

# Activate your environment
spack env activate actinf-21052701
# Create a job name
baseJobName="Interception_AIF"
# Build the command you want to run. Note the extra decorators at the top of bash script.
str="#!/bin/bash\npython3 train_agent.py"
# Make sure the command is correct
echo $str
# Write to a temporary file called command.lock, it is just a text file
echo -e $str > command.lock
# Create your RC run command. Note the parameters used. If you need less than 16 GB then I suggest using 2XP4 GPUs.
sbatch -J ${baseJobName} -o "RC_generated_log_file.o" -e "RC_generated_error_file.e" --mem=16G --cpus-per-task=4 -p tier3 -A actinf --gres=gpu:p4:1 -t 1-0:0:0 --mail-user=zy8981@rit.edu --mail-type=ALL command.lock