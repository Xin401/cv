################################################################
# NOTE:                                                        #
# You can modify these values to train with different settings #
# p.s. this file is only for training                          #
################################################################

# Experiment Settings
exp_name   = 'adam_da' # name of experiment

# Model Options
model_type = 'resnet18' # 'mynet' or 'resnet18'
# model_type = 'mynet' # 'mynet' or 'resnet18'

# Learning Options
epochs     = 20           # train how many epochs
batch_size = 32           # batch size for dataloader 
use_adam   = True        # Adam or SGD optimizer
lr         = 1e-4         # learning rate
milestones = [10, 32, 45] # reduce learning rate at 'milestones' epochs
