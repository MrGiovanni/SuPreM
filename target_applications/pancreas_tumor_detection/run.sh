# STEP I. Train - train an AI on the training set

# Multiple GPUs
bash shell_scripts/step1.train.multigpu.sh segresnet pretrained_weights/supervised_suprem_segresnet_2100.pth 8 4 >> logs/segresnet.tiny.jhh.txt
# bash shell_scripts/step1.train.multigpu.sh swinunetr pretrained_weights/supervised_suprem_swinunetr_2100.pth  2 2 >> logs/swinunetr.jhh.txt

# Single GPU
# bash shell_scripts/step1.train.singlegpu.sh segresnet pretrained_weights/supervised_suprem_segresnet_2100.pth 8 4 >> logs/segresnet.tiny.jhh.txt
# bash shell_scripts/step1.train.singlegpu.sh swinunetr pretrained_weights/supervised_suprem_swinunetr_2100.pth  2 2 >> logs/swinunetr.jhh.txt

# STEP II. Test - test the AI on the test set

# STEP II.1 Make predictions

# Multiple GPUs
# bash shell_scripts/step2.inference.multigpu.sh segresnet out/segresnet.jhh/model.pth segresnet.jhh 8
# bash shell_scripts/step2.inference.multigpu.sh swinunetr out/swinunetr.jhh/model.pth swinunetr.jhh 8

# Single GPU
# bash shell_scripts/step2.inference.singlegpu.sh

## If you wanted to test multiple AI checkpoints
# for epoch in {270..100..-10}; do bash shell_scripts/step2.inference.multigpu.sh segresnet out/segresnet.jhh/checkpoint_epoch_$epoch.pth segresnet.jhh.checkpoint.$epoch 8; done

# STEP II.2 Perform post-processing, report eval metrics (e.g., sensitivity & specificity), save TP, TN, FP, FN IDs and visuals in the error_analysis folder

# bash shell_scripts/step3.eval.sh inference/segresnet.jhh /data/zzhou82/data/JHH_ROI_0.5mm segresnet.jhh >> logs/segresnet.jhh.txt

## If you wanted to evaluate multiple AI predictions
# for epoch in {270..100..-10}; do bash shell_scripts/step3.eval.sh inference/segresnet.jhh.checkpoint.$epoch /data/zzhou82/data/JHH_ROI_0.5mm segresnet.jhh.checkpoint.$epoch >> logs/segresnet.jhh.checkpoint.$epoch.txt; done 
