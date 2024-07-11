ARCH="segresnet"

PRETRAINED_MODEL="pretrained_weights/supervised_suprem_segresnet_2100.pth"

BATCH_SIZE=8

NUM_SAMPLES=4

# This is the number of GPUs used for testing
NUM_GPUS=8


for i in {0..4}
do
  echo "Training fold $i..."
  bash shell_scripts/step1.train.multigpu.sh $ARCH $PRETRAINED_MODEL $BATCH_SIZE $NUM_SAMPLES jhh_train_fold_${i} $i >> logs/segresnet.jhh.fold_${i}.txt

  # echo "Inference fold $i..."
  # bash shell_scripts/step2.inference.multigpu.sh $ARCH out/$ARCH.jhh.fold_$i/model.pth $ARCH.jhh_fold_${i} $NUM_GPUS dataset/dataset_list/jhh_test_fold_${i}.txt

  # echo "Evaluation fold $i..."
  # bash shell_scripts/step3.eval.sh inference/$ARCH.jhh.fold_$i /data/zzhou82/data/JHH_ROI_0.5mm $ARCH.jhh_fold_${i} $i >> logs/segresnet.jhh.fold_${i}.txt
done
