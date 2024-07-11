# STEP 2. Inference

# STAGE 1
txtfilepath=$5
eachfilecontainslines=30
savetemptxtpath=dataset/dataset_list/temp
filesuffix=subset

python ./dataset/dataset_list/split_test_into_multiple_files.py --file_path $txtfilepath --save_dir $savetemptxtpath --lines_per_file $eachfilecontainslines --file_suffix $filesuffix --extension .txt

# STAGE 2
datapath=/ccvl/net/ccvl15/zzhou82/data/JHH_ROI_0.5mm
arch=$1
suprem_path=$2
savepath=./inference/$3.fold_$6 # 
num_gpus=$4

counter=1
for dataset_file in $(ls $savetemptxtpath)
do
  gpu_id=$(( (counter - 1) % $num_gpus ))
  CUDA_VISIBLE_DEVICES=$gpu_id python -W ignore inference.py --save_dir $savepath --checkpoint $suprem_path --data_root_path $datapath --data_txt_path $savetemptxtpath --num_class 5 --map_type jhh --backbone $arch --dataset_list $dataset_file --a_min -100 --a_max 200 --saveprobabilities --stage test &
  counter=$((counter + 1))
  sleep 1
done

wait

rm -r $savetemptxtpath
