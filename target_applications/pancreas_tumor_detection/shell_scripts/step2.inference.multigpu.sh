# STEP 2. Inference

# STAGE 1
txtfilepath=dataset/dataset_list/jhh_test.txt # change to your saved text path
eachfilecontainslines=30 # change to the number you want each file to contain, increase the number if the GPU memory is not enough
savetemptxtpath=dataset/dataset_list/temp # change to the path of directory you want to temporarily save the split files, will be automatically deleted at the end
filesuffix=subset

python ./dataset/dataset_list/split_test_into_multiple_files.py --file_path $txtfilepath --save_dir $savetemptxtpath --lines_per_file $eachfilecontainslines --file_suffix $filesuffix --extension .txt

# STAGE 2
datapath=/data/zzhou82/data/JHH_ROI_0.5mm # change to the path of the dataset you want to inference 
arch=$1
suprem_path=$2
savepath=./inference/$3 # change to the path of directory your want to save inference
num_gpus=$4 # change to the number of GPUs you want to use

counter=1
for dataset_file in $(ls $savetemptxtpath)
do
  gpu_id=$(( (counter - 1) % $num_gpus )) # Calculate which GPU to use: the modulo operation will assign a GPU index (0 to num_gpus-1)
  CUDA_VISIBLE_DEVICES=$gpu_id python -W ignore inference.py --save_dir $savepath --checkpoint $suprem_path --data_root_path $datapath --data_txt_path $savetemptxtpath --num_class 5 --map_type jhh --backbone $arch --dataset_list $dataset_file --a_min -100 --a_max 200 --saveprobabilities --stage test &
  counter=$((counter + 1))
  sleep 1 # Delay to stagger the start times of the processes to manage resource allocation better
done

wait # Wait for all background processes to finish

rm -r $savetemptxtpath
