predpath=$1
truthpath=$2
csvpath=error_analysis/$3.fold_$4/ID 
visualpath=error_analysis/$3.fold_$4/visual 
rocpath=error_analysis/$3.fold_$4/roc 

python -W ignore eval.py --predpath $predpath --truthpath $truthpath --savecsvpath $csvpath --postprocessing --multiprocessing --pdac_size 30 --cyst_size 40 --pnet_size 35 --savevisualpath $visualpath --FP --FN --plotroc --saverocpath $rocpath
# --savect
