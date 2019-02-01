#mkdir -p LOG_TRAIN
#GLOG_log_dir="./LOG_TRAIN/"
GLOG_logtostderr=1 %s %s %s 2>&1 | tee -a my_model.log
