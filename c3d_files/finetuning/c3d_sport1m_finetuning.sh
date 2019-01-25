mkdir -p LOG_TRAIN
GLOG_log_dir="./LOG_TRAIN/"
GLOG_logtostderr=1 /home/prdcv/PycharmProjects/c3d_luanvan/C3D_sourcecode/C3D-v1.0/build/tools/finetune_net.bin /home/prdcv/PycharmProjects/c3d_luanvan/c3d_files/finetuning/c3d_sport1m_finetuning_solver.prototxt /home/prdcv/PycharmProjects/c3d_luanvan/c3d_files/pretrained_model_and_volume_mean/conv3d_deepnetA_sport1m_iter_1900000

