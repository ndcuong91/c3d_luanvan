pc_name = '300'  # 300, mica: for running # duycuong: for coding

compute_volume_mean = True
finetuning = True
feature_extract = True
num_action=12

# optional
base_lr = 0.0001
gamma = 0.1
data_type_train = 'original_pre_3'
data_type_test = 'original_pre_3'
batch_size_test = 20
batch_size_finetune = 20
center_crop=False
pretrained_model='c3d_ucf101_finetune_whole_iter_20000_fc6_1024_77.26%'
#pretrained_model='c3d_ucf101_finetune_whole_iter_20000_fc6_1024_fc7_1024'

if (num_action==12):
    subject_list = 'Giang,Hai,Long,Minh,Thuy,Tuyen'
    subject_test = 'Giang,Hai,Long,Minh,Thuy,Tuyen'
    Kinects = 'K1,K2,K3,K4,K5'
    actions=[1,2,3,4,5,6,7,8,9,10,11,12]

if (num_action==5):
    subject_list = 'Binh,Giang,Hung,Tan,Thuan'
    subject_test = 'Binh,Giang,Hung,Tan,Thuan'
    Kinects = 'K1,K3,K5'
    actions=[1,2,3,4,5]

# default is mica
output_dir = "output"
template_dir = "template"
c3d_files_dir = "c3d_files"

if (num_action==12):
    c3d_data_root = "data/12gestures_images_segmented"
if (num_action == 5):
    c3d_data_root = "data"

if(center_crop==True):
    tool_dir='C3D_code/C3D-v1.0/build_center_crop/tools'
else: #random crop in training
    tool_dir='C3D_code/C3D-v1.0/build/tools'
#tool_dir = '/home/dangmanhtruong95/C3D-master_JPG/build_CuongND/tools'

if (pc_name == 'duycuong'):
    compute_volume_mean = False
    finetuning = False
    feature_extract = False
