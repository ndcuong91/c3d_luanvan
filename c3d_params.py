



pc_name='japan' # japan, 300, mica: for running # duycuong: for coding

compute_volume_mean=True
finetuning=True
feature_extract=True

#default is mica
output_dir = "/home/dangmanhtruong95/Cuong/c3d_luanvan/output"
template_dir = "/home/dangmanhtruong95/Cuong/c3d_luanvan/template"
c3d_data_root = "/media/data2/users/dangmanhtruong95/Cuong_data"
c3d_files_dir = "/home/dangmanhtruong95/Cuong/c3d_luanvan/c3d_files"
tool_dir='/home/dangmanhtruong95/C3D-master_JPG/build_CuongND/tools'

if(pc_name=='japan'):
    output_dir = "/home/prdcv/PycharmProjects/c3d_luanvan/output"
    template_dir = "/home/prdcv/PycharmProjects/c3d_luanvan/template"
    c3d_data_root = "/home/prdcv/PycharmProjects/c3d_luanvan/data"
    c3d_files_dir = "/home/prdcv/PycharmProjects/c3d_luanvan/c3d_files"
    tool_dir='/home/prdcv/PycharmProjects/c3d_luanvan/C3D_sourcecode/C3D-v1.0/build/tools'

if(pc_name=='duycuong'):
    compute_volume_mean=False
    finetuning=False
    feature_extract=False
    output_dir = "/home/titikid/PycharmProjects/c3d_luanvan/output"
    template_dir = "/home/titikid/PycharmProjects/c3d_luanvan/template"
    c3d_data_root = "/home/titikid/PycharmProjects/c3d_luanvan/data"
    c3d_files_dir = "/home/titikid/PycharmProjects/c3d_luanvan/c3d_files"

if(pc_name=='300'):
    output_dir = "/home/titikid/PycharmProjects/c3d_luanvan/output"
    template_dir = "/home/titikid/PycharmProjects/c3d_luanvan/template"
    c3d_data_root = "/home/titikid/PycharmProjects/c3d_luanvan/data"
    c3d_files_dir = "/home/titikid/PycharmProjects/c3d_luanvan/c3d_files"
    tool_dir='...tools'
