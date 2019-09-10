from image_processing_utils import *

root_dir = "/mnt/4T/brain_imgs/"
save_dir = "/mnt/4T/brain_result/"
prepare_atlas_tissue = False
registration = False
Ants_script = "/home/silasi/ANTs/Scripts"
app_tran = True
write_summary = True
show = False
show_atlas = False
intro = False


main(root_dir=root_dir, save_dir=save_dir, prepare_atlas_tissue=prepare_atlas_tissue,
     registration=registration, Ants_script=Ants_script, app_tran=app_tran,
     write_summary=write_summary, show=show, show_atlas=show_atlas, intro=intro)