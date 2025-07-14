wandb login *********
deepspeed --include localhost:0,1,2,3,4,5,6,7 ./trainer_base_ds_mul_fs_tp.py -cp conf/exp/ -cn qwq_glide_8gpu_slim6b