export CUDA_VISIBLE_DEVICES='0'
# python scripts/inverse.py \
#     --file_id='00015.png' \
#     --task_config='configs/super_resolution_config_psld.yaml' \
#     --outdir='outputs/psld-samples-sr';


# python scripts/inverse.py \
#     --file_id='00015.png' \
#     --task_config='configs/super_resolution_config_psld.yaml' \
#     --outdir='outputs/psld-samples-sr'\
#     --prompt='happy dog'\
#     --ddim_steps=50;


python scripts/inverse.py \
    --file_id='00014.png' \
    --task_config='configs/super_resolution_config_psld.yaml' \
    --outdir='outputs/psld-samples-sr'\
    --prompt='happy dogs'\
    --ddim_steps=50;


python scripts/inverse.py \
    --file_id='00014.png' \
    --task_config='configs/style_extraction_config.yaml' \
    --outdir='outputs/psld-samples-fr'\
    --prompt='happy dog'\
    --ddim_steps=100;


python scripts/inverse.py \
    --file_id='im1.jpg' \
    --task_config='configs/style_extraction_config.yaml' \
    --outdir='outputs/psld-samples-fr'\
    --prompt='happy dog'\
    --ddim_steps=1000;