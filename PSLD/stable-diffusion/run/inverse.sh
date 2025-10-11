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
    --omega=10\
    --ddim_steps=100;


python scripts/inverse.py \
    --file_id='im1.jpg' \
    --task_config='configs/style_extraction_config.yaml' \
    --outdir='outputs/psld-samples-clip_exp3'\
    --prompt='A tropic island with a volcano'\
    --ddim_eta=0.5\
    --omega=10\
    --scale=5\
    --ddim_steps=100;

# UGD-enhanced style transfer example
python scripts/inverse_ugd.py \
    --file_id='im2.jpg' \
    --task_config='configs/style_extraction_config.yaml' \
    --outdir='outputs/ugd-style-samples'\
    --prompt='A tropic island with a volcano'\
    --ddim_eta=0.5\
    --omega=10\
    --scale=7.5\
    --ddim_steps=100\
    --optim_forward_guidance\
    --style_image='../../pics/im2.jpg'\
    --optim_num_steps=6\
    --optim_forward_guidance_wt=50.0\
    --guidance_domain='image';


# UGD-enhanced style transfer example
python scripts/inverse_ugd.py \
    --file_id='im2.jpg' \
    --task_config='configs/style_extraction_config.yaml' \
    --outdir='outputs/ugd-style-samples'\
    --prompt='A tropic island with a volcano'\
    --ddim_eta=0.5\
    --omega=10\
    --scale=7.5\
    --ddim_steps=100\
    --optim_forward_guidance\
    --style_image='../../pics/im2.jpg'\
    --optim_num_steps=6\
    --optim_forward_guidance_wt=50.0\
    --guidance_domain='image';


# picasso version
python scripts/inverse_ugd.py \
    --file_id='cubism_picasso_faces.jpg' \
    --task_config='configs/style_extraction_config.yaml' \
    --outdir='outputs/ugd-style-samples'\
    --prompt='A tropic island with a volcano'\
    --ddim_eta=0.5\
    --omega=10\
    --scale=7.5\
    --ddim_steps=100\
    --optim_forward_guidance\
    --style_image='../../pics/cubism_picasso_faces.jpg'\
    --optim_num_steps=10\
    --optim_forward_guidance_wt=100.0\
    --guidance_domain='image';




# picasso version
python scripts/inverse_ugd.py \
    --file_id='im3.jpg' \
    --task_config='configs/style_extraction_config.yaml' \
    --outdir='outputs/ugd-style-samples'\
    --prompt='A dog playing in the park'\
    --ddim_eta=0.5\
    --omega=10\
    --k_recur=6\
    --normalize_grad\
    --scale=6.0\
    --ddim_steps=100\
    --optim_forward_guidance\
    --style_image='../../pics/im3.jpg'\
    --optim_num_steps=3\
    --optim_forward_guidance_wt=15.0\
    --guidance_domain='image';


# painting version
python scripts/inverse_ugd.py \
    --file_id='buzz.jpg' \
    --task_config='configs/style_extraction_config.yaml' \
    --outdir='outputs/ugd-style-samples'\
    --prompt='A happy dog playing in the park'\
    --ddim_eta=0.5\
    --omega=10\
    --scale=3.0\
    --ddim_steps=100\
    --optim_forward_guidance\
    --style_image='../../pics/buzz.jpg'\
    --optim_num_steps=10\
    --optim_forward_guidance_wt=100.0\
    --guidance_domain='image';





python scripts/inverse.py \
    --file_id='im3.jpg' \
    --task_config='configs/style_extraction_config.yaml' \
    --outdir='outputs/psld-samples-clip_exp4'\
    --prompt='A tropic island with a volcano'\
    --ddim_eta=0.5\
    --omega=20\
    --scale=5\
    --ddim_steps=50;




python scripts/inverse.py \
    --file_id='cubism_picasso_faces.jpg' \
    --task_config='configs/style_extraction_config.yaml' \
    --outdir='outputs/psld-samples-clip_exp4'\
    --prompt='A happy dog playing in the park'\
    --ddim_eta=0.5\
    --omega=10\
    --scale=7.5\
    --ddim_steps=150;






python scripts/inverse_ugd.py \
    --file_id='00014.png' \
    --task_config='configs/super_resolution_config_psld.yaml' \
    --outdir='outputs/psld-samples-sr' \
    --prompt='happy dogs' \
    --ddim_steps=50






#'A tropic island with a volcano'
