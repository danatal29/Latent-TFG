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
    --scale=5.0\
    --ddim_steps=500\
    --optim_forward_guidance\
    --style_image='../../pics/im3.jpg'\
    --optim_num_steps=6\
    --optim_forward_guidance_wt=100.0\
    --guidance_domain='image';