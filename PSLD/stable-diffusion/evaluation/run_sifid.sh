# Single image pair
python sifid.py --ref ./eval_dirs/reference/starry_night_full.jpg --gen ./eval_dirs/generated/vanilla.png --device mps --K 100 --multiscale 3

#python sifid.py --ref ./eval_dirs/reference/starry_night_full.jpg --gen ./eval_dirs/reference/starry_night_full.jpg --device mps --K 60 --multiscale 1


# Two folders (matched filenames)
#python style_sifid_gram_mps.py --ref style_refs/ --gen outputs_both/ --device mps --K 120 --multiscale 3
