# Single pair (from Python)
from pathlib import Path
from cfsd import cfsd_from_paths
score = cfsd_from_paths(Path("./eval_dirs/generated/vase.jpg"), Path("./eval_dirs/reference/starry_night_full.jpg"), size=512, use_vgg19=True, tau=1.0, stride=2)
print("CFSD =", score)

# # Batch over folders (matching filenames)
# python /mnt/data/evaluate_cfsd.py \
#   --content_dir /path/to/COCO_subset \
#   --stylized_dir /path/to/your_outputs \
#   --size 512 --use_vgg19 --tau 1.0 --stride 2 --row_chunk 512 \
#   --out_csv cfsd_results.csv
