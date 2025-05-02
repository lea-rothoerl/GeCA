## Mammomat + Planmed findings​ (MA_PL_FI)
11 labels

### CSV Customization
python ../../GeCA/VinDr_Mammo_Preprocessing/customize_csv.py annotations.csv MA_PL_FI.csv --columns finding_categories model --conditions model=Mammomat\ Inspiration model=Planmed\ Nuance --label-column finding_categories

### Feature Extraction
CUDA_VISIBLE_DEVICES=0 nice -n 10 torchrun --nnodes=1 --master-port 29504 --nproc_per_node=1 extract_features.py --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/MA_PL_FI.csv --features-path /home/lea_urv/images/fullfield/features/MA_PL_FI --fold 5 --image-
size 128 --label-column finding_categories

### Model Training
CUDA_VISIBLE_DEVICES=0,1 nice -n 10 accelerate launch --main_process_port $(shuf -i 30000-35000 -n 1) --multi-gpu --num_processes 2 --mixed_precision fp16 train.py --model GeCA-S --feature-path /home/lea_urv/images/fullfield/features/MA_PL_FI/ --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/MA_PL_FI.csv --label-column finding_categories --results-dir /home/lea_urv/images/fullfield/weights/MA_PL_FI/ --num-classes 11 --epochs 2500 --global-batch-size 32 --fold 5 --validate_every 50 --image-size 128 --num-workers 2

### Sampling


### Evaluation

