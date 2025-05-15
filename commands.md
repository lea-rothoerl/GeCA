## Mammomat + Planmed findings​ (MA_PL_FI)
11 labels

### CSV Customization
python ../../GeCA/VinDr_Mammo_Preprocessing/customize_csv.py annotations.csv MA_PL_FI.csv --columns finding_categories model --conditions model=Mammomat\ Inspiration model=Planmed\ Nuance --label-column finding_categories

### Feature Extraction
CUDA_VISIBLE_DEVICES=0 nice -n 10 torchrun --nnodes=1 --master-port 29504 --nproc_per_node=1 extract_features.py --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/MA_PL_FI.csv --features-path /home/lea_urv/images/fullfield/features/MA_PL_FI --fold 5 --image-size 128 --label-column finding_categories

### Model Training
CUDA_VISIBLE_DEVICES=0,1 nice -n 10 accelerate launch --main_process_port $(shuf -i 30000-35000 -n 1) --multi-gpu --num_processes 2 --mixed_precision fp16 train.py --model GeCA-S --feature-path /home/lea_urv/images/fullfield/features/MA_PL_FI/ --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/MA_PL_FI.csv --label-column finding_categories --results-dir /home/lea_urv/images/fullfield/weights/MA_PL_FI/ --num-classes 11 --epochs 1000 --global-batch-size 32 --fold 5 --validate_every 50 --image-size 128 --num-workers 2

### Sampling
CUDA_VISIBLE_DEVICES=0 nice -n 10 torchrun --master-port $(shuf -i 30000-35000 -n 1) --nnodes=1 --nproc_per_node=1 sample_ddp_val.py --expand_ratio 1 --model GeCA-S --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/MA_PL_FI.csv --fold 5 --num-sampling-steps 250 --ckpt /home/lea_urv/images/fullfield/weights/MA_PL_FI/000-GeCA-S-5/checkpoints/best_ckpt.pt --sample-dir /home/lea_urv/images/fullfield/synthetic/MA_PL_FI/ --num-classes 11 --image-size 128 --label-column finding_categories

### Evaluation
python evaluate.py --fold 5 --image-size 128 --device_list cuda:0 --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/MA_PL_FI.csv --gen /home/lea_urv/images/fullfield/synthetic/MA_PL_FI/GeCA-S-GS-fold-5-nstep-250-best_ckpt-size-128-vae-ema-cfg-1.5-seed-0/ --label-column finding_categories

--------

## Planmed findings​ (PL_FI)
10 labels

### CSV Customization
python ../../GeCA/VinDr_Mammo_Preprocessing/customize_csv.py annotations.csv PL_FI.csv --columns finding_categories model --conditions model=Planmed\ Nuance --label-column finding_categories

### Feature Extraction
CUDA_VISIBLE_DEVICES=0 nice -n 10 torchrun --nnodes=1 --master-port 29504 --nproc_per_node=1 extract_features.py --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/PL_FI.csv --features-path /home/lea_urv/images/fullfield/features/PL_FI --fold 5 --image-size 128 --label-column finding_categories

### Model Training
CUDA_VISIBLE_DEVICES=0,1 nice -n 10 accelerate launch --main_process_port $(shuf -i 30000-35000 -n 1) --multi-gpu --num_processes 2 --mixed_precision fp16 train.py --model GeCA-S --feature-path /home/lea_urv/images/fullfield/features/PL_FI/ --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/PL_FI.csv --label-column finding_categories --results-dir /home/lea_urv/images/fullfield/weights/PL_FI/ --num-classes 10 --epochs 2500 --global-batch-size 32 --fold 5 --validate_every 50 --image-size 128 --num-workers 2

ran till the end

### Sampling
CUDA_VISIBLE_DEVICES=1 nice -n 10 torchrun --master-port $(shuf -i 30000-35000 -n 1) --nnodes=1 --nproc_per_node=1 sample_ddp_val.py --expand_ratio 1 --model GeCA-S --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/PL_FI.csv --fold 5 --num-sampling-steps 250 --ckpt /home/lea_urv/images/fullfield/weights/PL_FI/000-GeCA-S-5/checkpoints/best_ckpt.pt --sample-dir /home/lea_urv/images/fullfield/synthetic/PL_FI/ --num-classes 10 --image-size 128 --label-column finding_categories

### Evaluation
python evaluate.py --fold 5 --image-size 128 --device_list cuda:0 --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/PL_FI.csv --gen /home/lea_urv/images/fullfield/synthetic/PL_FI/GeCA-S-GS-fold-5-nstep-250-best_ckpt-size-128-vae-ema-cfg-1.5-seed-0/ --label-column finding_categories

--------

## Mammomat density (MA_DE)
4 labels

### CSV Customization
python ../../GeCA/VinDr_Mammo_Preprocessing/customize_csv.py annotations.csv MA_DE.csv --columns breast_density model --conditions model=Mammomat\ Inspiration --label-column breast_density

### Feature Extraction
CUDA_VISIBLE_DEVICES=0 nice -n 10 torchrun --nnodes=1 --master-port 29504 --nproc_per_node=1 extract_features.py --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/MA_DE.csv --features-path /home/lea_urv/images/fullfield/features/MA_DE --fold 5 --image-size 128 --label-column breast_density

### Model Training
CUDA_VISIBLE_DEVICES=0,1 nice -n 10 accelerate launch --main_process_port $(shuf -i 30000-35000 -n 1) --multi-gpu --num_processes 2 --mixed_precision fp16 train.py --model GeCA-S --feature-path /home/lea_urv/images/fullfield/features/MA_DE/ --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/MA_DE.csv --label-column breast_density --results-dir /home/lea_urv/images/fullfield/weights/MA_DE/ --num-classes 4 --epochs 1000 --global-batch-size 32 --fold 5 --validate_every 50 --image-size 128 --num-workers 2

### Sampling
CUDA_VISIBLE_DEVICES=1 nice -n 10 torchrun --master-port $(shuf -i 30000-35000 -n 1) --nnodes=1 --nproc_per_node=1 sample_ddp_val.py --expand_ratio 1 --model GeCA-S --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/MA_DE.csv --fold 5 --num-sampling-steps 250 --ckpt /home/lea_urv/images/fullfield/weights/MA_DE/000-GeCA-S-5/checkpoints/best_ckpt.pt --sample-dir /home/lea_urv/images/fullfield/synthetic/MA_DE/ --num-classes 4 --label-column breast_density --image-size 128

### Evaluation
python evaluate.py --fold 5 --image-size 128 --device_list cuda:0 --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/MA_DE.csv --gen /home/lea_urv/images/fullfield/synthetic/MA_DE/GeCA-S-GS-fold-5-nstep-250-best_ckpt-size-128-vae-ema-cfg-1.5-seed-0/ --label-column breast_density

### Classifier
#### Big Sampling 
CUDA_VISIBLE_DEVICES=1 nice -n 10 torchrun --master-port $(shuf -i 30000-35000 -n 1) --nnodes=1 --nproc_per_node=1 sample_ddp_val.py --expand_ratio 1 --model GeCA-S --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/MA_DE.csv --fold 5 --num-sampling-steps 250 --ckpt /home/lea_urv/images/fullfield/weights/MA_DE/000-GeCA-S-5/checkpoints/best_ckpt.pt --sample-dir /home/lea_urv/images/fullfield/synthetic/MA_DE_Syn/ --num-classes 4 --label-column breast_density --image-size 128 --expand_ratio 13

#### Targeted Sampling
CUDA_VISIBLE_DEVICES=0 nice -n 10 torchrun --master-port $(shuf -i 30000-35000 -n 1) --nnodes=1 --nproc_per_node=1 sample_ddp_val.py --expand_ratio 1 --model GeCA-S --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/MA_DE.csv --fold 5 --num-sampling-steps 250 --ckpt /home/lea_urv/images/fullfield/weights/MA_DE/000-GeCA-S-5/checkpoints/best_ckpt.pt --sample-dir /home/lea_urv/images/fullfield/synthetic/MA_DE_TargSyn/ --num-classes 4 --label-column breast_density --image-size 128 --select-labels DENSITY\ A DENSITY\ B DENSITY\ D --samples-per-label 10000

#### Mixed CSV Creation
python /home/lea_urv/GeCA/build_mixed_csv.py --reference-csv /home/lea_urv/images/fullfield/MA_DE.csv --synth-annotation-csv /home/lea_urv/images/fullfield/synthetic/MA_DE_Syn/GeCA-S-GS-fold-5-nstep-250-best_ckpt-size-128-vae-ema-cfg-1.5-seed-0/val_syn_5.csv --output-csv /home/lea_urv/images/fullfield/MA_DE_Syn.csv 

#### Targeted Mixed CSV Creation

#### Reference
python /home/lea_urv/GeCA/classify_cnn.py --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/MA_DE.csv --label-column breast_density --epochs 25

#### Test
python /home/lea_urv/GeCA/classify_cnn.py --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/MA_DE_Syn.csv --label-column breast_density --epochs 25


--------

## Planmed density (PL_DE)
4 labels

### CSV Customization
python ../../GeCA/VinDr_Mammo_Preprocessing/customize_csv.py annotations.csv PL_DE.csv --columns breast_density model --conditions model=Planmed\ Nuance --label-column breast_density

### Feature Extraction
CUDA_VISIBLE_DEVICES=0 nice -n 10 torchrun --nnodes=1 --master-port 29504 --nproc_per_node=1 extract_features.py --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/PL_DE.csv --features-path /home/lea_urv/images/fullfield/features/PL_DE --fold 5 --image-size 128 --label-column breast_density

### Model Training
CUDA_VISIBLE_DEVICES=0,1 nice -n 10 accelerate launch --main_process_port $(shuf -i 30000-35000 -n 1) --multi-gpu --num_processes 2 --mixed_precision fp16 train.py --model GeCA-S --feature-path /home/lea_urv/images/fullfield/features/PL_DE/ --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/PL_DE.csv --label-column breast_density --results-dir /home/lea_urv/images/fullfield/weights/PL_DE/ --num-classes 4 --epochs 2500 --global-batch-size 32 --fold 5 --validate_every 50 --image-size 128 --num-workers 2

ran til the end

### Sampling
CUDA_VISIBLE_DEVICES=1 nice -n 10 torchrun --master-port $(shuf -i 30000-35000 -n 1) --nnodes=1 --nproc_per_node=1 sample_ddp_val.py --expand_ratio 1 --model GeCA-S --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/PL_DE.csv --fold 5 --num-sampling-steps 250 --ckpt /home/lea_urv/images/fullfield/weights/PL_DE/000-GeCA-S-5/checkpoints/best_ckpt.pt --sample-dir /home/lea_urv/images/fullfield/synthetic/PL_DE/ --num-classes 4 --label-column breast_density --image-size 128

### Evaluation
python evaluate.py --fold 5 --image-size 128 --device_list cuda:0 --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/PL_DE.csv --gen /home/lea_urv/images/fullfield/synthetic/PL_DE/GeCA-S-GS-fold-5-nstep-250-best_ckpt-size-128-vae-ema-cfg-1.5-seed-0/ --label-column breast_density

### Classifier
#### Big Sampling 
CUDA_VISIBLE_DEVICES=1 nice -n 10 torchrun --master-port $(shuf -i 30000-35000 -n 1) --nnodes=1 --nproc_per_node=1 sample_ddp_val.py --expand_ratio 1 --model GeCA-S --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/PL_DE.csv --fold 5 --num-sampling-steps 250 --ckpt /home/lea_urv/images/fullfield/weights/PL_DE/000-GeCA-S-5/checkpoints/best_ckpt.pt --sample-dir /home/lea_urv/images/fullfield/synthetic/PL_DE_Syn/ --num-classes 4 --label-column breast_density --image-size 128 --expand_ratio 14

#### Targeted Sampling
CUDA_VISIBLE_DEVICES=0 nice -n 10 torchrun --master-port $(shuf -i 30000-35000 -n 1) --nnodes=1 --nproc_per_node=1 sample_ddp_val.py --expand_ratio 1 --model GeCA-S --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/PL_DE.csv --fold 5 --num-sampling-steps 250 --ckpt /home/lea_urv/images/fullfield/weights/PL_DE/000-GeCA-S-5/checkpoints/best_ckpt.pt --sample-dir /home/lea_urv/images/fullfield/synthetic/PL_DE_TargSyn/ --num-classes 4 --label-column breast_density --image-size 128 --select-labels DENSITY\ A DENSITY\ B DENSITY\ D --samples-per-label 2000

#### Mixed CSV Creation
python /home/lea_urv/GeCA/build_mixed_csv.py --reference-csv /home/lea_urv/images/fullfield/PL_DE.csv --synth-annotation-csv /home/lea_urv/images/fullfield/synthetic/PL_DE_Syn/GeCA-S-GS-fold-5-nstep-250-best_ckpt-size-128-vae-ema-cfg-1.5-seed-0/syn_per_label_5.csv --output-csv /home/lea_urv/images/fullfield/PL_DE_Syn.csv --ratio 0.5

python /home/lea_urv/GeCA/build_mixed_csv.py --reference-csv /home/lea_urv/images/fullfield/PL_DE.csv --synth-annotation-csv /home/lea_urv/images/fullfield/synthetic/PL_DE_TargSyn/GeCA-S-GS-fold-5-nstep-250-best_ckpt-size-128-vae-ema-cfg-1.5-seed-0/val_syn_5.csv --output-csv /home/lea_urv/images/fullfield/PL_DE_TargSyn.csv

#### Reference
python /home/lea_urv/GeCA/classify_cnn.py --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/PL_DE.csv --label-column breast_density --epochs 25

#### Test
python /home/lea_urv/GeCA/classify_cnn.py --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/PL_DE_Syn.csv --label-column breast_density --epochs 25

python /home/lea_urv/GeCA/classify_cnn.py --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/PL_DE_TargSyn.csv --label-column breast_density --epochs 25

--------

## Mammomat + Planmed density (MA_PL_DE)
4 labels

### CSV Customization
python ../../GeCA/VinDr_Mammo_Preprocessing/customize_csv.py annotations.csv MA_PL_DE.csv --columns breast_density model --conditions model=Mammomat\ Inspiration --model=Planmed\ Nuance --label-column breast_density

### Feature Extraction
CUDA_VISIBLE_DEVICES=0 nice -n 10 torchrun --nnodes=1 --master-port 29504 --nproc_per_node=1 extract_features.py --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/MA_PL_DE.csv --features-path /home/lea_urv/images/fullfield/features/MA_PL_DE --fold 5 --image-size 128 --label-column breast_density

### Model Training
CUDA_VISIBLE_DEVICES=0,1 nice -n 10 accelerate launch --main_process_port $(shuf -i 30000-35000 -n 1) --multi-gpu --num_processes 2 --mixed_precision fp16 train.py --model GeCA-S --feature-path /home/lea_urv/images/fullfield/features/MA_PL_DE/ --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/MA_PL_DE.csv --label-column breast_density --results-dir /home/lea_urv/images/fullfield/weights/MA_PL_DE/ --num-classes 4 --epochs 1000 --global-batch-size 32 --fold 5 --validate_every 50 --image-size 128 --num-workers 2

### Sampling
CUDA_VISIBLE_DEVICES=1 nice -n 10 torchrun --master-port $(shuf -i 30000-35000 -n 1) --nnodes=1 --nproc_per_node=1 sample_ddp_val.py --expand_ratio 1 --model GeCA-S --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/MA_PL_DE.csv --fold 5 --num-sampling-steps 250 --ckpt /home/lea_urv/images/fullfield/weights/MA_PL_DE/000-GeCA-S-5/checkpoints/best_ckpt.pt --sample-dir /home/lea_urv/images/fullfield/synthetic/MA_PL_DE/ --num-classes 4 --label-column breast_density --image-size 128

### Evaluation
python evaluate.py --fold 5 --image-size 128 --device_list cuda:0 --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/MA_PL_DE.csv --gen /home/lea_urv/images/fullfield/synthetic/MA_PL_DE/GeCA-S-GS-fold-5-nstep-250-best_ckpt-size-128-vae-ema-cfg-1.5-seed-0/ --label-column breast_density

### Classifier
#### Big Sampling 
CUDA_VISIBLE_DEVICES=1 nice -n 10 torchrun --master-port $(shuf -i 30000-35000 -n 1) --nnodes=1 --nproc_per_node=1 sample_ddp_val.py --expand_ratio 1 --model GeCA-S --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/MA_PL_DE.csv --fold 5 --num-sampling-steps 250 --ckpt /home/lea_urv/images/fullfield/weights/MA_PL_DE/000-GeCA-S-5/checkpoints/best_ckpt.pt --sample-dir /home/lea_urv/images/fullfield/synthetic/MA_PL_DE_Syn/ --num-classes 4 --label-column breast_density --image-size 128 --expand_ratio 13

#### Targeted Sampling
CUDA_VISIBLE_DEVICES=0 nice -n 10 torchrun --master-port $(shuf -i 30000-35000 -n 1) --nnodes=1 --nproc_per_node=1 sample_ddp_val.py --expand_ratio 1 --model GeCA-S --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/MA_PL_DE.csv --fold 5 --num-sampling-steps 250 --ckpt /home/lea_urv/images/fullfield/weights/MA_PL_DE/000-GeCA-S-5/checkpoints/best_ckpt.pt --sample-dir /home/lea_urv/images/fullfield/synthetic/MA_PL_DE_TargSyn/ --num-classes 4 --label-column breast_density --image-size 128 --select-labels DENSITY\ A DENSITY\ B DENSITY\ D --samples-per-label 10000

#### Mixed CSV Creation
python /home/lea_urv/GeCA/build_mixed_csv.py --reference-csv /home/lea_urv/images/fullfield/MA_PL_DE.csv --synth-annotation-csv /home/lea_urv/images/fullfield/synthetic/MA_PL_DE_Syn/GeCA-S-GS-fold-5-nstep-250-best_ckpt-size-128-vae-ema-cfg-1.5-seed-0/val_syn_5.csv --output-csv /home/lea_urv/images/fullfield/MA_PL_DE_Syn.csv 

#### Reference
python /home/lea_urv/GeCA/classify_cnn.py --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/MA_PL_DE.csv --label-column breast_density --epochs 25

#### Test
python /home/lea_urv/GeCA/classify_cnn.py --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/MA_PL_DE_Syn.csv --label-column breast_density --epochs 25

--------

## Mammomat findings (MA_FI)
11 labels

### CSV Customization
python ../../GeCA/VinDr_Mammo_Preprocessing/customize_csv.py annotations.csv MA_FI.csv --columns finding_categories model --conditions model=Mammomat\ Inspiration --label-column finding_categories

### Feature Extraction
CUDA_VISIBLE_DEVICES=0 nice -n 10 torchrun --nnodes=1 --master-port 29504 --nproc_per_node=1 extract_features.py --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/MA_FI.csv --features-path /home/lea_urv/images/fullfield/features/MA_FI --fold 5 --image-size 128 --label-column finding_categories

### Model Training
CUDA_VISIBLE_DEVICES=0,1 nice -n 10 accelerate launch --main_process_port $(shuf -i 30000-35000 -n 1) --multi-gpu --num_processes 2 --mixed_precision fp16 train.py --model GeCA-S --feature-path /home/lea_urv/images/fullfield/features/MA_FI/ --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/MA_FI.csv --label-column finding_categories --results-dir /home/lea_urv/images/fullfield/weights/MA_FI/ --num-classes 11 --epochs 1000 --global-batch-size 32 --fold 5 --validate_every 50 --image-size 128 --num-workers 2

### Sampling
CUDA_VISIBLE_DEVICES=1 nice -n 10 torchrun --master-port $(shuf -i 30000-35000 -n 1) --nnodes=1 --nproc_per_node=1 sample_ddp_val.py --expand_ratio 1 --model GeCA-S --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/MA_FI.csv --fold 5 --num-sampling-steps 250 --ckpt /home/lea_urv/images/fullfield/weights/MA_FI/000-GeCA-S-5/checkpoints/best_ckpt.pt --sample-dir /home/lea_urv/images/fullfield/synthetic/MA_FI/ --num-classes 11 --label-column finding_categories --image-size 128

### Evaluation
python evaluate.py --fold 5 --image-size 128 --device_list cuda:0 --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/MA_FI.csv --gen /home/lea_urv/images/fullfield/synthetic/MA_FI/GeCA-S-GS-fold-5-nstep-250-best_ckpt-size-128-vae-ema-cfg-1.5-seed-0/ --label-column finding_categories

--------

## Masses model (MAS_MO)
4 labels

### CSV Customization
python ../../GeCA/VinDr_Mammo_Preprocessing/customize_csv.py annotations.csv MAS_MO.csv --columns finding_categories model --conditions finding_categories=Mass --label-column model

### Feature Extraction
CUDA_VISIBLE_DEVICES=0 nice -n 10 torchrun --nnodes=1 --master-port 29504 --nproc_per_node=1 extract_features.py --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/MAS_MO.csv --features-path /home/lea_urv/images/fullfield/features/MAS_MO --fold 5 --image-size 128 --label-column model

### Model Training
CUDA_VISIBLE_DEVICES=0,1 nice -n 10 accelerate launch --main_process_port $(shuf -i 30000-35000 -n 1) --multi-gpu --num_processes 2 --mixed_precision fp16 train.py --model GeCA-S --feature-path /home/lea_urv/images/fullfield/features/MAS_MO/ --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/MAS_MO.csv --label-column model --results-dir /home/lea_urv/images/fullfield/weights/MAS_MO/ --num-classes 4 --epochs 2500 --global-batch-size 32 --fold 5 --validate_every 50 --image-size 128 --num-workers 2

### Sampling
CUDA_VISIBLE_DEVICES=1 nice -n 10 torchrun --master-port $(shuf -i 30000-35000 -n 1) --nnodes=1 --nproc_per_node=1 sample_ddp_val.py --expand_ratio 1 --model GeCA-S --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/MAS_MO.csv --fold 5 --num-sampling-steps 250 --ckpt /home/lea_urv/images/fullfield/weights/MAS_MO/000-GeCA-S-5/checkpoints/best_ckpt.pt --sample-dir /home/lea_urv/images/fullfield/synthetic/MAS_MO/ --num-classes 4 --label-column model --image-size 128

### Evaluation
python evaluate.py --fold 5 --image-size 128 --device_list cuda:0 --image-root /home/lea_urv/images/fullfield/png/ --annotation-path /home/lea_urv/images/fullfield/MAS_MO.csv --gen /home/lea_urv/images/fullfield/synthetic/MAS_MO/GeCA-S-GS-fold-5-nstep-250-best_ckpt-size-128-vae-ema-cfg-1.5-seed-0/ --label-column model





