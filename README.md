# Datasets:
Publicly available datasets can be found as below: <br>
1. BUS: https://scholar.cu.edu.eg/Dataset_BUSI.zip <br>
2. Robotic Surgery: https://endovissub2018-roboticscenesegmentation.grand-challenge.org/Data/ <br>


# Commands
## Testing (Common for all models)

## IID (Vanilla)

```shell
python train_model.py \
    --data bus \
    --num-classes 2 \
    --batch-size 24 \
    --epochs 100 \
    --model-path outputs/bus/iid/unet.pth \
    --tb \
    --model-name UNet \
    --train-opt IID
```

## KD

```shell
python train_model.py \
    --train-opt KD \
    --teacher-path outputs/bus/iid/unet.pth \
    --model-path "outputs/bus/kd/kd_unet.pth" \
    --model-name UNet \
    --epochs 150 \
    --data bus \
    --batch-size 24 \
    --kld-loss-temp 4.5 \
    --tb
```

## PCD

```shell
python train_model.py \
    --data bus \
    --batch-size 24 \
    --epochs 150 \
    --model-path "outputs/bus/cd-tpl/cd_tpl_unet.pth" \
    --teacher-path outputs/bus/iid/unet.pth \
    --tb \
    --model-name UNet \
    --train-opt CD \
    --cd-mode tpl \
    --beta 0.7 \
    --gamma 0.8 \
    --kld-loss-temp 4.5 \
    --opt-t 2.51
```
