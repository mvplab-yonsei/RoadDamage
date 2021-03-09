# Detection for Road Damage Dataset

도로 상의 장애물 및 이상 상태 검출을 위한 네트워크: YOLO v5

## Environment
- Cython
- matplotlib>=3.2.2
- numpy>=1.18.5
- opencv-python>=4.1.2
- Pillow
- PyYAML>=5.3.1
- scipy>=1.4.1
- tensorboard>=2.2
- torch>=1.7.0

```bash
  $ pip install -r requirements.txt
```

## Datasets

1. Download databases

다운로드 링크는 기재 예정입니다.

2. Convert json to txt file. Make 'database' directory and place data as follows.
```bash
dataset

├── split

	├── train
  
		├── [file_name.jpg]
    ├── [file_name.txt]
        ...
    
	├── valid
  
		├── [file_name.jpg]
    ├── [file_name.txt]
        ...
```
## Train

```bash
    python train.py ~~~~~~~
```

## Pre-trained Models

1. Download our pre-trained models.

   기재 예정입니다.

2. Place pre-trained models in '/runs/train' directory.

## Evaluation

Run Evaluation.py to evaluate the performance of trained models with following commands.

```bash
    python train.py ~~~~~~~
```



