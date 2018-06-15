# image-captioning
Implementations for image captioning models in PyTorch, currently only supports pretrained ResNet152 and VGG16 with batch normalization.
Model without attention is implemented from ["show and tell"](https://arxiv.org/pdf/1411.4555.pdf), 
while the model with attention is from ["show, attend and tell"](https://arxiv.org/pdf/1502.03044.pdf)

Evaluate captions via `capeval/`, which is derived from [tylin/coco-caption](https://github.com/tylin/coco-caption) with minor changes for a better Python 3 support

## Requirements
- MSCOCO original dataset, please put them in the same directory, e.g. `COCO2014/`, and modify the `COCO_ROOT` in `configs.py`, you can get them here: 
    - [train2014 images](http://images.cocodataset.org/zips/train2014.zip)
    - [val2014 images](http://images.cocodataset.org/zips/val2014.zip)
- Instead of using random split, [Karpathy's split](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) is required, please put it in the `COCO_PATH`
- PyTorch v0.3.1 or newer with GPU support.

## Usage
### 1. Preprocessing
First of all we should preprocess the images and store them locally. Specifying phases is available if parallel processing is required.
All preprocessed images are stored in HDF5 databases in `COCO_ROOT`
```bash
python preprocess.py
```

### 2. Extract image features
Extract the image features offline by the encoder and store them locally. 
Currently only ResNet152 and VGG16 with batch normalization are supported.
```bash
python extract.py --pretrained=resnet --batch_size=10 --gpu=0
```

### 3. Training the model
Training can be performed only after the image features are extracted. 
If training on the full dataset is desired, please specify the `train_size` as `-1`
Immediate evaluation with beam search after training is also available, please set the flag as `true`. 
The scores are stored in `scores/`
```bash
python train.py --train_size=100 --val_size=10 --test=10 --epoch=30 --verbose=10 --learning_rate=1e-3 --batch_size=10 --gpu=0 --pretrained=resnet --attention=false --evaluation=true
```

### 4. Offline evaluation
After the training is over, an offline evaluation can be performed.
All generated captions are stored in `results/`
```bash
python evaluation.py --train_size=100 --test_size=10 --num=3 --batch_size=10 --gpu=10 --pretrained=resnet --attention=false --encoder=<path_to_encoder> --decoder=<path_to_decoder>
```
Note that the `train_size` must match the size of images for training

### 5. Visualize attention weights
For the model with attention.

## Results
### Good captions
![alt text](https://github.com/daveredrum/image-captioning/blob/master/demo/high.png)
### Okay captions
![alt text](https://github.com/daveredrum/image-captioning/blob/master/demo/medium.png)
### Bad captions
![alt text](https://github.com/daveredrum/image-captioning/blob/master/demo/low.png)
