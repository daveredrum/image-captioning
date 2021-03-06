# image-captioning
Implementations for image captioning models in PyTorch, different types of attention mechanisms supported. Currently only provides pretrained ResNet152 and VGG16 with batch normalization as encoders.

Model supported: 
<br/>__FC__ from ["show and tell"](https://arxiv.org/pdf/1411.4555.pdf)
<br/>__Att2all__ from ["show and tell"](https://arxiv.org/pdf/1411.4555.pdf)
<br/>__Att2in__ from ["Self-critical Sequence Training for Image Captioning"](https://arxiv.org/pdf/1612.00563.pdf)
<br/>__Spatial attention__ from ["Knowing When to Look: Adaptive Attention via
A Visual Sentinel for Image Captioning"](https://arxiv.org/pdf/1411.4555.pdf)
<br/>__Adaptive attention__ from ["Knowing When to Look: Adaptive Attention via
A Visual Sentinel for Image Captioning"](https://arxiv.org/pdf/1411.4555.pdf)

Evaluate captions via `capeval/`, which is derived from [tylin/coco-caption](https://github.com/tylin/coco-caption) with minor changes for a better Python 3 support

## Requirements
- MSCOCO original dataset, please put them in the same directory, e.g. `COCO2014/`, and modify the `COCO_ROOT` in `configs.py`, you can get them here: 
    - [train2014 images](http://images.cocodataset.org/zips/train2014.zip)
    - [val2014 images](http://images.cocodataset.org/zips/val2014.zip)
- Instead of using random split, [Karpathy's split](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) is required, please put it in the `COCO_PATH`
- PyTorch v0.3.1 or newer with GPU support.
- TensorBoardX

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
python train.py --train_size=100 --val_size=10 --test=10 --epoch=30 --verbose=10 --learning_rate=1e-3 --batch_size=10 --gpu=0 --pretrained=resnet --attention=none --evaluation=true
```

### 4. Offline evaluation
After the training is over, an offline evaluation can be performed.
All generated captions are stored in `results/`
```bash
python evaluation.py --train_size=100 --test_size=10 --num=3 --batch_size=10 --gpu=10 --pretrained=resnet --attention=none --encoder=<path_to_encoder> --decoder=<path_to_decoder>
```
Note that the `train_size` must match the size of images for training

### 5. Visualize attention weights
For the model with attention.
```bash
python show_attention.py --phase=test --pretrained=resnet --train_size=-1 --val_size=-1 --test_size=-1 --num=10 --encoder=<path_to_encoder> --decoder=<path_to_decoder> --gpu=0
```

## Results
### Good captions
![alt text](https://github.com/daveredrum/image-captioning/blob/master/demo/high.png)
### Okay captions
![alt text](https://github.com/daveredrum/image-captioning/blob/master/demo/medium.png)
### Bad captions
![alt text](https://github.com/daveredrum/image-captioning/blob/master/demo/low.png)

## Attention
### Good results
![alt text](https://github.com/daveredrum/image-captioning/blob/master/demo/attention_good_1.png)
![alt text](https://github.com/daveredrum/image-captioning/blob/master/demo/attention_good_2.png)
![alt text](https://github.com/daveredrum/image-captioning/blob/master/demo/attention_good_3.png)
### Bad results
![alt text](https://github.com/daveredrum/image-captioning/blob/master/demo/attention_bad.png)

## Performance
|Model|BLEU-1|BLEU-2|BLEU-3|BLEU-4|CIDEr|
|---|---|---|---|---|---|
|Baseline (Nearest neighbor)|0.48|0.281|0.166|0.1|0.383|
|__FC__|__0.720__|__0.536__|__0.388__|__0.286__|__0.805__|
|__Att2in__|__0.732__|__0.553__|__0.402__|__0.296__|__0.837__|
|__Att2all__|__0.732__|__0.554__|__0.403__|__0.296__|__0.838__|
|__Spatial attention__|__0.725__|__0.537__|__0.389__|__0.287__|__0.812__|
|__Adaptive attention__|__0.716__|__0.524__|__0.379__|__0.278__|__0.808__|
|NeuralTalk2|0.625|0.45|0.321|0.23|0.66|
|Show and Tell|0.666|0.461|0.329|0.27|-|
|Show, Attend and Tell|0.707|0.492|0.344|0.243|-|
|Adaptive Attention|0.742|0.580|0.439|0.266|1.085|
|Neural Baby Talk|0.755|-|-|0.347|1.072|

> __best models:__
>
> |Model|train_size|test_size|learning_rate|weight_decay|batch_size|beam_size|dropout|
> |---|---|---|---|---|---|---|---|
> |__FC__|-1|-1|2e-4|0|512|7|0|
> |__Att2in__|-1|-1|5e-4|1e-4|256|7|0|
> |__Att2all__|-1|-1|5e-4|1e-4|256|7|0|
> |__Spatial attention__|-1|-1|2e-4|1e-4|256|7|0|
> |__Adaptive attention__|-1|-1|2e-4|1e-4|256|7|0|
