python train.py \
--train_size=-1 \
--val_size=-1 \
--test=-1 \
--epoch=20 \
--verbose=200 \
--learning_rate=5e-4 \
--weight_decay=1e-4 \
--batch_size=256 \
--gpu=2 \
--pretrained=resnet \
--attention=adaptive \
--evaluation=true