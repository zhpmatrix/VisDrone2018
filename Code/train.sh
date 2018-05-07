BATCH_SIZE=2
WORKER_NUMBER=8
LEARNING_RATE=0.001
DECAY_STEP=100
EPOCH=20
python trainval_net.py \
                   --dataset pascal_voc --net vgg16  \
                   --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                   --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
                   --epochs $EPOCH --cuda --mGPUs 
