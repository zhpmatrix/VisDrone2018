rm data/cache/*
BATCH_SIZE=4
WORKER_NUMBER=8
LEARNING_RATE=0.01
DECAY_STEP=4
EPOCH=20
python trainval_net.py \
                   --dataset pascal_voc --net res101  \
                   --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                   --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
		   --cuda --mGPUs
