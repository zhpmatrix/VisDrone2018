SESSION=1
EPOCH=4
CHECKPOINT=3508
python demo.py --net vgg16 \
               --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
               --cuda --load_dir models
