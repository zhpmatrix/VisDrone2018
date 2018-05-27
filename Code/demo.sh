MAXCHECKPOINT=3234
for((EPOCH=8;EPOCH<=12;EPOCH++))
do
{
CUDA_VISIBLE_DEVICES=0 python demo.py --net res101 \
	       --load_name faster_rcnn_1_${EPOCH}_${MAXCHECKPOINT}.pth \
               --cuda --load_dir models
               #--checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
}&
done

for((EPOCH=13;EPOCH<=18;EPOCH++))
do
{
CUDA_VISIBLE_DEVICES=1 python demo.py --net res101 \
	       --load_name faster_rcnn_1_${EPOCH}_${MAXCHECKPOINT}.pth \
               --cuda --load_dir models
               #--checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
}&
done

wait
echo "Done!"
