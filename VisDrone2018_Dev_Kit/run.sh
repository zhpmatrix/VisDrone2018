source ~/.bashrc

scp -r b418-xiwei@172.18.147.43:/home/b418-xiwei/zhanghaipeng/faster-rcnn.pytorch-master/annotations/faster_rcnn* /Users/zhpmatrix/Desktop/VisDrone/Results/Det/small_anchors/annotations

MAXCHECKPOINT=3234
START=8
END=18
for((EPOCH=$START;EPOCH<=$END;EPOCH++))
do
{
ANNO_DIR=faster_rcnn_1_${EPOCH}_${MAXCHECKPOINT}.pth
matlab -nodesktop -r "anno_dir='$ANNO_DIR'",evalDET
}&
done
wait
echo "Done!"
