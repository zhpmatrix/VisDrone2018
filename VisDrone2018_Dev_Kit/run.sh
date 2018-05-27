MAXCHECKPOINT=3234
for((EPOCH=10;EPOCH<=10;EPOCH++))
do
{
ANNO_DIR=faster_rcnn_1_${EPOCH}_${MAXCHECKPOINT}.pth
matlab -nodesktop -r "anno_dir='$ANNO_DIR'",evalDET
}&
done
echo "Done!"
