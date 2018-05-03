# VisDrone2018

ECCV2018的一个workshop举办的比赛，详见[Vision Meets Drones: A Challenge](http://www.aiskyeye.com/).

VisDrone2018_Dev_Kit: 官方提供的针对数据集的工具，用于评测。可以改为其他工具，比如在图片上显示BBox；

Txt2XML:官方给定数据集的Ground Truth是自己的标注方式（Txt），该工具将该标注方式转化为PASCAL VOC2007的标注方式（XML）；Python实现；

ShowBBOXFromXML:针对PASCAL VOC2007，在图片上显示BBox；Python实现；该工具已经和官方给定基于Matlab的代码做过准确度对比，检验通过；

未分享代码：离线Badcase分析工具，通过该工具，发现官方给定数据集的三个BBox的标注问题（会造成NaN问题）；

VisEval: VisDrone2018的Python评估代码，逻辑有误。