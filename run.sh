#!/bin/bash

python featureGenerator.py /home/tahjidashfaquemostafa/Thesis/Data/Normal\ Crowds/341-46_l.mov 1
python featureGenerator.py /home/tahjidashfaquemostafa/Thesis/Data/Normal\ Crowds/637-147_l.mov 1
python featureGeneratorOther.py /home/tahjidashfaquemostafa/Thesis/Data/Normal\ Crowds/688-10_l.mov 1
python featureGeneratorOther.py /home/eshan/Thesis/Data/Normal\ Crowds/shibuya1.mov 1
python featureGenerator.py /home/tahjidashfaquemostafa/Thesis/Data/Normal\ Crowds/CRW116.mov 1
python featureGenerator0.py /home/tahjidashfaquemostafa/Thesis/Data/Normal\ Crowds/821-10_l.mov 1
python featureGenerator0.py /home/tahjidashfaquemostafa/Thesis/Data/Normal\ Crowds/879-38_l.mov 1
python featureGenerator1.py /home/tahjidashfaquemostafa/Thesis/Data/Normal\ Crowds/906-17_l.mov 1
python featureGenerator1.py /home/tahjidashfaquemostafa/Thesis/Data/Normal\ Crowds/3687-18_70.mov 1
python featureGenerator1.py /home/tahjidashfaquemostafa/Thesis/Data/Normal\ Crowds/9019-13_l.mov 1
python featureGenerator0.py /home/tahjidashfaquemostafa/Thesis/Data/Normal\ Crowds/81872953_l.mov 1
python featureGenerator0.py /home/tahjidashfaquemostafa/Thesis/Data/Normal\ Crowds/sfw20110024_l.mov 1

python featureGeneratorOther.py /home/tahjidashfaquemostafa/Thesis/Data/Abnormal\ Crowds/263C044_064_c.mov 0
python featureGeneratorOther.py /home/tahjidashfaquemostafa/Thesis/Data/Abnormal\ Crowds/263C044_060_c.mov 0
python featureGenerator.py /home/tahjidashfaquemostafa/Thesis/Data/Abnormal\ Crowds/490-208_l.mov 0
python featureGenerator.py /home/tahjidashfaquemostafa/Thesis/Data/Abnormal\ Crowds/1183-88_l.mov 0
python featureGeneratorOther.py /home/tahjidashfaquemostafa/Thesis/Data/Abnormal\ Crowds/2010-291_l.avi 0
python featureGenerator0.py /home/tahjidashfaquemostafa/Thesis/Data/Abnormal\ Crowds/2014-140_l.avi 0
python featureGenerator1.py /home/tahjidashfaquemostafa/Thesis/Data/Abnormal\ Crowds/2017-420_l.mov 0
python featureGenerator1.py /home/tahjidashfaquemostafa/Thesis/Data/Abnormal\ Crowds/3452204_031_c.mov 0
echo '7' | python classifier.py
