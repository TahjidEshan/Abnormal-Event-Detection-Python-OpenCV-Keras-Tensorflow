#!/bin/bash

#python featureGenerator.py /home/tahjidashfaquemostafa/Thesis/Data/Normal\ Crowds/341-46_l.mov 1
#echo "Done with 1"
#python featureGenerator.py /home/tahjidashfaquemostafa/Thesis/Data/Abnormal\ Crowds/1183-88_l.mov 0
#echo "Done with 2"
#python featureGenerator.py /home/tahjidashfaquemostafa/Thesis/Data/Abnormal\ Crowds/490-208_l.mov 0
#echo "Done with 3"
#python featureGenerator.py /home/tahjidashfaquemostafa/Thesis/Data/Normal\ Crowds/637-147_l.mov 1
#echo "Done with 4"
#echo '7' | python classifier.py
#echo "Done with 5" > 'Track.txt'
#python plot.py
python featureGeneratorOther.py /home/tahjidashfaquemostafa/Thesis/Data/Abnormal\ Crowds/263C044_064_c.mov 0
#echo "Done with 5"
python featureGeneratorOther.py /home/tahjidashfaquemostafa/Thesis/Data/Abnormal\ Crowds/263C044_060_c.mov 0
#echo "Done with 4"
python featureGeneratorOther.py /home/tahjidashfaquemostafa/Thesis/Data/Normal\ Crowds/688-10_l.mov 1
python featureGeneratorOther.py /home/tahjidashfaquemostafa/Thesis/Data/Normal\ Crowds/shibuya1.mov 1
python featureGeneratorOther.py /home/tahjidashfaquemostafa/Thesis/Data/Normal\ Crowds/CRW116.mov 1

python featureGeneratorOther.py /home/tahjidashfaquemostafa/Thesis/Data/Abnormal\ Crowds/2010-291_l.avi 0
