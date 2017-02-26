#!/bin/bash

python featureGenerator.py /home/tahjidashfaquemostafa/Thesis/Data/Normal\ Crowds/341-46_l.mov 1
echo "Done with 1" > 'Track.txt'
python featureGenerator.py /home/tahjidashfaquemostafa/Thesis/Data/Abnormal\ Crowds/1183-88_l.mov 0
echo "Done with 2" > 'Track.txt'
python featureGenerator.py /home/tahjidashfaquemostafa/Thesis/Data/Abnormal\ Crowds/490-208_l.mov 0
echo "Done with 3" > 'Track.txt'
python featureGenerator.py /home/tahjidashfaquemostafa/Thesis/Data/Normal\ Crowds/637-147_l.mov 1
echo "Done with 4" > 'Track.txt'
echo '7' | python classifier.py
echo "Done with 5" > 'Track.txt'
python plot.py
