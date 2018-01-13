#!/bin/bash
wget 'https://www.dropbox.com/s/4u4pupow9g3v3a5/hw6_model.h5?dl=1'
wget 'https://www.dropbox.com/s/3u1tcg8hr631lqp/kmeans.sav?dl=1'
mv hw6_model.h5?dl=1 hw6_model.h5
mv kmeans.sav?dl=1 kmeans.sav
python3 hw6.py $1 $2 $3