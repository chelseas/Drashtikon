#!/bin/bash

#now=`date `
#echo "*********************************************************" >> echo $now >> ./output/reports/osd_ptosis.txt
#echo "*********************************************************" >> echo $now >>  ./output/reports/str_ptosis.txt
#echo "*********************************************************" >> echo $now >> ./output/reports/osd_str.txt

#python3 frame.py test_train_partitions/osd_partition/osd_all/ test_train_partitions/ptosis_partition/ptosis_all/ 3 --hog --all & #>> ./output/reports/osd_ptosis.txt #&
#python3 frame.py test_train_partitions/str_partition/str_all/ test_train_partitions/ptosis_partition/ptosis_all/ 3 --bright --all & #>> ./output/reports/str_ptosis.txt  &
#wait
#python3 frame.py test_train_partitions/osd_partition/osd_all/ test_train_partitions/str_partition/str_all/ 3 --random  --all & #>> ./output/reports/osd_str.txt &


#python3 frame.py test_train_partitions/osd_partition/osd_all/ test_train_partitions/ptosis_partition/ptosis_all/ 3 --random --all & #>> ./output/reports/osd_ptosis.txt #&
#wait
python frame.py test_train_partitions/str_partition/str_all/ test_train_partitions/ptosis_partition/ptosis_all/ 3 --hog --all & #>> ./output/reports/str_ptosis.txt  &
#python3 frame.py test_train_partitions/osd_partition/osd_all/ test_train_partitions/str_partition/str_all/ 3 --bright  --all & #>> ./output/reports/osd_str.txt &

#wait

#python3 frame.py test_train_partitions/osd_partition/osd_all/ test_train_partitions/ptosis_partition/ptosis_all/ 3 --bright --all & #>> ./output/reports/osd_ptosis.txt #&
#wait
#python3 frame.py test_train_partitions/str_partition/str_all/ test_train_partitions/ptosis_partition/ptosis_all/ 3 --random  --all & #>> ./output/reports/str_ptosis.txt  &
python frame.py test_train_partitions/osd_partition/osd_all/ test_train_partitions/str_partition/str_all/ 3 --hog --all & #>> ./output/reports/osd_str.txt &


echo "Done"
