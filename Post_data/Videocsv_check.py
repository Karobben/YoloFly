#!/usr/bin/env python3

for i in $(ls csv/*.csv);
do NUM=$(tail -n 1 $i| awk '{print $1}')
    VIDEO=$(echo $i| sed 's=csv/==;s/.csv//')
echo $i $(grep $VIDEO Video_list.csv | awk '{OFS="-"; print '$NUM',$5}'|bc)
done
