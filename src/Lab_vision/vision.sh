#! /bin/bash
echo "starting shell script..";

count=`ls -1 *.json 2>/dev/null | wc -l`
if [ $count != 0 ]
    then
    echo "There is an existing JSON data...running statistics on it"
    python print_hpo_stats.py -p $(ls *.json)

else
     python download_hpo.py;
     python print_hpo_stats.py -p $(ls *.json)
fi

