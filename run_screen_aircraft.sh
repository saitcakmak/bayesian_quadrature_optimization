lw=$1
up=$2

for i in $(seq $lw $up)
do
	screen -d -m  bash aircraft.sh $i	
done
