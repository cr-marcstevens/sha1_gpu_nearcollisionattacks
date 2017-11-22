#!/bin/bash

echo "Generating basesolutions for 21 minutes..."
# Generate base solutions for 21 minutes
SEED=$RANDOM$RANDOM$RANDOM
bin/sha1_freestart76_basesolgen --seed 1$SEED -f -o fs76_1.bin > fs76_1.log &
bin/sha1_freestart76_basesolgen --seed 2$SEED -f -o fs76_2.bin > fs76_2.log &
bin/sha1_freestart76_basesolgen --seed 3$SEED -f -o fs76_3.bin > fs76_3.log &
bin/sha1_freestart76_basesolgen --seed 4$SEED -f -o fs76_4.bin > fs76_4.log &

echo "21 minutes left..."
for ((i=20; i>=0; --i)); do 
	sleep 60
	echo "$i minutes left..."
done

killall sha1_freestart76_basesolgen

# Loop: run GPU attack on available basesolutions, meanwhile generate new basesolutions
for ((i=1;;++i)); do
	# combine generated basesolutions in 1 file
	inputs=""
	for f in fs76_*.bin ; do
		inputs="$inputs -i $f"
	done
	echo bin/sha1_freestart76_basesolgen -f -o fs76.bin $inputs
	bin/sha1_freestart76_basesolgen -f -o fs76.bin $inputs
	rm fs76_*.bin

	# start new generation of basesolutions in background
	SEED=$RANDOM$RANDOM$RANDOM
	bin/sha1_freestart76_basesolgen --seed 1${i}0$SEED -f -o fs76_1.bin >> fs76_1.log &
	bin/sha1_freestart76_basesolgen --seed 2${i}0$SEED -f -o fs76_2.bin >> fs76_2.log &
	bin/sha1_freestart76_basesolgen --seed 3${i}0$SEED -f -o fs76_3.bin >> fs76_3.log &
	bin/sha1_freestart76_basesolgen --seed 4${i}0$SEED -f -o fs76_4.bin >> fs76_4.log &

	# start timer for 21 minutes
	rm -f fs76.20min.timer
	(sleep $((21*60)); touch fs76.20min.timer) &

	# start GPU attack
	bin/sha1_freestart76_gpuattack -a -i fs76.bin -o fs76_q56_$i.bin | tee -a bs76_gpu.log
	
	# check for freestart collision
	bin/sha1_freestart76_basesolgen -v -i fs76_q56_$i.bin > fs76_q56_$i.log
	if grep "Found solution" -A52 -B80 fs76_q56_$i.log ; then
		killall sha1_freestart76_basesolgen
		exit 0
	fi
	
	# wait till timer has elapsed if it has not already
	while [ ! -f fs76.20min.timer ]; do sleep 10; done
	rm -f fs76.20min.timer
	
	# kill background generation of basesolutions
	killall sha1_freestart76_basesolgen
done
