#!/bin/bash

export BINDIR=$(dirname $0)/bin
export BASESOLGEN=$BINDIR/sha1_freestart76_basesolgen
export GPUATTACK=$BINDIR/sha1_freestart76_gpuattack

echo "Generating basesolutions for 21 minutes..."
# Generate base solutions for 21 minutes
SEED=$RANDOM$RANDOM$RANDOM$RANDOM
$BASESOLGEN --seed 1_$SEED -g -m 262144 -o basesol76_1.bin > basesol76_1.log &
$BASESOLGEN --seed 2_$SEED -g -m 262144 -o basesol76_2.bin > basesol76_2.log &
$BASESOLGEN --seed 3_$SEED -g -m 262144 -o basesol76_3.bin > basesol76_3.log &
$BASESOLGEN --seed 4_$SEED -g -m 262144 -o basesol76_4.bin > basesol76_4.log &

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
	for f in basesol76_*.bin ; do
		inputs="$inputs -i $f"
	done
	echo $BASESOLGEN -f -o basesol76.run$i.bin $inputs
	$BASESOLGEN -f -o basesol76.run$i.bin $inputs
	rm basesol76_*.bin

	# start new generation of basesolutions in background
	SEED=$RANDOM$RANDOM$RANDOM$RANDOM
	$BASESOLGEN --seed 1_${i}_$SEED -g -m 262144 -o basesol76_1.bin >> basesol76_1.log &
	$BASESOLGEN --seed 2_${i}_$SEED -g -m 262144 -o basesol76_2.bin >> basesol76_2.log &
	$BASESOLGEN --seed 3_${i}_$SEED -g -m 262144 -o basesol76_3.bin >> basesol76_3.log &
	$BASESOLGEN --seed 4_${i}_$SEED -g -m 262144 -o basesol76_4.bin >> basesol76_4.log &

	# start timer for 21 minutes
	rm -f basesol76.20min.timer
	(sleep $((21*60)); touch basesol76.20min.timer) &

	# start GPU attack
	$GPUATTACK -a -i basesol76.run$i.bin -o fs76_q56_$i.bin | tee -a fs76_gpu.log
	
	# check for freestart collision
	$BASESOLGEN -v -i fs76_q56_$i.bin > fs76_q56_$i.log
	if grep "Found solution" -A52 -B80 fs76_q56_$i.log ; then
		killall sha1_freestart76_basesolgen
		exit 0
	fi
	
	# wait till timer has elapsed if it has not already
	while [ ! -f basesol76.20min.timer ]; do sleep 10; done
	rm -f basesol76.20min.timer
	
	# kill background generation of basesolutions
	killall sha1_freestart76_basesolgen
done
