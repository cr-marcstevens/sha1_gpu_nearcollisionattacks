#!/bin/bash

bin/sha1_freestart76_basesolgen --seed 1$RANDOM$RANDOM -f -o bs76_1.bin &
bin/sha1_freestart76_basesolgen --seed 2$RANDOM$RANDOM -f -o bs76_2.bin &
bin/sha1_freestart76_basesolgen --seed 3$RANDOM$RANDOM -f -o bs76_3.bin &
bin/sha1_freestart76_basesolgen --seed 4$RANDOM$RANDOM -f -o bs76_4.bin &

sleep 660

killall sha1_freestart76_basesolgen

bin/sha1_freestart76_basesolgen -f `ls bs76_*.bin | sed s/\(bs76_.*.bin\)/-i \1/g` -o bs76.bin
