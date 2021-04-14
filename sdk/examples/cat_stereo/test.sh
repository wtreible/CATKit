#!/bin/bash
python cat_sgm.py --left cones/im2.png --right cones/im6.png --left_gt cones/disp2.png --right_gt cones/disp6.png --output disparity_map.png --disp 64 --images False --eval True
