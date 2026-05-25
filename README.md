# RP_rover_global_loc

This repository holds the code for the research project on Global Localization SLAM.

Running evaluation:

export NAME=MY_2020_FINAL

ros2 bag record     /ekf/pose     /ekf/path     /ekf/landmarks     /ground_truth/odom     /cmd_odom     /rock_global_pose -o $NAME

python3 /home/ws/src/evaluation/extract_bag_ldm.py     --bag /home/ws/src/rosbags/$NAME     --out /home/ws/results/$NAME/csv

python3 /home/ws/src/evaluation/run_all_2.py     --csv_dir /home/ws/results/$NAME/csv     --out /home/ws/results/$NAME/     --skip_extract