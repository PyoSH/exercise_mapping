from utils_mapping import *
import os

if __name__ =='__main__':
    ROOT_DIR = '/home/pyo/b1_perception/exercise_mapping'
    bag_file_path = os.path.join(ROOT_DIR, 'b1_test.bag')  # ROS bag 파일 경로
    timestamp_path = os.path.join(ROOT_DIR, 'pcds', 'pc_timestamp.txt')
    topic_name = '/NX1_front/depth/color/points'

    test = PCL_FROM_ROS(ROOT_DIR, bag_file_path, topic_name, timestamp_path)
    test.run()

