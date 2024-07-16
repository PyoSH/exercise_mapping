from utils_mapping import *

if __name__ =='__main__':
    topic_name = '/odom'  # 오도메트리 정보가 발행되는 ROS 토픽 이름
    ROOT_DIR = '/home/pyo/b1_perception/exercise_mapping'
    bag_file_path = os.path.join(ROOT_DIR, 'b1_test.bag')  # ROS bag 파일 경로
    output_file = os.path.join(ROOT_DIR, 'odometry.txt')  # 결과를 저장할 디렉토리 경로

    test = ODOM_FROM_ROS(ROOT_DIR, bag_file_path, topic_name, output_file)
    test.save_odometry_to_txt()
