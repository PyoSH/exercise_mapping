import rosbag
from nav_msgs.msg import Odometry
import rospy
import os
import numpy as np
from copy import deepcopy
import tf.transformations as tf_trans

def save_odometry_to_txt(topic, bag_file, output_file):
    bag = rosbag.Bag(bag_file, 'r')
    count = 0
    infoVector=np.zeros(9)
    time_0 = 0
    pose_0 = None
    pose_0_orientation_inv = None
    is_1st = True

    with open(output_file, 'w') as file:
        for _, msg, t_ in bag.read_messages(topics=[topic]):
            # 오도메트리 데이터 추출
            _pose=msg.pose.pose
            _timestamp =msg.header
            
            ts = _timestamp.stamp
            t = ts.secs + ts.nsecs / float(1e9)	
            
            # 1st iteration -> set t_0 
            if(is_1st):
                time_0 = t
                pose_0 = deepcopy(_pose)
                is_1st = False

                # pose_0_orientation_inv = tf_trans.quaternion_inverse([pose_0.orientation.x, 
                                        # pose_0.orientation.y,pose_0.orientation.z, pose_0.orientation.w])
            else:
                pass

            t = t - time_0

            curr_orientation =[_pose.orientation.x, _pose.orientation.y, _pose.orientation.z, _pose.orientation.w]
            # relative_orientation = tf_trans.quaternion_multiply(curr_orientation, pose_0_orientation_inv)
            
            infoVector[0] = _pose.position.x 
            infoVector[1] = _pose.position.y
            infoVector[2] = _pose.position.z 
            infoVector[3] = curr_orientation[0]
            infoVector[4] = curr_orientation[1]
            infoVector[5] = curr_orientation[2]
            infoVector[6] = curr_orientation[3]
            infoVector[7] = int(_timestamp.seq)
            infoVector[8] = '{:.12f}'.format(t)

            if(infoVector[0]):
                cur_pose=infoVector
                file.write(str(cur_pose[0])+','+str(cur_pose[1])+','+str(cur_pose[2])+','
                    +str(cur_pose[3])+','+str(cur_pose[4])+','+str(cur_pose[5])+','+str(cur_pose[6])+','
                    +str(cur_pose[7])+','+str(cur_pose[8])+'\n')
                            
                count += 1
                print(f"Data at {count} saved.")

    bag.close()
    print(f"Total {count} odometry entries saved to {output_file}.")


topic_name = '/odom'  # 오도메트리 정보가 발행되는 ROS 토픽 이름
ROOT_DIR='/home/pyo/b1_perception'
bag_file_path = os.path.join(ROOT_DIR, 'b1_test.bag')  # ROS bag 파일 경로
output_file = os.path.join(ROOT_DIR, 'odometry.txt')  # 결과를 저장할 디렉토리 경로

save_odometry_to_txt(topic_name, bag_file_path, output_file)