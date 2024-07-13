import rosbag
from sensor_msgs.msg import PointCloud2
import open3d as o3d
import rospy
import os
from sensor_msgs import point_cloud2

def save_pcd_open3d(topic, bag_file, timestamp_file):
    bag = rosbag.Bag(bag_file, 'r')
    count = 0
    time_0= None 
    is_1st = True 

    with open(timestamp_file, 'w') as timestamp_txt:
        for topic, msg, t_ in bag.read_messages(topics=[topic]):
            # 변환: ROS PointCloud2 -> Open3D PointCloud
            
            _timestamp =msg.header
            
            ts = _timestamp.stamp
            t = ts.secs + ts.nsecs / float(1e9)	

            if(is_1st):
                time_0 = t
                is_1st = False
            else:
                pass

            t = t - time_0

            pc = point_cloud2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))
            pcloud = o3d.geometry.PointCloud()
            pcloud.points = o3d.utility.Vector3dVector(list(pc))

            # PCD 파일로 저장
            if(msg):            
                pcd_filename = f"output_{count}.pcd"
                o3d.io.write_point_cloud(pcd_filename, pcloud)
                print(f"Saved {pcd_filename}")

                timestamp_txt.write(f"{int(msg.header.seq)}, {'{:.12f}'.format(t)}\n")
                
                count += 1

    bag.close()
    print(f"Total {count} point clouds saved.")

# 사용 예
ROOT_DIR='/home/pyo/b1_perception'
bag_path = os.path.join(ROOT_DIR, 'b1_test.bag')
timestamp_path = os.path.join(ROOT_DIR, 'pc_timestamp.txt')
save_pcd_open3d('/NX1_front/depth/color/points', bag_path, timestamp_path)
