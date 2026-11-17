import sys

from utilities import euler_from_quaternion, calculate_angular_error, calculate_linear_error
from pid import PID_ctrl

from rclpy import init, spin, spin_once
from rclpy.node import Node
from geometry_msgs.msg import Twist


from rclpy.qos import QoSProfile
from nav_msgs.msg import Odometry as odom

from localization import localization, rawSensors, kalmanFilter

from planner import TRAJECTORY_PLANNER, POINT_PLANNER, SPIRAL_4TUNE, planner
from controller import controller, trajectoryController


class decision_maker(Node):
    
    
    def __init__(self, publisher_msg, publishing_topic, qos_publisher, rate=10, motion_type=POINT_PLANNER,
                 Q=0.5, R=0.5, logName="robotPose.csv"):

        super().__init__("decision_maker")

        self.publisher=self.create_publisher(publisher_msg, publishing_topic, qos_profile=qos_publisher)
        
        
        publishing_period=1/rate

        self.reachThreshold=0.01


        # TODO Part 3: use the Kalman Filter
        # Pose comes from the EKF (odom + IMU fusion) instead of just odometry. The filter's dt
        # matches the timer publishing period because localization runs once per timerCallback.
        self.localizer = localization(type=kalmanFilter, dt=publishing_period, Q=Q, R=R, loggerName=logName)
        
        if motion_type==POINT_PLANNER:
            self.controller=controller(klp=0.2, klv=0.5, kap=0.8, kav=0.6)      
            self.planner=planner(POINT_PLANNER)

        
        elif motion_type==TRAJECTORY_PLANNER:
            self.controller=trajectoryController(klp=0.2, klv=0.5, kap=0.8, kav=0.6)      
            self.planner=planner(TRAJECTORY_PLANNER)
        elif motion_type==SPIRAL_4TUNE:
            self.controller=None
            self.planner=None
            self.linearVelocity = 0
        
        else:
            print("Error! you don't have this type of planner", file=sys.stderr)
            return

        self.motion_type=motion_type
        self.goal=self.planner.plan() if self.motion_type != SPIRAL_4TUNE else None
        

        self.create_timer(publishing_period, self.timerCallback)



    def timerCallback(self):
        
        spin_once(self.localizer)

        if self.localizer.getPose() is  None:
            print("waiting for odom msgs ....")
            return
        
        
        vel_msg=Twist()

        if self.motion_type == SPIRAL_4TUNE:
            # Spiral for EKF tuning: constant yaw rate, linear velocity ramps up to 1.0 m/s.
            # The ramp was reduced from 0.01 to 0.005 per step so the spiral grows more
            # slowly and fits in the lab space.
            self.linearVelocity += 0.005 if self.linearVelocity < 1.0 else 0.0
            vel_msg.linear.x=self.linearVelocity
            vel_msg.angular.z=1.0
            self.publisher.publish(vel_msg)
            return

        
        if type(self.goal) == list:
            reached_goal=True if calculate_linear_error(self.localizer.getPose(), self.goal[-1]) <self.reachThreshold else False
        else: 
            reached_goal=True if calculate_linear_error(self.localizer.getPose(), self.goal) <self.reachThreshold else False


        if reached_goal:
            print("reached goal")
            self.publisher.publish(vel_msg)
            
            self.controller.PID_angular.logger.save_log()
            self.controller.PID_linear.logger.save_log()
            
            raise SystemExit
        
        velocity, yaw_rate = self.controller.\
            vel_request(self.localizer.getPose(), self.goal, True)

        
        vel_msg.linear.x=velocity
        vel_msg.angular.z=yaw_rate
        
        self.publisher.publish(vel_msg)


import argparse
def main(args=None):
    
    
    init()
    
    
    # one CSV per run so the Q/R variations don't overwrite each other
    # (e.g. spiral_Q0.5_R0.5.csv), unless --log gives a name explicitly
    logName = args.log or f"{args.motion}_Q{args.Q}_R{args.R}.csv"
    # same EKF settings for every motion type
    kw = dict(Q=args.Q, R=args.R, logName=logName)

    if args.motion == "point":
        DM=decision_maker(Twist, "/cmd_vel", 10, motion_type=POINT_PLANNER, **kw)
    elif args.motion == "trajectory":
        DM=decision_maker(Twist, "/cmd_vel", 10, motion_type=TRAJECTORY_PLANNER, **kw)
    elif args.motion == "spiral":
        DM=decision_maker(Twist, "/cmd_vel", 10, motion_type=SPIRAL_4TUNE, **kw)
    else:
        print("invalid motion type", file=sys.stderr)
        return



    try:
        spin(DM)
    except SystemExit:
        print(f"reached there successfully {DM.localizer.pose}")




if __name__=="__main__":
    
    argParser=argparse.ArgumentParser(description="point or trajectory") 
    argParser.add_argument("--motion", type=str, default="spiral")
    # EKF tuning parameters, so Q/R variations can be run without editing code
    # e.g. python3 decisions.py --motion spiral --Q 5.0 --R 0.5
    argParser.add_argument("--Q", type=float, default=0.5, help="process covariance scale")
    argParser.add_argument("--R", type=float, default=0.5, help="measurement covariance scale")
    argParser.add_argument("--log", type=str, default=None, help="CSV name (default: <motion>_Q<Q>_R<R>.csv)")
    args = argParser.parse_args()

    main(args)
