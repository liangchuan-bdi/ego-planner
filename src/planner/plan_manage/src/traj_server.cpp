#include <ego_planner/TrajExecTime.h>
#include <geometry_msgs/PoseStamped.h>
#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>

#include "bspline_opt/uniform_bspline.h"
#include "ego_planner/Bspline.h"
#include "nav_msgs/Odometry.h"
#include "quadrotor_msgs/PositionCommand.h"
#include "std_msgs/Empty.h"
#include "tf/tf.h"
#include "tf/transform_datatypes.h"
#include "visualization_msgs/Marker.h"

typedef enum {
  Cmd_None = 0,
  Cmd_BSpline = 1,
} Cmd_Type;

ros::Publisher pos_cmd_pub;
ros::Publisher fast_goal_pub;
ros::Publisher traj_exec_time_pub;

Eigen::Vector3d fastGoalLast_pos;
double fastGoalLast_yaw;

Eigen::Vector3d turnInPlaceTargetPos;
double turnInPlaceTargetYaw;

Eigen::Vector3d blindGoalTargetPos;
double blindGoalTargetSpeed;
double blindGoalTargetYaw;

double blind_goal_slow_down_dis;
double blind_goal_plan_ahead_dis_min;
double blind_goal_plan_ahead_dis_max;
double blind_goal_plan_ahead_dis_step_change_rate_max;

Eigen::Vector3d odomPos;
double odomYaw;
Eigen::Vector3d curSpeed;
bool hasOdom = false;

boost::shared_ptr<ego_planner::TrajExecTime> traj_exec_time_msg;

quadrotor_msgs::PositionCommand cmd;
double pos_gain[3] = {0, 0, 0};
double vel_gain[3] = {0, 0, 0};

using ego_planner::UniformBspline;

Cmd_Type cmd_type = Cmd_None;
vector<UniformBspline> traj_;
double traj_duration_;
ros::Time start_time_;
int traj_id_;

// yaw control
double last_yaw_, last_yaw_dot_;
double max_yaw_rate;
double max_yaw_offset_tolerance;
double time_forward_;

bool yawOffsetWasLarge = false;
Eigen::Vector3d last_pos;
double t_cur;

string planning_frame;
string planning_child_frame;

float tfTimeOut;
boost::shared_ptr<tf::TransformListener> mpTFListener = nullptr;

template <typename T>
inline void getParam(ros::NodeHandle &privateNode, const std::string &paramName,
                     T &param) {
  if (!privateNode.getParam(paramName, param)) {
    ROS_ERROR_STREAM("Parameter " << paramName << " is not set.");
    ROS_BREAK();
  }

  ROS_INFO_STREAM("Found parameter: " << paramName << ", value: " << param);
}

inline bool getTransform(const tf::TransformListener &TFListener,
                         const std::string &target_frame,
                         const std::string &source_frame, const ros::Time &time,
                         const ros::Duration &timeout,
                         tf::StampedTransform &transform,
                         const bool quiet = false) {
  if (TFListener.waitForTransform(target_frame, source_frame, time, timeout)) {
    try {
      TFListener.lookupTransform(target_frame, source_frame, time, transform);
    } catch (tf::LookupException &ex) {
      if (!quiet) {
        ROS_WARN("no transform available: %s\n", ex.what());
      }
      return false;
    } catch (tf::ConnectivityException &ex) {
      if (!quiet) {
        ROS_WARN("connectivity error: %s\n", ex.what());
      }
      return false;
    } catch (tf::ExtrapolationException &ex) {
      if (!quiet) {
        ROS_WARN("extrapolation error: %s\n", ex.what());
      }
      return false;
    }

    return true;
  }

  if (!quiet) {
    ROS_ERROR_STREAM(std::setprecision(9)
                     << std::fixed << "transformation not available from "
                     << source_frame << " to " << target_frame << " at "
                     << time.toSec());
  }

  return false;
}

void getLatestPose(tf::StampedTransform &tf_Planning_PlanningChild,
                   Eigen::Vector3d &pos) {
  if (!getTransform(*mpTFListener, planning_frame, planning_child_frame,
                    ros::Time(0), ros::Duration(tfTimeOut),
                    tf_Planning_PlanningChild)) {
    ROS_ERROR_STREAM("Transform not available in cmdCallback from "
                     << planning_child_frame << " to " << planning_frame
                     << " at " << ros::Time::now().toSec());
    return;
  }

  pos[0] = tf_Planning_PlanningChild.getOrigin().x();
  pos[1] = tf_Planning_PlanningChild.getOrigin().y();
  pos[2] = tf_Planning_PlanningChild.getOrigin().z();

  const float tfDelay =
      ros::Time::now().toSec() - tf_Planning_PlanningChild.stamp_.toSec();
  if (tfDelay > 0.3) {
    ROS_WARN_STREAM(
        "tfDelay of tf_Planning_PlanningChild is large: " << tfDelay);
  }
}

void bsplineCallback(ego_planner::BsplineConstPtr msg) {
  if (!hasOdom) {
    return;
  }

  // parse pos traj

  Eigen::MatrixXd pos_pts(3, msg->pos_pts.size());

  Eigen::VectorXd knots(msg->knots.size());
  for (size_t i = 0; i < msg->knots.size(); ++i) {
    knots(i) = msg->knots[i];
  }

  for (size_t i = 0; i < msg->pos_pts.size(); ++i) {
    pos_pts(0, i) = msg->pos_pts[i].x;
    pos_pts(1, i) = msg->pos_pts[i].y;
    pos_pts(2, i) = msg->pos_pts[i].z;
  }

  UniformBspline pos_traj(pos_pts, msg->order, 0.1);
  pos_traj.setKnot(knots);

  // parse yaw traj

  // Eigen::MatrixXd yaw_pts(msg->yaw_pts.size(), 1);
  // for (int i = 0; i < msg->yaw_pts.size(); ++i) {
  //   yaw_pts(i, 0) = msg->yaw_pts[i];
  // }

  // UniformBspline yaw_traj(yaw_pts, msg->order, msg->yaw_dt);

  start_time_ = msg->start_time;
  traj_id_ = msg->traj_id;

  traj_.clear();
  traj_.push_back(pos_traj);
  traj_.push_back(traj_[0].getDerivative());
  traj_.push_back(traj_[1].getDerivative());

  traj_duration_ = traj_[0].getTimeSum();

  cmd_type = Cmd_BSpline;
  yawOffsetWasLarge = false;
}

inline void roundAngle(double &angle) {
  while (angle < -M_PI) {
    angle += 2 * M_PI;
  }

  while (angle > M_PI) {
    angle -= 2 * M_PI;
  }
}

inline double computeYawDiff(double yaw, double yawCurrent) {
  roundAngle(yaw);
  roundAngle(yawCurrent);

  double yawDiff = yaw - yawCurrent;
  roundAngle(yawDiff);

  return yawDiff;
}

void calculate_yaw_helper(const double &yawTarget, const double &yawCurrent,
                          const ros::Time &time_now, const ros::Time &time_last,
                          bool &yawOffsetLarge,
                          std::pair<double, double> &yaw_yawdot) {
  double yaw = 0;
  double yawdot = 0;

  // compute yaw offset
  const double yawOffset = computeYawDiff(yawTarget, yawCurrent);
  yawOffsetLarge = fabs(yawOffset) > max_yaw_offset_tolerance;

  double max_yaw_change = max_yaw_rate * (time_now - time_last).toSec();

  if (yawTarget - last_yaw_ > M_PI) {
    if (yawTarget - last_yaw_ - 2 * M_PI < -max_yaw_change) {
      yaw = last_yaw_ - max_yaw_change;
      if (yaw < -M_PI) yaw += 2 * M_PI;

      yawdot = -max_yaw_rate;
    } else {
      yaw = yawTarget;
      if (yaw - last_yaw_ > M_PI)
        yawdot = -max_yaw_rate;
      else
        yawdot = (yawTarget - last_yaw_) / (time_now - time_last).toSec();
    }
  } else if (yawTarget - last_yaw_ < -M_PI) {
    if (yawTarget - last_yaw_ + 2 * M_PI > max_yaw_change) {
      yaw = last_yaw_ + max_yaw_change;
      if (yaw > M_PI) yaw -= 2 * M_PI;

      yawdot = max_yaw_rate;
    } else {
      yaw = yawTarget;
      if (yaw - last_yaw_ < -M_PI)
        yawdot = max_yaw_rate;
      else
        yawdot = (yawTarget - last_yaw_) / (time_now - time_last).toSec();
    }
  } else {
    if (yawTarget - last_yaw_ < -max_yaw_change) {
      yaw = last_yaw_ - max_yaw_change;
      if (yaw < -M_PI) yaw += 2 * M_PI;

      yawdot = -max_yaw_rate;
    } else if (yawTarget - last_yaw_ > max_yaw_change) {
      yaw = last_yaw_ + max_yaw_change;
      if (yaw > M_PI) yaw -= 2 * M_PI;

      yawdot = max_yaw_rate;
    } else {
      yaw = yawTarget;
      if (yaw - last_yaw_ > M_PI)
        yawdot = -max_yaw_rate;
      else if (yaw - last_yaw_ < -M_PI)
        yawdot = max_yaw_rate;
      else
        yawdot = (yawTarget - last_yaw_) / (time_now - time_last).toSec();
    }
  }

  if (fabs(yaw - last_yaw_) <= max_yaw_change)
    yaw = 0.5 * last_yaw_ + 0.5 * yaw;  // nieve LPF
  yawdot = 0.5 * last_yaw_dot_ + 0.5 * yawdot;

  yaw_yawdot.first = yaw;
  yaw_yawdot.second = yawdot;
}

std::pair<double, double> calculate_yaw(const double &t_cur,
                                        const double &yawCurrent,
                                        const Eigen::Vector3d &curPos,
                                        const Eigen::Vector3d &targetPos,
                                        const ros::Time &time_now,
                                        const ros::Time &time_last,
                                        bool &yawOffsetLarge) {
  Eigen::Vector3d dir = targetPos - curPos;
  if (dir.norm() <= 0.1) {
    // look forward
    dir = t_cur + time_forward_ <= traj_duration_
              ? traj_[0].evaluateDeBoorT(t_cur + time_forward_) - targetPos
              : traj_[0].evaluateDeBoorT(traj_duration_) - targetPos;
  }

  dir(2) = 0.0;
  double yawTarget = dir.norm() > 0.1 ? atan2(dir(1), dir(0)) : yawCurrent;

  std::pair<double, double> yaw_yawdot(0, 0);
  calculate_yaw_helper(yawTarget, yawCurrent, time_now, time_last,
                       yawOffsetLarge, yaw_yawdot);
  return yaw_yawdot;
}

void cmdCallback(const ros::TimerEvent &e) {
  /* no publishing before receive traj_ */
  if (cmd_type == Cmd_None) {
    return;
  }

  Eigen::Vector3d pos(Eigen::Vector3d::Zero()), vel(Eigen::Vector3d::Zero()),
      acc(Eigen::Vector3d::Zero()), pos_f;
  std::pair<double, double> yaw_yawdot(0, 0);

  ros::Time time_now = ros::Time::now();
  static ros::Time time_last = ros::Time::now();

  switch (cmd_type) {
    case Cmd_BSpline: {
      if (!yawOffsetWasLarge) {
        t_cur = (time_now - start_time_).toSec();
      }

      // pub traj_exec_time
      traj_exec_time_msg->traj_id = traj_id_;
      traj_exec_time_msg->exec_time = t_cur;
      traj_exec_time_pub.publish(traj_exec_time_msg);

      // get latest pose
      tf::StampedTransform tf_Planning_PlanningChild;
      Eigen::Vector3d curPos;
      getLatestPose(tf_Planning_PlanningChild, curPos);
      const double yawCurrent =
          tf::getYaw(tf_Planning_PlanningChild.getRotation());

      if (t_cur < traj_duration_ && t_cur >= 0.0) {
        pos = traj_[0].evaluateDeBoorT(t_cur);
        vel = traj_[1].evaluateDeBoorT(t_cur);
        acc = traj_[2].evaluateDeBoorT(t_cur);

        /*** calculate yaw ***/
        bool yawOffsetLarge;
        yaw_yawdot = calculate_yaw(t_cur, yawCurrent, curPos, pos, time_now,
                                   time_last, yawOffsetLarge);
        /*** calculate yaw ***/

        if (yawOffsetLarge) {
          // stay and rotate
          if (!yawOffsetWasLarge) {
            last_pos = curPos;
            yawOffsetWasLarge = true;
          }
          pos = last_pos;
          vel.setZero();
          acc.setZero();
        } else if (yawOffsetWasLarge) {
          start_time_ = ros::Time().fromSec(time_now.toSec() - t_cur);
          yawOffsetWasLarge = false;
        }

        double tf = min(traj_duration_, t_cur + 2.0);
        pos_f = traj_[0].evaluateDeBoorT(tf);
      } else if (t_cur >= traj_duration_) {
        /* hover when finish traj_ */
        pos = traj_[0].evaluateDeBoorT(traj_duration_);
        vel.setZero();
        acc.setZero();

        yaw_yawdot.first = yawCurrent;
        yaw_yawdot.second = 0;

        pos_f = pos;
      } else {
        cout << "[Traj server]: invalid time." << endl;
      }

      // cmd.header.stamp = time_now;
      // cmd.header.frame_id = "world";
      // cmd.trajectory_flag =
      //     quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_READY;
      // cmd.trajectory_id = traj_id_;

      // cmd.position.x = pos(0);
      // cmd.position.y = pos(1);
      // cmd.position.z = pos(2);

      // cmd.velocity.x = vel(0);
      // cmd.velocity.y = vel(1);
      // cmd.velocity.z = vel(2);

      // cmd.acceleration.x = acc(0);
      // cmd.acceleration.y = acc(1);
      // cmd.acceleration.z = acc(2);

      // cmd.yaw = yaw_yawdot.first;
      // cmd.yaw_dot = yaw_yawdot.second;

      // pos_cmd_pub.publish(cmd);
    } break;

    default:
      break;
  }

  geometry_msgs::PoseStamped fastGoal;
  fastGoal.header.frame_id = "world";
  fastGoal.pose.position.x = pos(0);
  fastGoal.pose.position.y = pos(1);
  fastGoal.pose.position.z = pos(2);
  fastGoal.pose.orientation = tf::createQuaternionMsgFromYaw(yaw_yawdot.first);
  fast_goal_pub.publish(fastGoal);

  // record for yaw control
  time_last = time_now;
  last_yaw_ = yaw_yawdot.first;
  last_yaw_dot_ = yaw_yawdot.second;
}

void getRPYFromQuat(const Eigen::Quaterniond &quat, double &roll, double &pitch,
                    double &yaw) {
  tf::Matrix3x3 rot;
  tf::matrixEigenToTF(quat.toRotationMatrix(), rot);
  rot.getRPY(roll, pitch, yaw);
}
void getRPYFromQuat(const geometry_msgs::Quaternion &quat, double &roll,
                    double &pitch, double &yaw) {
  Eigen::Quaterniond eigQ;
  eigQ.x() = quat.x;
  eigQ.y() = quat.y;
  eigQ.z() = quat.z;
  eigQ.w() = quat.w;
  getRPYFromQuat(eigQ, roll, pitch, yaw);
}

void odomCallback(const nav_msgs::OdometryConstPtr &msg) {
  odomPos[0] = msg->pose.pose.position.x;
  odomPos[1] = msg->pose.pose.position.y;
  odomPos[2] = msg->pose.pose.position.z;
  double odomRoll, odomPitch;
  getRPYFromQuat(msg->pose.pose.orientation, odomRoll, odomPitch, odomYaw);
  curSpeed =
      Eigen::Vector3d(msg->twist.twist.linear.x, msg->twist.twist.linear.y,
                      msg->twist.twist.linear.z);
  if (!hasOdom) {
    last_yaw_ = odomYaw;
  }
  hasOdom = true;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "traj_server");
  ros::NodeHandle node;
  ros::NodeHandle nh("~");

  ros::Subscriber bspline_sub =
      node.subscribe("planning/bspline", 10, bsplineCallback);
  ros::Subscriber odom_sub =
      node.subscribe("/mavros/local_position/odom", 10, odomCallback);

  pos_cmd_pub =
      node.advertise<quadrotor_msgs::PositionCommand>("/position_cmd", 50);
  fast_goal_pub = node.advertise<geometry_msgs::PoseStamped>("/fast_goal", 50);
  traj_exec_time_pub =
      node.advertise<ego_planner::TrajExecTime>("/planning/traj_exec_time", 10);

  traj_exec_time_msg = boost::make_shared<ego_planner::TrajExecTime>();

  ros::Timer cmd_timer = node.createTimer(ros::Duration(0.01), cmdCallback);

  /* control parameter */
  cmd.kx[0] = pos_gain[0];
  cmd.kx[1] = pos_gain[1];
  cmd.kx[2] = pos_gain[2];

  cmd.kv[0] = vel_gain[0];
  cmd.kv[1] = vel_gain[1];
  cmd.kv[2] = vel_gain[2];

  nh.param("traj_server/blind_goal_slow_down_dis", blind_goal_slow_down_dis,
           10.0);
  nh.param("traj_server/blind_goal_plan_ahead_dis_min",
           blind_goal_plan_ahead_dis_min, 0.5);
  nh.param("traj_server/blind_goal_plan_ahead_dis_max",
           blind_goal_plan_ahead_dis_max, 10.0);
  nh.param("traj_server/blind_goal_plan_ahead_dis_step_change_rate_max",
           blind_goal_plan_ahead_dis_step_change_rate_max, 1.0);

  nh.param("traj_server/time_forward", time_forward_, -1.0);

  nh.param("traj_server/max_yaw_rate", max_yaw_rate, 30.0);
  max_yaw_rate = max_yaw_rate / 180.0 * M_PI;
  nh.param("traj_server/max_yaw_offset_tolerance", max_yaw_offset_tolerance,
           30.0);
  max_yaw_offset_tolerance = max_yaw_offset_tolerance / 180.0 * M_PI;

  getParam(nh, "traj_server/planning_frame", planning_frame);
  getParam(nh, "traj_server/planning_child_frame", planning_child_frame);
  nh.param("tfTimeOut", tfTimeOut, 0.01f);
  mpTFListener = boost::make_shared<tf::TransformListener>(ros::Duration(3600));

  last_yaw_ = 0.0;
  last_yaw_dot_ = 0.0;

  ros::Duration(1.0).sleep();

  ROS_WARN("[Traj server]: ready.");

  ros::spin();

  return 0;
}