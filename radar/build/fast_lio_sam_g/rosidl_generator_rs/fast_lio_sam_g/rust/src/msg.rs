#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};



// Corresponds to fast_lio_sam_g__msg__Pose6D
/// the preintegrated Lidar states at the time of IMU measurements in a frame

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Pose6D {
    /// the offset time of IMU measurement w.r.t the first lidar point
    pub offset_time: f64,

    /// the preintegrated total acceleration (global frame) at the Lidar origin
    pub acc: [f64; 3],

    /// the unbiased angular velocity (body frame) at the Lidar origin
    pub gyr: [f64; 3],

    /// the preintegrated velocity (global frame) at the Lidar origin
    pub vel: [f64; 3],

    /// the preintegrated position (global frame) at the Lidar origin
    pub pos: [f64; 3],

    /// the preintegrated rotation (global frame) at the Lidar origin
    pub rot: [f64; 9],

}



impl Default for Pose6D {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::Pose6D::default())
  }
}

impl rosidl_runtime_rs::Message for Pose6D {
  type RmwMsg = super::msg::rmw::Pose6D;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        offset_time: msg.offset_time,
        acc: msg.acc,
        gyr: msg.gyr,
        vel: msg.vel,
        pos: msg.pos,
        rot: msg.rot,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
      offset_time: msg.offset_time,
        acc: msg.acc,
        gyr: msg.gyr,
        vel: msg.vel,
        pos: msg.pos,
        rot: msg.rot,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      offset_time: msg.offset_time,
      acc: msg.acc,
      gyr: msg.gyr,
      vel: msg.vel,
      pos: msg.pos,
      rot: msg.rot,
    }
  }
}


