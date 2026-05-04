#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};


#[link(name = "fast_lio_sam_g__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__fast_lio_sam_g__msg__Pose6D() -> *const std::ffi::c_void;
}

#[link(name = "fast_lio_sam_g__rosidl_generator_c")]
extern "C" {
    fn fast_lio_sam_g__msg__Pose6D__init(msg: *mut Pose6D) -> bool;
    fn fast_lio_sam_g__msg__Pose6D__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<Pose6D>, size: usize) -> bool;
    fn fast_lio_sam_g__msg__Pose6D__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Pose6D>);
    fn fast_lio_sam_g__msg__Pose6D__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<Pose6D>, out_seq: *mut rosidl_runtime_rs::Sequence<Pose6D>) -> bool;
}

// Corresponds to fast_lio_sam_g__msg__Pose6D
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// the preintegrated Lidar states at the time of IMU measurements in a frame

#[repr(C)]
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
    unsafe {
      let mut msg = std::mem::zeroed();
      if !fast_lio_sam_g__msg__Pose6D__init(&mut msg as *mut _) {
        panic!("Call to fast_lio_sam_g__msg__Pose6D__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for Pose6D {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { fast_lio_sam_g__msg__Pose6D__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { fast_lio_sam_g__msg__Pose6D__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { fast_lio_sam_g__msg__Pose6D__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for Pose6D {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for Pose6D where Self: Sized {
  const TYPE_NAME: &'static str = "fast_lio_sam_g/msg/Pose6D";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__fast_lio_sam_g__msg__Pose6D() }
  }
}


