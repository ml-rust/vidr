//! The capture of validation errors, for a device opened with validation on.

use std::ffi::{CStr, c_void};
use std::sync::{Arc, Mutex, PoisonError};

use ash::vk;

use super::error::vk_error;
use crate::error::DecodeError;

/// The Khronos validation layer's name.
const LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

/// Whether the loader lists the Khronos validation layer.
pub(super) fn layer_present(entry: &ash::Entry) -> bool {
    // SAFETY: `entry` is a live loader, and the call reads no caller memory.
    let layers = unsafe { entry.enumerate_instance_layer_properties() };
    layers.is_ok_and(|layers| {
        layers
            .iter()
            .any(|layer| layer.layer_name_as_c_str().is_ok_and(|name| name == LAYER))
    })
}

/// Holds each validation error the layer reports.
pub(super) struct ValidationCapture {
    utils: ash::ext::debug_utils::Instance,
    messenger: vk::DebugUtilsMessengerEXT,
    /// An `Arc`, so the address the messenger holds stays fixed and a move of this type
    /// keeps that pointer valid.
    errors: Arc<Mutex<Vec<String>>>,
}

impl ValidationCapture {
    /// Creates a messenger on `instance` that keeps each message of error severity.
    pub(super) fn new(entry: &ash::Entry, instance: &ash::Instance) -> Result<Self, DecodeError> {
        let utils = ash::ext::debug_utils::Instance::new(entry, instance);
        let errors = Arc::new(Mutex::new(Vec::new()));
        let user_data: *const Mutex<Vec<String>> = Arc::as_ptr(&errors);
        let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(on_message))
            .user_data(user_data.cast_mut().cast());
        // SAFETY: `info` is fully initialised. `user_data` points into the `Arc`, which
        // this type keeps until its drop destroys the messenger.
        let messenger = unsafe { utils.create_debug_utils_messenger(&info, None) }
            .map_err(|result| vk_error("vkCreateDebugUtilsMessengerEXT", result, "the capture"))?;
        Ok(Self {
            utils,
            messenger,
            errors,
        })
    }

    /// Every error so far.
    pub(super) fn errors(&self) -> Vec<String> {
        self.errors
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

impl Drop for ValidationCapture {
    fn drop(&mut self) {
        // SAFETY: this type created the messenger, and destroys it once.
        unsafe {
            self.utils
                .destroy_debug_utils_messenger(self.messenger, None)
        };
    }
}

/// Stores one message. It takes no action that can unwind across the driver.
unsafe extern "system" fn on_message(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    user_data: *mut c_void,
) -> vk::Bool32 {
    if !severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) || user_data.is_null() {
        return vk::FALSE;
    }
    let text = if data.is_null() {
        String::new()
    } else {
        // SAFETY: the layer passes valid callback data for the length of this call.
        let message = unsafe { (*data).p_message };
        if message.is_null() {
            String::new()
        } else {
            // SAFETY: the layer passes a NUL-terminated string.
            unsafe { CStr::from_ptr(message) }
                .to_string_lossy()
                .into_owned()
        }
    };
    // SAFETY: `user_data` points to the capture's mutex, which outlives the messenger.
    let errors = unsafe { &*user_data.cast::<Mutex<Vec<String>>>() };
    errors
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .push(text);
    vk::FALSE
}
