//! The encoder session: open on numr's context, check the GPU, and initialize.

use std::ffi::{CStr, c_int, c_void};

use numr::runtime::cuda::CudaClient;
use numr::runtime::{Device, RuntimeClient};

use super::ffi::{
    self, API_VERSION, CAPS_HEIGHT_MAX, CAPS_HEIGHT_MIN, CAPS_NUM_MAX_BFRAMES, CAPS_PARAM_VER,
    CAPS_SUPPORT_10BIT, CAPS_WIDTH_MAX, CAPS_WIDTH_MIN, CONFIG_VER, DEVICE_TYPE_CUDA, Guid,
    NvEncCapsParam, NvEncHandle, NvEncOpenSessionParams, NvEncPresetConfig, NvEncStatus, Nvenc,
    OPEN_SESSION_VER, PRESET_CONFIG_VER, RC_PARAMS_VER,
};
use super::params;
use crate::decode::PixelFormat;
use crate::encode::EncodeConfig;
use crate::error::EncodeError;

/// An open NVENC session. Dropping it destroys the encoder, so the owner
/// releases every buffer and resource first.
pub(super) struct Session {
    pub(super) nvenc: &'static Nvenc,
    pub(super) handle: NvEncHandle,
    pub(super) client: CudaClient,
}

impl Session {
    /// Makes numr's context current on the calling thread.
    pub(super) fn bind(&self) -> Result<(), EncodeError> {
        bind(&self.client)
    }

    /// Opens a session on numr's context. It creates no second context.
    pub(super) fn open(nvenc: &'static Nvenc, client: &CudaClient) -> Result<Self, EncodeError> {
        bind(client)?;
        let mut open = NvEncOpenSessionParams::zeroed();
        open.version = OPEN_SESSION_VER;
        open.device_type = DEVICE_TYPE_CUDA;
        open.device = client.context().cu_ctx().cast::<c_void>();
        open.api_version = API_VERSION;
        let mut handle: NvEncHandle = std::ptr::null_mut();
        // SAFETY: `open` is fully initialised with zeroed reserved fields, its
        // device is numr's live context, retained by `client`, and `handle` is
        // writable.
        let code = unsafe { (nvenc.open_encode_session_ex)(&mut open, &mut handle) };
        let input = || format!("numr's CUDA context on {}", client.device().name());
        if handle.is_null() {
            ffi::check("nvEncOpenEncodeSessionEx", code, input, String::new)?;
            return Err(EncodeError::Cuda {
                operation: "nvEncOpenEncodeSessionEx",
                reason: "the driver returns success and a null session".to_owned(),
            });
        }
        let session = Self {
            nvenc,
            handle,
            client: client.clone(),
        };
        session.check("nvEncOpenEncodeSessionEx", code, input)?;
        Ok(session)
    }

    /// `Ok` for success, else an error with the driver's last error string.
    pub(super) fn check(
        &self,
        call: &'static str,
        code: NvEncStatus,
        input: impl FnOnce() -> String,
    ) -> Result<(), EncodeError> {
        ffi::check(call, code, input, || self.last_error())
    }

    fn last_error(&self) -> String {
        // SAFETY: `handle` is this session's live encoder.
        let text = unsafe { (self.nvenc.get_last_error_string)(self.handle) };
        if text.is_null() {
            return String::new();
        }
        // SAFETY: the driver returns a NUL-terminated string it owns, valid
        // until the next call on this session. The copy ends before that.
        unsafe { CStr::from_ptr(text) }
            .to_string_lossy()
            .into_owned()
    }

    fn codec_guids(&self) -> Result<Vec<Guid>, EncodeError> {
        let input = || "the session".to_owned();
        let mut count = 0_u32;
        // SAFETY: `handle` is live and `count` is writable.
        let code = unsafe { (self.nvenc.get_encode_guid_count)(self.handle, &mut count) };
        self.check("nvEncGetEncodeGUIDCount", code, input)?;
        let mut guids = vec![Guid::new(0, 0, 0, [0; 8]); count as usize];
        let mut written = 0_u32;
        // SAFETY: `guids` holds `count` writable GUIDs, and `written` is writable.
        let code = unsafe {
            (self.nvenc.get_encode_guids)(self.handle, guids.as_mut_ptr(), count, &mut written)
        };
        self.check("nvEncGetEncodeGUIDs", code, input)?;
        guids.truncate(written as usize);
        Ok(guids)
    }

    fn caps(&self, guid: Guid, query: c_int) -> Result<u32, EncodeError> {
        let mut param = NvEncCapsParam::zeroed();
        param.version = CAPS_PARAM_VER;
        param.caps_to_query = query;
        let mut value: c_int = 0;
        // SAFETY: `handle` is live, `param` is fully initialised, and `value`
        // is writable.
        let code =
            unsafe { (self.nvenc.get_encode_caps)(self.handle, guid, &mut param, &mut value) };
        self.check("nvEncGetEncodeCaps", code, || {
            format!("NV_ENC_CAPS {query}")
        })?;
        Ok(u32::try_from(value).unwrap_or(0))
    }

    /// Refuses what the GPU cannot encode, by name.
    pub(super) fn check_support(&self, config: &EncodeConfig) -> Result<(), EncodeError> {
        let codec = config.codec;
        let guid = params::codec_guid(codec);
        let guids = self.codec_guids()?;
        if !guids.contains(&guid) {
            return Err(EncodeError::CodecUnsupported {
                codec,
                available: params::codec_names(&guids),
            });
        }
        if config.format == PixelFormat::P010 && self.caps(guid, CAPS_SUPPORT_10BIT)? == 0 {
            return Err(EncodeError::BitDepthUnsupported {
                codec,
                format: config.format,
            });
        }
        let maximum = self.caps(guid, CAPS_NUM_MAX_BFRAMES)?;
        if config.b_frames > maximum {
            return Err(EncodeError::BFramesUnsupported {
                codec,
                requested: config.b_frames,
                maximum,
            });
        }
        let minimum = (
            self.caps(guid, CAPS_WIDTH_MIN)?,
            self.caps(guid, CAPS_HEIGHT_MIN)?,
        );
        let maximum = (
            self.caps(guid, CAPS_WIDTH_MAX)?,
            self.caps(guid, CAPS_HEIGHT_MAX)?,
        );
        let fits = (minimum.0..=maximum.0).contains(&config.width)
            && (minimum.1..=maximum.1).contains(&config.height);
        if !fits {
            return Err(EncodeError::SizeOutOfRange {
                codec,
                width: config.width,
                height: config.height,
                minimum,
                maximum,
            });
        }
        Ok(())
    }

    /// Queries the preset, applies the config over it, and initializes the
    /// encoder in synchronous mode.
    pub(super) fn initialize(&self, config: &EncodeConfig) -> Result<(), EncodeError> {
        let codec = params::codec_guid(config.codec);
        let preset = params::preset_guid(config.preset);
        let describe = || {
            format!(
                "{} {}x{} {}, {:?}, {:?}",
                config.codec,
                config.width,
                config.height,
                config.format,
                config.preset,
                config.tuning
            )
        };

        let mut preset_config = NvEncPresetConfig::zeroed();
        preset_config.version = PRESET_CONFIG_VER;
        preset_config.preset_cfg.version = CONFIG_VER;
        preset_config.preset_cfg.rc_params.version = RC_PARAMS_VER;
        // SAFETY: `handle` is live, and `preset_config` is writable with its
        // versions set and every other field zero.
        let code = unsafe {
            (self.nvenc.get_encode_preset_config_ex)(
                self.handle,
                codec,
                preset,
                params::tuning_info(config.tuning),
                &mut preset_config,
            )
        };
        self.check("nvEncGetEncodePresetConfigEx", code, describe)?;

        let mut cfg = preset_config.preset_cfg;
        cfg.version = CONFIG_VER;
        params::apply(config, &mut cfg);
        let mut init = params::initialize_params(config, &mut cfg)?;
        // SAFETY: `handle` is live, `init` is fully initialised with zeroed
        // reserved fields, and `init.encode_config` points to `cfg`, which
        // outlives the call.
        let code = unsafe { (self.nvenc.initialize_encoder)(self.handle, &mut init) };
        self.check("nvEncInitializeEncoder", code, describe)
    }
}

impl Drop for Session {
    /// Destroys the encoder. A driver error here has no caller to reach, so
    /// drop ignores it.
    fn drop(&mut self) {
        let _ = self.bind();
        // SAFETY: `handle` came from `nvEncOpenEncodeSessionEx` and is destroyed
        // once. The owner has released every buffer and resource of it.
        let _ = unsafe { (self.nvenc.destroy_encoder)(self.handle) };
    }
}

fn bind(client: &CudaClient) -> Result<(), EncodeError> {
    client
        .context()
        .bind_to_thread()
        .map_err(|error| EncodeError::Cuda {
            operation: "binding numr's CUDA context to the calling thread",
            reason: error.to_string(),
        })
}
