//! The command line: flags, the Ctrl-C handler, printing, and exit codes.
//!
//! Every step of the work lives in [`remaster`].

use std::ffi::OsString;
use std::io::{self, Write as _};
use std::path::PathBuf;
use std::process::{self, ExitCode};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Once, PoisonError};

use clap::Parser;
use vidr::pipeline::PumpControl;

use crate::error::RemasterError;
use crate::execution::{DecoderChoice, EncoderChoice, RuntimeChoice};
use crate::notices::notices;
use crate::publish::remove_temp_directories;
use crate::request::{Codec, ModelId, Request};
use crate::resolution::ResolutionMode;
use crate::run::{Progress, RunOutcome, remaster};

/// The exit code of a complete run and of `--help`.
pub const EXIT_SUCCESS: u8 = 0;
/// The exit code of any refusal or failure, a bad flag included.
pub const EXIT_FAILURE: u8 = 1;
/// The exit code of a run a signal stopped.
pub const EXIT_INTERRUPTED: u8 = 130;

/// What the command prints after a stopped run.
const INTERRUPTED_TEXT: &str = "Interrupted. The source video remains unchanged.";

/// What the help text states under the flags.
const AFTER_HELP: &str = "\
--codec av1 --encoder cpu runs rav1r, with the cpu-av1 feature. rav1r is a fork of rav1e under \
the BSD-2-Clause licence and the Alliance for Open Media Patent License 1.0.
--notices prints the licence notices of every codec in this build.";

/// Remaster a complete video, and size it for practical storage.
#[derive(Debug, Parser)]
#[command(name = "vidr-remaster", version, after_help = AFTER_HELP)]
struct Cli {
    /// Read the source video.
    #[arg(value_name = "INPUT", required_unless_present = "notices")]
    input: Option<PathBuf>,
    /// Select the model (default: purevision).
    #[arg(
        long,
        value_name = "fast|realesrgan|purevision",
        default_value_t = ModelId::default(),
        hide_default_value = true
    )]
    model: ModelId,
    /// Choose another output path, .mp4 or .mkv (default: beside the input).
    #[arg(short, long, value_name = "PATH")]
    output: Option<PathBuf>,
    /// Set the model root (default: $VIDR_MODELS_DIR).
    #[arg(long, value_name = "DIR")]
    models_dir: Option<PathBuf>,
    /// Set encoder quality, 1 to 51, lower is better (default: 24).
    #[arg(
        long,
        value_name = "1-51",
        default_value_t = crate::request::DEFAULT_QUALITY,
        hide_default_value = true
    )]
    quality: u32,
    /// Select the output codec (default: hevc).
    #[arg(
        long,
        value_name = "hevc|h264|av1",
        default_value_t = Codec::default(),
        hide_default_value = true
    )]
    codec: Codec,
    /// Select the tensor runtime (default: auto).
    #[arg(
        long,
        value_name = "auto|cuda|wgpu|cpu",
        default_value_t = RuntimeChoice::default(),
        hide_default_value = true
    )]
    runtime: RuntimeChoice,
    /// Select the decoder (default: auto, hardware then CPU).
    #[arg(
        long,
        value_name = "auto|nvdec|vulkan|cpu",
        default_value_t = DecoderChoice::default(),
        hide_default_value = true
    )]
    decoder: DecoderChoice,
    /// Select the encoder (default: auto, hardware then CPU).
    #[arg(
        long,
        value_name = "auto|nvenc|cpu",
        default_value_t = EncoderChoice::default(),
        hide_default_value = true
    )]
    encoder: EncoderChoice,
    /// Set the path of Cisco's OpenH264 binary (default: none).
    #[arg(long, value_name = "PATH")]
    openh264_library: Option<PathBuf>,
    /// Replace an existing output file (default: off).
    #[arg(long)]
    overwrite: bool,
    /// Set final shorter edge (default: auto, minimum: 480p).
    #[arg(
        long,
        value_name = "auto|original|2x|480|720|1080|1440|2160|4320",
        default_value_t = ResolutionMode::default(),
        hide_default_value = true
    )]
    resolution: ResolutionMode,
    /// Resize this enhanced video, with no AI pass (default: none).
    #[arg(long, value_name = "PATH", conflicts_with = "force_ai")]
    from_upscaled: Option<PathBuf>,
    /// Run AI even when a matching 2x output exists (default: off).
    #[arg(long)]
    force_ai: bool,
    /// Print the licence notices of this build (default: off).
    #[arg(long, conflicts_with = "input")]
    notices: bool,
}

/// What a command line asks for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    /// Remaster one video.
    Run(Request),
    /// Print the licence notices of this build.
    Notices,
}

impl From<Cli> for Command {
    fn from(cli: Cli) -> Self {
        let Some(source) = cli.input else {
            return Self::Notices;
        };
        Self::Run(Request {
            source,
            model: cli.model,
            output: cli.output,
            quality: cli.quality,
            codec: cli.codec,
            runtime: cli.runtime,
            decoder: cli.decoder,
            encoder: cli.encoder,
            openh264_library: cli.openh264_library,
            overwrite: cli.overwrite,
            resolution: cli.resolution,
            from_upscaled: cli.from_upscaled,
            force_ai: cli.force_ai,
            models_dir: cli.models_dir,
        })
    }
}

/// Parses a command line, program name first.
///
/// # Errors
///
/// [`clap::Error`] for a bad flag, and for `--help` and `--version`.
pub fn parse_command(args: impl IntoIterator<Item = OsString>) -> Result<Command, clap::Error> {
    Cli::try_parse_from(args).map(Command::from)
}

/// Parses a command line, program name first, into a request.
///
/// # Errors
///
/// [`clap::Error`] for a bad flag, for `--help` and `--version`, and for
/// `--notices`, which asks for no run.
pub fn parse_request(args: impl IntoIterator<Item = OsString>) -> Result<Request, clap::Error> {
    use clap::CommandFactory as _;

    match parse_command(args)? {
        Command::Run(request) => Ok(request),
        Command::Notices => Err(Cli::command().error(
            clap::error::ErrorKind::MissingRequiredArgument,
            "--notices asks for no run. Pass INPUT without --notices.",
        )),
    }
}

/// The help text of each flag: flag name, then its line.
pub fn flag_help() -> Vec<(String, String)> {
    use clap::CommandFactory as _;

    Cli::command()
        .get_arguments()
        .filter(|argument| !matches!(argument.get_id().as_str(), "help" | "version"))
        .map(|argument| {
            (
                argument.get_id().to_string(),
                argument
                    .get_help()
                    .map(ToString::to_string)
                    .unwrap_or_default(),
            )
        })
        .collect()
}

/// The exit code of a run's result.
pub fn exit_code(result: &Result<RunOutcome, RemasterError>) -> u8 {
    match result {
        Ok(RunOutcome::Stopped { .. }) => EXIT_INTERRUPTED,
        // The run promises the output and its sidecar, and the sidecar is absent.
        Ok(RunOutcome::OutputWithoutSidecar { .. }) => EXIT_FAILURE,
        Ok(_) => EXIT_SUCCESS,
        Err(_) => EXIT_FAILURE,
    }
}

/// One run's stop state.
///
/// The first press asks the run to stop. The pump cannot interrupt a call into
/// the driver, so a second press removes the temporary output and ends the
/// process, as a last resort.
#[derive(Debug)]
struct Interrupt {
    control: Arc<PumpControl>,
    pressed: AtomicBool,
}

impl Interrupt {
    fn new(control: Arc<PumpControl>) -> Self {
        Self {
            control,
            pressed: AtomicBool::new(false),
        }
    }

    /// Takes one press and returns `true` when it is not the first.
    fn press(&self) -> bool {
        self.control.request_stop();
        self.pressed.swap(true, Ordering::Relaxed)
    }
}

/// The run the signal handler acts on. The handler lives for the process, and
/// `main_with_args` can run more than once in it.
static CURRENT: Mutex<Option<Arc<Interrupt>>> = Mutex::new(None);

/// The handler `ctrlc` calls.
///
/// `ctrlc`'s own signal handler only posts a semaphore. Its `ctrl-c` thread waits on
/// that semaphore, then calls this function. So this function runs on a normal
/// thread outside signal context, and locks, allocation, and file removal are all
/// allowed here. A panic stops that thread for good, so nothing here panics.
fn on_signal() {
    // An `Option` has no partial state, so a poisoned lock is still correct.
    let current = CURRENT
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    if let Some(interrupt) = current
        && interrupt.press()
    {
        // `process::exit` skips `Drop`, so the temporary directory goes first.
        remove_temp_directories();
        process::exit(i32::from(EXIT_INTERRUPTED));
    }
}

/// Points the handler at `control`, and installs the handler the first time.
fn watch_signals(control: Arc<PumpControl>) {
    static INSTALL: Once = Once::new();
    *CURRENT.lock().unwrap_or_else(PoisonError::into_inner) =
        Some(Arc::new(Interrupt::new(control)));
    INSTALL.call_once(|| {
        if let Err(error) = ctrlc::set_handler(on_signal) {
            eprint_line(&format!(
                "Cannot install the Ctrl-C handler: {error}. Ctrl-C now ends the process, and \
                 the temporary output stays beside the output."
            ));
        }
    });
}

/// Prints a line. A closed pipe does not end the run, and is not an error.
fn print_line(text: &str) {
    let _ = writeln!(io::stdout(), "{text}");
}

fn eprint_line(text: &str) {
    let _ = writeln!(io::stderr(), "{text}");
}

/// Prints each event, keeping the frame count on one line as the PoC does.
fn printer() -> impl FnMut(Progress) + Send {
    let mut line_open = false;
    move |event| {
        if let Progress::Frames { .. } = event {
            let mut stdout = io::stdout();
            let _ = write!(stdout, "{event}\r");
            let _ = stdout.flush();
            line_open = true;
            return;
        }
        if line_open {
            print_line("");
            line_open = false;
        }
        if matches!(event, Progress::ReportFailed { .. }) {
            eprint_line(&event.to_string());
        } else {
            print_line(&event.to_string());
        }
    }
}

/// Runs the command on `args`, program name first, and returns its exit code.
pub fn run_command(args: impl IntoIterator<Item = OsString>) -> u8 {
    let request = match parse_command(args) {
        Ok(Command::Run(request)) => request,
        Ok(Command::Notices) => {
            print_line(&notices());
            return EXIT_SUCCESS;
        }
        Err(error) => {
            let _ = error.print();
            return if error.use_stderr() {
                EXIT_FAILURE
            } else {
                EXIT_SUCCESS
            };
        }
    };
    let control = Arc::new(PumpControl::new());
    watch_signals(Arc::clone(&control));
    let result = remaster(request, printer(), &control);
    match &result {
        Ok(RunOutcome::Stopped { .. }) => print_line(INTERRUPTED_TEXT),
        Ok(RunOutcome::OutputWithoutSidecar { missing, .. }) => {
            eprint_line(&missing.to_string());
        }
        Ok(_) => {}
        Err(error) => eprint_line(&error.to_string()),
    }
    exit_code(&result)
}

/// The `vidr-remaster` command: exit 0 for a complete run, 1 for a refusal, a
/// failure, or an output whose sidecar did not land, and 130 for a run a signal
/// stopped. A report that does not write after the publish still exits 0.
pub fn main_with_args(args: impl IntoIterator<Item = OsString>) -> ExitCode {
    ExitCode::from(run_command(args))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::publish::MissingSidecar;

    #[test]
    fn the_first_press_stops_and_the_second_asks_to_exit() {
        let control = Arc::new(PumpControl::new());
        let interrupt = Interrupt::new(Arc::clone(&control));
        assert!(!control.stop_requested());
        assert!(!interrupt.press());
        assert!(control.stop_requested());
        assert!(interrupt.press());
    }

    #[test]
    fn exit_codes_follow_the_outcome() {
        let complete = Ok(RunOutcome::Complete {
            output: PathBuf::from("out.mp4"),
            report: Some(PathBuf::from("report.json")),
        });
        assert_eq!(exit_code(&complete), 0);
        let no_report = Ok(RunOutcome::Complete {
            output: PathBuf::from("out.mp4"),
            report: None,
        });
        assert_eq!(exit_code(&no_report), 0);
        let no_sidecar = Ok(RunOutcome::OutputWithoutSidecar {
            missing: MissingSidecar {
                output: PathBuf::from("out.mp4"),
                sidecar: PathBuf::from("out.mp4.upscale.json"),
                reason: "disk gone".to_owned(),
            },
            report: None,
        });
        assert_eq!(exit_code(&no_sidecar), 1);
        let stopped = Ok(RunOutcome::Stopped { report: None });
        assert_eq!(exit_code(&stopped), 130);
        assert_eq!(exit_code(&Err(RemasterError::NoGpu)), 1);
    }
}
