//! The removal a second signal runs before `process::exit`, which skips `Drop`.
//!
//! This binary holds one test. The function removes every live temporary directory
//! in its process, so a shared process breaks other tests' directories.

use std::fs;

use vidr_remaster::{TempOutput, remove_temp_directories};

#[test]
fn the_removal_takes_every_live_directory_and_a_later_drop_does_no_harm() {
    let root = tempfile::tempdir().expect("tempdir");
    let first = TempOutput::beside(&root.path().join("first.mp4")).expect("temp");
    let second = TempOutput::beside(&root.path().join("second.mp4")).expect("temp");
    fs::write(first.prepared(), b"partial").expect("write");
    fs::write(second.prepared(), b"partial").expect("write");
    assert!(first.directory().is_dir());
    assert!(second.directory().is_dir());

    remove_temp_directories();

    assert!(!first.directory().exists());
    assert!(!second.directory().exists());
    assert_eq!(fs::read_dir(root.path()).expect("read_dir").count(), 0);

    // The registry is empty now, so a second call has nothing to do.
    remove_temp_directories();
    // The main thread can still reach this drop before the process ends.
    drop(first);
    drop(second);
    assert_eq!(fs::read_dir(root.path()).expect("read_dir").count(), 0);
}
