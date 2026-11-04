//! The model catalog: directory names, resolving, error text, and no network.

use std::ffi::OsStr;
use std::fs;
use std::path::Path;

use vidr_remaster::{
    CatalogError, ModelId, Network, catalog_entry, licence_notice, models_root, resolve_model_dir,
};

const LICENCE_TEXT: &str = "CC BY-NC-SA 4.0";

/// The directory names and URLs of the PoC's `model_catalog.py`.
const POC_CATALOG: [(ModelId, &str, &str); 3] = [
    (
        ModelId::Fast,
        "realesr-general-x4v3",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    ),
    (
        ModelId::RealEsrgan,
        "RealESRGAN_x2plus",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    ),
    (
        ModelId::PureVision,
        "2x_PureVision",
        "https://github.com/limitlesslab/AI-upscaling-models/releases/download/PureVision/2x_PureVision.pth",
    ),
];

fn with_checkpoint(root: &Path, directory: &str) {
    let path = root.join(directory);
    fs::create_dir(&path).expect("mkdir");
    fs::write(path.join("weights.pth"), b"").expect("write checkpoint");
}

#[test]
fn entries_match_the_poc_catalog() {
    for (model, directory, url) in POC_CATALOG {
        let entry = catalog_entry(model);
        assert_eq!(entry.model, model);
        assert_eq!(entry.directory, directory);
        assert_eq!(entry.url, url);
    }
    assert_eq!(catalog_entry(ModelId::Fast).network, Network::SrVggCompact);
    assert_eq!(catalog_entry(ModelId::RealEsrgan).network, Network::Rrdb);
    assert_eq!(catalog_entry(ModelId::PureVision).network, Network::Rrdb);
}

#[test]
fn a_present_directory_resolves() {
    let root = tempfile::tempdir().expect("tempdir");
    for (model, directory, _) in POC_CATALOG {
        with_checkpoint(root.path(), directory);
        let resolved = resolve_model_dir(root.path(), model).expect("resolved");
        assert_eq!(resolved, root.path().join(directory));
    }
}

#[test]
fn a_missing_directory_names_the_path_and_the_url() {
    let root = tempfile::tempdir().expect("tempdir");
    for (model, directory, url) in POC_CATALOG {
        let error = resolve_model_dir(root.path(), model).expect_err("refused");
        assert!(
            matches!(error, CatalogError::MissingDirectory { .. }),
            "{error}"
        );
        let message = error.to_string();
        let expected = root.path().join(directory);
        assert!(
            message.contains(&expected.display().to_string()),
            "{message}"
        );
        assert!(message.contains(url), "{message}");
        assert!(message.contains(model.as_str()), "{message}");
    }
}

#[test]
fn a_directory_with_no_checkpoint_names_the_url() {
    let root = tempfile::tempdir().expect("tempdir");
    let (model, directory, url) = POC_CATALOG[0];
    fs::create_dir(root.path().join(directory)).expect("mkdir");
    fs::write(root.path().join(directory).join("notes.txt"), b"").expect("write");
    let error = resolve_model_dir(root.path(), model).expect_err("refused");
    assert!(
        matches!(error, CatalogError::MissingCheckpoint { .. }),
        "{error}"
    );
    assert!(error.to_string().contains(url), "{error}");
}

#[test]
fn only_purevision_states_the_noncommercial_licence() {
    let root = tempfile::tempdir().expect("tempdir");
    for (model, _, _) in POC_CATALOG {
        let message = resolve_model_dir(root.path(), model)
            .expect_err("refused")
            .to_string();
        let is_purevision = model == ModelId::PureVision;
        assert_eq!(message.contains(LICENCE_TEXT), is_purevision, "{message}");
        assert_eq!(licence_notice(model).is_some(), is_purevision);
    }
    let notice = licence_notice(ModelId::PureVision).expect("notice");
    assert!(notice.contains(LICENCE_TEXT), "{notice}");
}

#[test]
fn the_flag_wins_over_the_environment_and_empty_counts_as_unset() {
    let (flag, environment) = (Path::new("/flag"), OsStr::new("/environment"));
    let root = |flag, environment| models_root(flag, environment).expect("root");
    assert_eq!(root(Some(flag), Some(environment)), flag);
    assert_eq!(root(None, Some(environment)), Path::new("/environment"));
    assert_eq!(
        root(Some(Path::new("")), Some(environment)),
        Path::new("/environment")
    );
    for error in [
        models_root(None, None),
        models_root(Some(Path::new("")), Some(OsStr::new(""))),
    ] {
        let message = error.expect_err("refused").to_string();
        assert!(message.contains("--models-dir"), "{message}");
        assert!(message.contains("VIDR_MODELS_DIR"), "{message}");
    }
}

/// A missing model creates nothing, and neither the module nor the manifest
/// names a network client.
#[test]
fn resolving_touches_no_network_and_creates_nothing() {
    let root = tempfile::tempdir().expect("tempdir");
    for (model, _, _) in POC_CATALOG {
        assert!(resolve_model_dir(root.path(), model).is_err());
    }
    let left = fs::read_dir(root.path()).expect("list").count();
    assert_eq!(left, 0, "resolving wrote into the model root");
    let source = include_str!("../src/catalog.rs");
    let manifest = include_str!("../Cargo.toml");
    for client in ["std::net", "TcpStream", "reqwest", "ureq", "hyper", "curl"] {
        assert!(!source.contains(client), "catalog.rs names {client}");
        assert!(!manifest.contains(client), "Cargo.toml names {client}");
    }
}
