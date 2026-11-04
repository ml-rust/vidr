//! A build refuses, by name, each model whose feature is off.

use vidr_remaster::{ModelId, catalog_entry, require_model};

fn built(model: ModelId) -> bool {
    match model {
        ModelId::Fast => cfg!(feature = "model-srvgg"),
        ModelId::RealEsrgan | ModelId::PureVision => cfg!(feature = "model-rrdb"),
    }
}

#[test]
fn each_model_follows_its_feature() {
    for model in ModelId::ALL {
        match require_model(model) {
            Ok(()) => assert!(built(model), "{model} passes with its feature off"),
            Err(error) => {
                assert!(
                    !built(model),
                    "{model} is refused with its feature on: {error}"
                );
                let message = error.to_string();
                assert!(message.contains(model.as_str()), "{message}");
                assert!(message.contains(catalog_entry(model).feature), "{message}");
            }
        }
    }
}

#[cfg(not(any(feature = "model-srvgg", feature = "model-rrdb")))]
#[test]
fn the_default_build_refuses_every_model() {
    for model in ModelId::ALL {
        assert!(require_model(model).is_err(), "{model} passes");
    }
    let message = require_model(ModelId::PureVision)
        .expect_err("refused")
        .to_string();
    assert!(message.contains("CC BY-NC-SA 4.0"), "{message}");
}
