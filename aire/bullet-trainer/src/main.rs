// Our evaluation network, described in bullet's terms.
//
// The architecture is (768 -> HIDDEN)x2 -> 1 with dual perspective, which is bullet's own
// `examples/simple.rs`. We arrived at the same place independently because both follow the
// same references, so there is no architecture to port -- only three conventions to get right,
// each of which fails silently rather than loudly if it is wrong.
//
//  1. Activation. `nnue.py` clamps the accumulator to QA and multiplies. It does not square.
//     bullet's example uses screlu, which squares and then divides by QA to compensate. Train
//     with screlu and the engine reads its own weights with the wrong arithmetic: no error,
//     just a network that evaluates badly. So: crelu.
//
//  2. Score perspective. Our labels are side-to-move relative, raw UCI centipawns. bullet's
//     text parser negates the score and flips the result when black is to move, which means
//     it wants white-relative input. tools/to_bullet.py does that flip; this file relies on it
//     having been done.
//
//  3. WDL weight. tools/train.py fits `0.6 * sigmoid(cp/400) + 0.4 * result`, so the default
//     here is 0.4 and the first bullet run changes the optimiser without changing what is
//     being fitted. bullet's own example uses 0.75, which is worth measuring afterwards, as
//     one variable rather than two.
//
// Everything else is read from the environment so an architecture sweep is a matter of
// submitting the same binary with different variables, not recompiling six times.

use std::env;

use bullet_lib::{
    game::inputs::Chess768,
    nn::optimiser::AdamW,
    trainer::{
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::{loader, ValueTrainerBuilder},
};

// Must match nnue.py. The engine reads these from the weights file, but the numbers it will
// find there are the ones written here.
const SCALE: i32 = 400;
const QA: i16 = 255;
const QB: i16 = 64;

fn env_or<T: std::str::FromStr>(name: &str, fallback: T) -> T {
    env::var(name).ok().and_then(|value| value.parse().ok()).unwrap_or(fallback)
}

fn main() {
    let hidden: usize = env_or("HIDDEN", 128);
    let batch_size: usize = env_or("BATCH", 16_384);
    let positions: usize = env_or("POSITIONS", 50_000_000);
    let epochs: usize = env_or("EPOCHS", 8);
    let wdl_weight: f32 = env_or("WDL", 0.4);
    let lr_start: f32 = env_or("LR", 0.001);
    let threads: usize = env_or("THREADS", 8);
    let data = env::var("DATA").unwrap_or_else(|_| "aire/data/positions.data".to_string());
    let net_id = env::var("NET_ID").unwrap_or_else(|_| "chess".to_string());

    // One superbatch is one pass over the data, so `end_superbatch` is the epoch count and the
    // schedule below can be read in epochs rather than in bullet's internal units.
    let batches_per_superbatch = positions / batch_size;

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(Chess768)
        // The order here is the order the weights land in raw.bin, and tools/from_bullet.py
        // reads them back in exactly this order. Changing one without the other writes a
        // network that loads cleanly and plays like noise.
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w").round().quantise::<i16>(QB),
            SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs| {
            let l0 = builder.new_affine("l0", 768, hidden);
            let l1 = builder.new_affine("l1", 2 * hidden, 1);

            // crelu, deliberately. See note 1 above.
            let stm_hidden = l0.forward(stm_inputs).crelu();
            let ntm_hidden = l0.forward(ntm_inputs).crelu();
            l1.forward(stm_hidden.concat(ntm_hidden))
        });

    let schedule = TrainingSchedule {
        net_id,
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size,
            batches_per_superbatch,
            start_superbatch: 1,
            end_superbatch: epochs,
        },
        wdl_scheduler: wdl::ConstantWDL { value: wdl_weight },
        // Drop the rate once, two thirds of the way in. With single-digit epoch counts a
        // step schedule that never fires is the same as a constant one, so the step is
        // derived from the epoch count rather than hard coded at bullet's 18.
        lr_scheduler: lr::StepLR { start: lr_start, gamma: 0.1, step: (epochs * 2 / 3).max(1) },
        save_rate: epochs,
    };

    let settings = LocalSettings {
        threads,
        test_set: None,
        output_directory: "checkpoints",
        batch_queue_size: 64,
    };

    let data_loader = loader::DirectSequentialDataLoader::new(&[data.as_str()]);

    println!("hidden {hidden}, {epochs} superbatches of {batches_per_superbatch} x {batch_size}");
    println!("data {data}, wdl {wdl_weight}, lr {lr_start}, threads {threads}");

    trainer.run(&schedule, &settings, &data_loader);
}
