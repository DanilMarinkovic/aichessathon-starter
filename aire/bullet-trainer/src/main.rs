// Our evaluation network, described in bullet's terms.
//
// The architecture is (768 -> HIDDEN)x2 -> 1 with dual perspective, which is bullet's own
// `examples/simple.rs`. We arrived at the same place independently because both follow the
// same references, so there is no architecture to port -- only four conventions to get right,
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
//  4. Output buckets. `nnue.py` picks its output bank with bullet's own MaterialCount rule,
//     (popcount(occupied) - 2) / ceil(32 / N), and reads the bank as a contiguous row -- which
//     is what `.transpose()` on l1w below produces. OUTPUT_BUCKETS=1 is the control: with one
//     bank MaterialCount always returns zero and the transpose is a no-op on a vector, so the
//     same code path reproduces a pre-bucket network exactly.
//
// Everything else is read from the environment so an architecture sweep is a matter of
// submitting the same binary with different variables, not recompiling six times.

use std::env;

use bullet_lib::{
    game::{inputs::ChessBucketsMirrored, outputs::MaterialCount},
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

/// The king-bucket layout, and it must agree with nnue.py's `_king_buckets` square for square.
///
/// Bullet takes 32 entries covering files a-d and mirrors them onto e-h itself, which is the
/// same board half nnue.py works in: `perspective()` computes `mirror = 7 if file >= 4` and
/// indexes KING_BUCKET with the already-mirrored square, so both sides only ever describe the
/// queenside. Index is rank * 4 + file in both.
///
/// Getting this wrong is silent. The engine would read a different weight bank than the one
/// trained, load cleanly, and simply play worse -- so the screening step after training checks
/// correlation against the labelling engine, which collapses if the banks do not line up.
fn king_buckets(count: usize) -> [usize; 32] {
    let mut table = [0usize; 32];
    if count == 1 {
        return table;
    }
    if count == 32 {
        // One bank per king square: full resolution over the mirrored half, which makes the
        // features (king square, piece, square). Must match nnue.py's `_king_buckets`, where
        // the same case is `rank * 4 + min(file, 3)` over the already-mirrored square.
        for (index, slot) in table.iter_mut().enumerate() {
            *slot = index;
        }
        return table;
    }
    for rank in 0..8 {
        for file in 0..4 {
            let corner = usize::from(file >= 2);
            let bucket = if rank < 2 {
                corner
            } else if rank < 4 {
                2 + corner
            } else {
                4
            };
            table[rank * 4 + file] = bucket.min(count - 1);
        }
    }
    table
}

fn main() {
    // A const generic cannot be read from the environment, so the supported counts are named
    // here and dispatched to one generic body. Anything else fails now, loudly, rather than
    // training a network whose shape nothing downstream can read.
    let output_buckets: usize = env_or("OUTPUT_BUCKETS", 1);
    match output_buckets {
        // Not 1. bullet's builder asserts the output bucket type has more than one bucket, so a
        // one-bank arm cannot be built here at all; the control comes from a network trained
        // before output buckets existed. Caught here rather than left to panic inside bullet
        // after the data has already been read.
        1 => panic!("OUTPUT_BUCKETS=1 is not buildable; bullet requires more than one bucket"),
        2 => train::<2>(),
        4 => train::<4>(),
        8 => train::<8>(),
        other => panic!("OUTPUT_BUCKETS={other} is not one of 1, 2, 4, 8"),
    }
}

fn train<const OUTPUT_BUCKETS: usize>() {
    let hidden: usize = env_or("HIDDEN", 128);
    let batch_size: usize = env_or("BATCH", 16_384);
    let positions: usize = env_or("POSITIONS", 50_000_000);
    let epochs: usize = env_or("EPOCHS", 8);
    let wdl_weight: f32 = env_or("WDL", 0.4);
    let lr_start: f32 = env_or("LR", 0.001);
    let threads: usize = env_or("THREADS", 8);
    let buckets: usize = env_or("BUCKETS", 4);
    // Finishing a schedule that was cut short, rather than starting it again.
    //
    // RESUME names a checkpoint directory; START is the superbatch to continue from, so the
    // learning-rate schedule sees the same absolute index it would have in an uninterrupted
    // run and a step that has already fired stays fired. The wall limit on the gpu partition
    // is what makes this worth having: a run killed at superbatch 362 of 380 costs eight
    // minutes to finish and an hour to repeat, and an arm trained 13% less than the arms it is
    // being compared against is not a comparison.
    let resume = env::var("RESUME").unwrap_or_default();
    let start: usize = env_or("START", 1);

    // Bullet sizes the input layer from the distinct banks the table actually names, so the
    // affine below has to agree with that rather than with what was asked for. The layout
    // saturates at five -- corner and centre on ranks 1-2, the same on 3-4, and everything
    // beyond -- so BUCKETS above five names banks that no king square ever selects. Fail here
    // rather than let bullet assert deep inside its graph builder with a shape mismatch.
    let table = king_buckets(buckets);
    let banks = table.iter().max().copied().unwrap_or(0) + 1;
    assert!(
        banks == buckets,
        "BUCKETS={buckets} but the layout only names {banks} distinct banks; it saturates at 5"
    );
    let data = env::var("DATA").unwrap_or_else(|_| "aire/data/positions.data".to_string());
    // No held-out set, because bullet does not have one.
    //
    // This used to pass TEST_DATA through as a TestDataset and the comment here claimed the
    // reported loss was generalisation rather than fit. It never was. bullet_lib's value
    // trainer answers a populated `test_set` with
    //
    //   Warning: Validation data not currently implemented! Please bother me on discord.
    //
    // and then ignores it -- see crates/bullet_lib/src/value.rs. Every loss this trainer has
    // ever printed is training loss. The wiring is gone rather than left in place, so nothing
    // reads a validation curve that was never computed. Generalisation is measured outside
    // this program, by tools/blindspot.py on held-out positions and by a clock match.
    let net_id = env::var("NET_ID").unwrap_or_else(|_| "chess".to_string());

    // One superbatch is one pass over the data, so `end_superbatch` is the epoch count and the
    // schedule below can be read in epochs rather than in bullet's internal units.
    let batches_per_superbatch = positions / batch_size;

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(table))
        .output_buckets(MaterialCount::<OUTPUT_BUCKETS>)
        // The order here is the order the weights land in raw.bin, and tools/from_bullet.py
        // reads them back in exactly this order. Changing one without the other writes a
        // network that loads cleanly and plays like noise.
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            // Transposed so each bucket's 2 x HIDDEN weights are contiguous, which is the
            // row nnue.py's forward pass takes once per evaluation. On one bucket this is a
            // no-op over the same bytes.
            SavedFormat::id("l1w").round().quantise::<i16>(QB).transpose(),
            SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0 = builder.new_affine("l0", 768 * buckets, hidden);
            let l1 = builder.new_affine("l1", 2 * hidden, OUTPUT_BUCKETS);

            // crelu, deliberately. See note 1 above.
            let stm_hidden = l0.forward(stm_inputs).crelu();
            let ntm_hidden = l0.forward(ntm_inputs).crelu();
            l1.forward(stm_hidden.concat(ntm_hidden)).select(output_buckets)
        });

    let schedule = TrainingSchedule {
        net_id,
        // The sigmoid scale the loss is written in, and it belongs to the data, not to the
        // trainer. The target is 0.6 * sigmoid(score / eval_scale) + 0.4 * game_result, so
        // eval_scale has to be near the spread of the scores actually in the file. Our own
        // labels have a median |score| of 518, and 400 spreads them well. The public Stockfish
        // binpack has a median of 90: at 400 half its positions land inside 0.056 of 0.5, the
        // score half of the target is nearly constant, and the network ends up fitting the
        // game result instead of the evaluation. Three runs on that file calibrated to
        // eval_scale 160-164 against 308-315 for every run on ours, and their correlation with
        // Stockfish's evaluations was 0.87 against 0.97. That is the signature of it.
        //
        // Overridable, because a dataset swap that leaves this at the previous dataset's value
        // is not a comparison of datasets.
        eval_scale: env_or("TRAIN_SCALE", SCALE) as f32,
        steps: TrainingSteps {
            batch_size,
            batches_per_superbatch,
            start_superbatch: start,
            end_superbatch: epochs,
        },
        wdl_scheduler: wdl::ConstantWDL { value: wdl_weight },
        // Drop the rate once, two thirds of the way in. With single-digit epoch counts a
        // step schedule that never fires is the same as a constant one, so the step is
        // derived from the epoch count rather than hard coded at bullet's 18.
        lr_scheduler: lr::StepLR { start: lr_start, gamma: 0.1, step: (epochs * 2 / 3).max(1) },
        // Eight checkpoints, not one at the end.
        //
        // bullet saves when `superbatch % save_rate == 0` or on the final superbatch, so
        // `save_rate: epochs` writes exactly one checkpoint, after the last superbatch. On the
        // gpu partition the wall limit is a SIGKILL, and a run that overruns by a minute then
        // leaves nothing at all -- no weights to convert, no network to play, and the slot
        // gone. It has already happened once, nine superbatches from the end of a 320
        // superbatch schedule.
        //
        // A checkpoint is a few megabytes and takes well under a second, so the insurance is
        // free: the worst case becomes losing the last eighth of the schedule rather than all
        // of it. The slurm script already picks the highest-numbered checkpoint, so a partial
        // run converts and screens exactly like a complete one.
        save_rate: (epochs / 8).max(1),
    };

    let settings = LocalSettings {
        threads,
        test_set: None,
        output_directory: "checkpoints",
        batch_queue_size: 64,
    };

    println!("hidden {hidden}, {buckets} king buckets, {OUTPUT_BUCKETS} output buckets, {epochs} superbatches of {batches_per_superbatch} x {batch_size}");
    println!(
        "data {data}, wdl {wdl_weight}, lr {lr_start}, threads {threads}, \
         train scale {}",
        env_or("TRAIN_SCALE", SCALE)
    );
    if !resume.is_empty() {
        println!("resuming from {resume}, superbatches {start} to {epochs}");
        trainer.load_from_checkpoint(&resume);
    } else if start != 1 {
        println!("START={start} without RESUME would train from scratch on a partial schedule");
        std::process::exit(1);
    }

    // A path ending .binpack is Stockfish's own training format, which bullet reads directly.
    // The rules put no restriction on training data -- only on shipping a network someone else
    // trained -- and a public binpack holds on the order of a billion positions from Stockfish
    // self-play against the 118M our own generation produced.
    if data.ends_with(".binpack") {
        use loader::sfbinpack::{MoveType, PieceType, SfBinpackLoader, TrainingDataEntry};

        // The same filter tools/label.py applies when generating our own set, and for the same
        // reason: a position whose best move is a capture, or where the side to move is in
        // check, is decided by a tactic the search resolves, and fitting a static evaluation to
        // it teaches the wrong thing. `ply >= 16` drops opening positions that repeat across
        // millions of games and would otherwise dominate.
        fn keep(entry: &TrainingDataEntry) -> bool {
            entry.ply >= 16
                && !entry.pos.is_checked(entry.pos.side_to_move())
                && entry.score.unsigned_abs() <= 10000
                && entry.mv.mtype() == MoveType::Normal
                && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None
        }

        let buffer_mb: usize = env_or("BINPACK_BUFFER_MB", 1024);
        println!("reading binpack with a {buffer_mb}MB buffer on {threads} threads");
        let data_loader = SfBinpackLoader::new(data.as_str(), buffer_mb, threads, keep);
        trainer.run(&schedule, &settings, &data_loader);
    } else {
        let data_loader = loader::DirectSequentialDataLoader::new(&[data.as_str()]);
        trainer.run(&schedule, &settings, &data_loader);
    }
}
