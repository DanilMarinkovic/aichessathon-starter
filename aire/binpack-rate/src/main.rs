//! Decode a prefix of a .binpack and report positions per second.
//!
//!   binpack-rate <file.binpack> [entries]
//!
//! Single threaded on purpose. bullet's SfBinpackLoader decompresses whole chunks on a pool of
//! worker threads, so the number to compare against a training rate is this one multiplied by
//! the thread count, bounded by what the filesystem can deliver. Reporting one thread keeps the
//! measurement honest about which of the two is the limit.

use std::env;
use std::fs::File;
use std::time::Instant;

use sfbinpack::chess::{piecetype::PieceType, r#move::MoveType};
use sfbinpack::{CompressedTrainingDataEntryReader, TrainingDataEntry};

/// The filter in aire/bullet-trainer/src/main.rs, character for character. A rate measured
/// without it would flatter the loader: rejected entries still cost a full decode.
fn keep(entry: &TrainingDataEntry) -> bool {
    entry.ply >= 16
        && !entry.pos.is_checked(entry.pos.side_to_move())
        && entry.score.unsigned_abs() <= 10000
        && entry.mv.mtype() == MoveType::Normal
        && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None
}

fn main() {
    let mut args = env::args().skip(1);
    let path = args.next().expect("usage: binpack-rate <file.binpack> [entries]");
    let target: u64 = args.next().map_or(20_000_000, |n| n.parse().expect("entries must be a number"));

    let file = File::options().read(true).open(&path).expect("cannot open binpack");
    let mut reader = CompressedTrainingDataEntryReader::new(file).expect("cannot read binpack");

    let start = Instant::now();
    let (mut seen, mut kept) = (0u64, 0u64);
    while seen < target && reader.has_next() {
        if keep(&reader.next()) {
            kept += 1;
        }
        seen += 1;
    }
    let elapsed = start.elapsed().as_secs_f64();

    let rate = seen as f64 / elapsed;
    println!("{seen} entries in {elapsed:.1}s");
    println!("  kept          {kept} ({:.1}%)", 100.0 * kept as f64 / seen as f64);
    println!("  one thread    {:.2}M entries/s", rate / 1e6);
    for threads in [8usize, 16] {
        println!("  x{threads:<12} {:.2}M entries/s if decoding scales", rate * threads as f64 / 1e6);
    }
}
