use std::hint::black_box;
use std::time::Instant;

use vectorcraft_doc::{Document, Node, NodeKind, hit::{self, HitOptions}};
use vectorcraft_geom::Point;

fn first_path_point(node: &Node) -> Option<Point> {
    if matches!(&node.kind, NodeKind::Path { .. }) {
        return node.geometric_bounds().map(|b| b.center());
    }
    node.children().and_then(|children| children.iter().find_map(|child| first_path_point(child)))
}

fn count_nodes(node: &Node) -> usize {
    1 + node.children().map_or(0, |children| children.iter().map(|child| count_nodes(child)).sum())
}

fn main() {
    let input = std::env::args().nth(1).expect("SVG input path");
    let svg = std::fs::read_to_string(input).expect("read SVG");
    let doc: Document = vectorcraft_svg::import(&svg).expect("import SVG");
    let nodes: usize = doc.layers.iter().map(|layer| count_nodes(layer)).sum();
    let point = doc.layers.iter().find_map(|layer| first_path_point(layer)).expect("a selectable path");
    let iterations = 2_000usize;
    let batches = 21usize;
    let mut per_hit = Vec::with_capacity(batches);
    let mut last = None;
    for _ in 0..batches {
        let start = Instant::now();
        for _ in 0..iterations {
            last = black_box(hit::hit_test(&doc, point, HitOptions::default()));
        }
        per_hit.push(start.elapsed().as_nanos() as f64 / iterations as f64);
    }
    per_hit.sort_by(f64::total_cmp);
    println!("nodes={nodes} hit={:?} median_ns={:.0} min_ns={:.0} max_ns={:.0}", last.map(|h| h.leaf), per_hit[batches / 2], per_hit[0], per_hit[batches - 1]);
}
