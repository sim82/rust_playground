extern crate crystal;
extern crate render_bits;

fn main() {
    let bm = crystal::read_map("hidden_ramp.txt");
    let mut planes = crystal::Planes::new();
    planes.create_planes(&bm);
    planes.print();
}
