extern crate crystal;
extern crate render_bits;

fn main() {
    let bm = crystal::read_map("hidden_ramp.txt").expect("could not read file");
    let mut planes = crystal::Planes::new();
    planes.create_planes(&bm);
    planes.print();

    let mut x: Vec<(&crystal::Point3i, &i32)> = planes.vertex_iter().collect();
    x.sort_by_key(|(_, v)| *v);

    let vertices: Vec<_> = x
        .iter()
        .map(|(plane, _)| render_bits::Vertex {
            position: (plane.x as f32, plane.y as f32, plane.z as f32),
        })
        .collect();
    let normals: Vec<_> = planes
        .dir_iter()
        .map(|dir| render_bits::Normal::from(dir.get_normal::<f32>()))
        .collect();
    assert!(vertices.len() == normals.len());

    let indices: Vec<_> = planes
        .planes_iter()
        .flat_map(|plane| {
            [plane[0], plane[1], plane[2], plane[0], plane[2], plane[3]]
                .iter()
                .map(|y| *y as u16).collect::<Vec<_>>()
        })
        .collect();
    render_bits::render_test(&vertices[..], &normals[..], &indices[..]);
}
