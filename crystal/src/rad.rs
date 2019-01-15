extern crate serde;

extern crate serde_json;

use super::DisplayWrap;
use super::{Bitmap, BlockMap, Point3, Point3i, Vec3, Vec3i};
use super::{Dir, Plane, PlanesSep};
use cgmath::prelude::*;

fn occluded(p0: Point3i, p1: Point3i, solid: &Bitmap) -> bool {
    // 3d bresenham, ripped from http://www.cobrabytes.com/index.php?topic=1150.0

    // println!("{} {}", DisplayWrap::from(p0), DisplayWrap::from(p1));

    let mut x0 = p0.x;
    let mut y0 = p0.y;
    let mut z0 = p0.z;

    let mut x1 = p1.x;
    let mut y1 = p1.y;
    let mut z1 = p1.z;

    //'steep' xy Line, make longest delta x plane
    let swap_xy = (y1 - y0).abs() > (x1 - x0).abs();
    if swap_xy {
        std::mem::swap(&mut x0, &mut y0);
        std::mem::swap(&mut x1, &mut y1);
    }

    // do same for xz
    let swap_xz = (z1 - z0).abs() > (x1 - x0).abs();
    if swap_xz {
        std::mem::swap(&mut x0, &mut z0);
        std::mem::swap(&mut x1, &mut z1);
    }

    // delta is Length in each plane
    let delta_x = (x1 - x0).abs();
    let delta_y = (y1 - y0).abs();
    let delta_z = (z1 - z0).abs();

    // drift controls when to step in 'shallow' planes
    // starting value keeps Line centred
    let mut drift_xy = delta_x / 2;
    let mut drift_xz = delta_x / 2;

    // direction of line
    let step_x = if x0 > x1 { -1 } else { 1 };
    let step_y = if y0 > y1 { -1 } else { 1 };
    let step_z = if z0 > z1 { -1 } else { 1 };

    // starting point
    let mut y = y0;
    let mut z = z0;

    // step through longest delta (which we have swapped to x)
    let mut x = x0;
    while x != x1 {
        // copy position
        let mut cx = x;
        let mut cy = y;
        let mut cz = z;

        // unswap (in reverse)
        if swap_xz {
            std::mem::swap(&mut cx, &mut cz);
        }

        if swap_xy {
            std::mem::swap(&mut cx, &mut cy);
        }

        // passes through this point
        // debugmsg(":" + cx + ", " + cy + ", " + cz)
        if solid.get(Point3i::new(cx, cy, cz)) {
            // println!("stop {}", DisplayWrap::from(Point3i::new(cx, cy, cz)));
            return true;
        }
        // update progress in other planes
        drift_xy = drift_xy - delta_y;
        drift_xz = drift_xz - delta_z;

        // step in y plane
        if drift_xy < 0 {
            y = y + step_y;
            drift_xy = drift_xy + delta_x;
        }

        // same in z
        if drift_xz < 0 {
            z = z + step_z;
            drift_xz = drift_xz + delta_x;
        }

        x += step_x;
    }

    // return false;
    false
}

fn normal_cull(pl1: &Plane, pl2: &Plane) -> bool {
    let d1 = pl1.dir;
    let d2 = pl2.dir;

    let p1 = pl1.cell;
    let p2 = pl2.cell;

    p1 == p2
        || d1 == d2
        || (d1 == Dir::XyNeg && d2 == Dir::XyPos && p1.z < p2.z)
        || (d1 == Dir::XyPos && d2 == Dir::XyNeg && p1.z > p2.z)
        || (d1 == Dir::YzNeg && d2 == Dir::YzPos && p1.x < p2.x)
        || (d1 == Dir::YzPos && d2 == Dir::YzNeg && p1.x > p2.x)
        || (d1 == Dir::ZxNeg && d2 == Dir::ZxPos && p1.y < p2.y)
        || (d1 == Dir::ZxPos && d2 == Dir::ZxNeg && p1.y > p2.y)
}

fn setup_formfactors(planes: &PlanesSep, bitmap: &BlockMap) -> Vec<(usize, usize, f32)> {
    let planes = planes.planes_iter().collect::<Vec<&Plane>>();
    println!("num planes: {}", planes.len());
    let mut ffs = Vec::new();
    for (i, plane1) in planes.iter().enumerate() {
        let norm1 = plane1.dir.get_normal::<i32>();
        let norm1f = Vec3::new(norm1.x as f32, norm1.y as f32, norm1.z as f32);
        let p1f = Vec3::new(
            plane1.cell.x as f32,
            plane1.cell.y as f32,
            plane1.cell.z as f32,
        );
        // println!("{}", i);
        for j in 0..i {
            let plane2 = planes[j];
            let norm2 = plane2.dir.get_normal::<i32>();
            let norm2f = Vec3::new(norm2.x as f32, norm2.y as f32, norm2.z as f32);
            let p2f = Vec3::new(
                plane2.cell.x as f32,
                plane2.cell.y as f32,
                plane2.cell.z as f32,
            );
            if normal_cull(plane1, plane2) {
                // println!("normal_cull");
                continue;
            }

            let dn = (p1f - p2f).normalize();
            let d2 = (p1f - p2f).magnitude2(); // uhm, will the compiler optimize the two calls?

            // std::cout << p1_3d << " " << p2_3d << " " << dn << "\n";
            //.norm();
            let ff1 = 0.0f32.max(cgmath::dot(norm1f, Vec3::zero() - dn));
            let ff2 = 0.0f32.max(cgmath::dot(norm2f, dn));

            let ff = (ff1 * ff2) / (3.1415 * d2);
            let dist_cull = ff < 5e-6;

            if !dist_cull && !occluded(plane1.cell + norm1, plane2.cell + norm2, bitmap) {
                ffs.push((i, j, ff));
                ffs.push((j, i, ff));
            }
        }
    }
    println!("num ffs: {}", ffs.len());

    let file = std::fs::File::create("ffs.json").unwrap();
    serde_json::to_writer(file, &ffs);
    panic!("panic");
    ffs
}

pub struct Scene {
    pub planes: PlanesSep,
    bitmap: Box<Bitmap>,
    pub emit: Vec<Vec3>,
    // pub ff: Vec<(usize, usize, f32)>,
}

impl Scene {
    pub fn new(planes: PlanesSep, bitmap: Box<BlockMap>) -> Self {
        Scene {
            emit: vec![Vec3::new(0.2f32, 0.2f32, 0.2f32); planes.num_planes()],
            // ff: setup_formfactors(&planes, &(*bitmap)),
            planes: planes,
            bitmap: bitmap,
        }
    }

    pub fn apply_light(&mut self, pos: Point3, color: Vec3) {
        let light_pos = Point3i::new(pos.x as i32, pos.y as i32, pos.z as i32);
        for (i, plane) in self.planes.planes_iter().enumerate() {
            // let trace_pos = Vec3::new(
            //     plane.cell.x as f32,
            //     plane.cell.x as f32,
            //     plane.cell.x as f32,
            // ) + Vec3::new(0.5, 0.5, 0.5)
            //     + plane.dir.get_normal() * 0.5;

            let trace_pos = plane.cell + plane.dir.get_normal();

            let mut d = (pos
                - Point3::new(trace_pos.x as f32, trace_pos.y as f32, trace_pos.z as f32))
            .normalize();

            // normalize: make directional light
            let len = d.magnitude();
            d /= len;
            let dot = cgmath::dot(d, plane.dir.get_normal());

            // if (dot > 0)
            // {
            //     emit_rgb_[i] += (p.col_diff() * light_color) * dot * (5 / (2 * 3.1415f * len * len));
            // }

            self.emit[i] = Vec3::zero(); //new(0.2, 0.2, 0.2);
            if !occluded(light_pos, trace_pos, &*self.bitmap) {
                // println!("light");
                self.emit[i] += color * dot * (5f32 / (2f32 * 3.1415f32 * len * len));
            }
        }
    }
}
