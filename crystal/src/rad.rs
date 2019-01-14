use super::DisplayWrap;
use super::{Bitmap, BlockMap, Point3, Point3i, Vec3, Vec3i};
use super::{Plane, PlanesSep};
use cgmath::prelude::*;

fn occluded(p0: Point3i, p1: Point3i, solid: &Bitmap) -> bool {
    // 3d bresenham, ripped from http://www.cobrabytes.com/index.php?topic=1150.0

    println!("{} {}", DisplayWrap::from(p0), DisplayWrap::from(p1));

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
            println!("stop {}", DisplayWrap::from(Point3i::new(cx, cy, cz)));
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

pub struct Scene {
    pub planes: PlanesSep,
    bitmap: Box<Bitmap>,
    pub emit: Vec<Vec3>,
}

impl Scene {
    pub fn new(planes: PlanesSep, bitmap: Box<BlockMap>) -> Self {
        Scene {
            emit: vec![Vec3::new(0.2f32, 0.2f32, 0.2f32); planes.num_planes()],
            planes: planes,
            bitmap: bitmap,
        }
    }

    pub fn apply_light(&mut self, pos: Point3, color: Vec3) {
        let ligth_pos = Point3i::new(pos.x as i32, pos.y as i32, pos.z as i32);
        for (i, plane) in self.planes.planes_iter().enumerate() {
            // let trace_pos = Vec3::new(
            //     plane.cell.x as f32,
            //     plane.cell.x as f32,
            //     plane.cell.x as f32,
            // ) + Vec3::new(0.5, 0.5, 0.5)
            //     + plane.dir.get_normal() * 0.5;

            let trace_pos = plane.cell + plane.dir.get_normal();

            if !occluded(ligth_pos, trace_pos, &*self.bitmap) {
                println!("light");
                self.emit[i] = color;
            }
        }
    }
}
