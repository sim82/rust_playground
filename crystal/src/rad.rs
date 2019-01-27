use image::ImageBuffer;
use serde::ser::{Serialize, SerializeStruct, Serializer};
// use self::simd::x86::avx::*;
// use self::simd::x86::sse3::*;
#[allow(unused_imports)]
use super::{Bitmap, BlockMap, DisplayWrap, Point3, Point3i, Vec3, Vec3i};
use super::{Dir, Plane, PlanesSep};
use cgmath::prelude::*;
use packed_simd::{f32x16, f32x2, f32x4, f32x8};

use std::cmp::Ordering;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::time::Instant;
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

fn setup_formfactors_single(planes: &PlanesSep, bitmap: &BlockMap) -> Vec<(u32, u32, f32)> {
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
                ffs.push((i as u32, j as u32, ff));
            }
        }
    }

    ffs
}

fn setup_formfactors(planes: &PlanesSep, bitmap: &BlockMap) -> Vec<(u32, u32, f32)> {
    let filename = "ffs3.bin";
    let version = "v2";
    if let Ok(f) = std::fs::File::open(filename) {
        println!("read from {}", filename);
        let (file_version, ffs): (String, Vec<(u32, u32, f32)>) =
            bincode::deserialize_from(BufReader::new(f)).unwrap();

        if file_version == version {
            println!("done");
            return ffs;
        }
        println!("wrong version");
    }

    let mut ffs = setup_formfactors_single(planes, bitmap);

    println!("num ffs: {}", ffs.len());

    let mut ffs2 = ffs.iter().map(|(i, j, ff)| (*j, *i, *ff)).collect();

    ffs.append(&mut ffs2);

    ffs.sort_unstable_by(
        |l: &(u32, u32, f32), r: &(u32, u32, f32)| match l.0.cmp(&r.0) {
            Ordering::Equal => l.1.cmp(&r.1),
            Ordering::Less => Ordering::Less,
            Ordering::Greater => Ordering::Greater,
        },
    );
    // {
    //     let file = std::fs::File::create("test.json").unwrap();

    //     serde_json::to_writer(BufWriter::new(file), &ffs);
    // }
    println!("sorted");

    if let Ok(file) = std::fs::File::create(filename) {
        bincode::serialize_into(BufWriter::new(file), &(version.to_string(), ffs.clone())).unwrap();
        println!("wrote {}", filename);
    }
    // write_ffs_debug(&ffs);
    ffs
}

fn write_ffs_debug(ffs: &Vec<(u32, u32, f32)>) {
    let width = ffs.iter().map(|(x, _, _)| *x).max().unwrap_or(0) + 1;
    let height = ffs.iter().map(|(_, y, _)| *y).max().unwrap_or(0) + 1;
    let maxf = ffs
        .iter()
        .map(|(_, _, f)| *f)
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)) // seriously?
        .unwrap_or(0f32);

    println!("{} {} {}", width, height, maxf);
    println!("painting...");
    let mut image = ImageBuffer::new(width, height);

    for (x, y, _) in ffs {
        let pixel = image.get_pixel_mut(*x, *y);
        // *pixel = image::Luma([((*f / maxf) * 255f32) as u8]);
        // *pixel = if *f != 0f32 {
        //     image::Luma([255u8])
        // } else {
        //     image::Luma([0u8])
        // }
        *pixel = image::Luma([255u8]);
    }
    println!("writing ffs.png");

    image.save("ffs.png").unwrap();
    println!("done");
}

fn split_formfactors(ff_in: Vec<(u32, u32, f32)>) -> Vec<Vec<(u32, f32)>> {
    let num = ff_in.iter().map(|(i, _, _)| i).max().unwrap() + 1;

    let mut ff_out = vec![Vec::new(); num as usize];
    for (i, j, ff) in ff_in.iter() {
        ff_out[*i as usize].push((*j, *ff));
    }

    ff_out
}

pub struct RadBuffer {
    pub r: Vec<f32>,
    pub g: Vec<f32>,
    pub b: Vec<f32>,
}

impl RadBuffer {
    /// Utility for making specifically aligned vectors
    pub fn aligned_vector<T>(len: usize, align: usize) -> Vec<T> {
        let t_size = std::mem::size_of::<T>();
        let t_align = std::mem::align_of::<T>();
        let layout = if t_align >= align {
            std::alloc::Layout::from_size_align(t_size * len, t_align).unwrap()
        } else {
            std::alloc::Layout::from_size_align(t_size * len, align).unwrap()
        };
        unsafe {
            let mem = std::alloc::alloc(layout);
            assert_eq!((mem as usize) % 16, 0);
            Vec::<T>::from_raw_parts(mem as *mut T, len, len)
        }
    }

    pub fn aligned_vector_init<T: Copy>(len: usize, align: usize, init: T) -> Vec<T> {
        let mut v = Self::aligned_vector::<T>(len, align);
        for x in v.iter_mut() {
            *x = init;
        }
        v
    }

    fn new(size: usize) -> RadBuffer {
        RadBuffer {
            // r: vec![0f32; size],
            // g: vec![0f32; size],
            // b: vec![0f32; size],
            r: Self::aligned_vector_init(size, 64, 0f32),
            g: Self::aligned_vector_init(size, 64, 0f32),
            b: Self::aligned_vector_init(size, 64, 0f32),
        }
    }
}

#[derive(Clone, Serialize)]
pub struct Blocklist {
    single: Vec<(u32, f32)>,
    vec2: Vec<(u32, [f32; 2])>,
    vec4: Vec<(u32, [f32; 4])>,
    vec8: Vec<(u32, [f32; 8])>,
    vec16: Vec<(u32, [f32; 16])>,
}

// pub struct Blockslices<'a> {
//     single: Vec<(&'a f32, &'a f32, &'a f32, f32)>,
//     vec2: Vec<(&'a [f32; 2], &'a [f32; 2], &'a [f32; 2], [f32; 2])>,
//     vec4: Vec<(&'a [f32; 4], &'a [f32; 4], &'a [f32; 4], [f32; 4])>,
//     vec8: Vec<(&'a [f32; 8], &'a [f32; 8], &'a [f32; 8], [f32; 8])>,
//     vec16: Vec<(&'a [f32; 16], &'a [f32; 16], &'a [f32; 16], [f32; 16])>,
// }

impl Blocklist {
    fn new(ff: &Vec<(u32, f32)>) -> Blocklist {
        // TODO: this crap can be done in a single scan over ff...
        let mut list1 = ff
            .iter()
            .map(|(i, ff)| (*i, (*i, *ff)))
            .collect::<std::collections::HashMap<u32, (u32, f32)>>();

        let mut list2 = std::collections::HashMap::new();
        let keys = list1.keys().map(|x| *x).collect::<Vec<_>>();

        for i in keys {
            if i % 2 != 0 {
                continue;
            }

            let mut remove = false;
            {
                let v1 = list1.get(&i);
                let v2 = list1.get(&(i + 1));

                if let Some((i1, ff1)) = v1 {
                    if let Some((_, ff2)) = v2 {
                        list2.insert(*i1, (*i1, [*ff1, *ff2]));
                        remove = true;
                    }
                }
            }
            if remove {
                list1.remove(&i);
                list1.remove(&(i + 1));
            }
        }

        let mut list4 = std::collections::HashMap::new();
        let keys = list2.keys().map(|x| *x).collect::<Vec<_>>();
        for i in keys {
            if i % 4 != 0 {
                continue;
            }

            let mut remove = false;
            {
                let v1 = list2.get(&i);
                let v2 = list2.get(&(i + 2));

                if let Some((i1, [ff1, ff2])) = v1 {
                    if let Some((_, [ff3, ff4])) = v2 {
                        list4.insert(*i1, (*i1, [*ff1, *ff2, *ff3, *ff4]));
                        remove = true;
                    }
                }
            }
            if remove {
                list2.remove(&i);
                list2.remove(&(i + 2));
            }
        }

        let mut list8 = std::collections::HashMap::new();;
        let keys = list4.keys().map(|x| *x).collect::<Vec<_>>();
        for i in keys {
            if i % 8 != 0 {
                continue;
            }

            let mut remove = false;
            {
                let v1 = list4.get(&i);
                let v2 = list4.get(&(i + 4));

                if let Some((i1, [ff1, ff2, ff3, ff4])) = v1 {
                    if let Some((_, [ff5, ff6, ff7, ff8])) = v2 {
                        list8.insert(*i1, (*i1, [*ff1, *ff2, *ff3, *ff4, *ff5, *ff6, *ff7, *ff8]));
                        remove = true;
                    }
                }
            }
            if remove {
                list4.remove(&i);
                list4.remove(&(i + 4));
            }
        }

        let mut list16 = Vec::new();
        let keys = list8.keys().map(|x| *x).collect::<Vec<_>>();
        for i in keys {
            if i % 16 != 0 {
                continue;
            }

            let mut remove = false;
            {
                let v1 = list8.get(&i);
                let v2 = list8.get(&(i + 8));

                if let Some((i1, [ff1, ff2, ff3, ff4, ff5, ff6, ff7, ff8])) = v1 {
                    if let Some((_, [ff9, ff10, ff11, ff12, ff13, ff14, ff15, ff16])) = v2 {
                        list16.push((
                            *i1,
                            [
                                *ff1, *ff2, *ff3, *ff4, *ff5, *ff6, *ff7, *ff8, *ff9, *ff10, *ff11,
                                *ff12, *ff13, *ff14, *ff15, *ff16,
                            ],
                        ));
                        remove = true;
                    }
                }
            }
            if remove {
                list8.remove(&i);
                list8.remove(&(i + 8));
            }
        }

        Blocklist {
            single: list1.values().map(|x| *x).collect(),
            vec2: list2.values().map(|x| *x).collect(),
            vec4: list4.values().map(|x| *x).collect(),
            vec8: list8.values().map(|x| *x).collect(),
            vec16: list16,
        }
    }

    fn print_stat(&self) {
        println!(
            "1: {} 2: {} 4: {} 8: {}",
            self.single.len(),
            self.vec2.len(),
            self.vec4.len(),
            self.vec8.len(),
        );
    }
}

// impl Blockslices {
//     fn new(blocklist: &Blocklist, radbuf: &RadBuffer) -> blockslices {}
// }

pub struct Scene {
    pub planes: PlanesSep,
    pub bitmap: BlockMap,
    pub emit: Vec<Vec3>,
    // pub ff: Vec<(u32, u32, f32)>,
    pub ff: Vec<Vec<(u32, f32)>>,
    pub blocks: Vec<Blocklist>,
    // pub rad_front: Vec<Vec3>,
    // pub rad_back: Vec<Vec3>,
    pub rad_front: RadBuffer,
    pub rad_back: RadBuffer,
    pub diffuse: Vec<Vec3>,
    pub pints: usize,
}

fn vec_mul(v1: &Vec3, v2: &Vec3) -> Vec3 {
    cgmath::vec3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z)
}

impl Scene {
    pub fn new(planes: PlanesSep, bitmap: BlockMap) -> Self {
        let formfactors = split_formfactors(setup_formfactors(&planes, &bitmap));
        let blocks = formfactors
            .iter()
            .map(|x| Blocklist::new(x))
            .collect::<Vec<_>>();
        let filename = "blocks.bin";
        if let Ok(file) = std::fs::File::create(filename) {
            bincode::serialize_into(BufWriter::new(file), &blocks).unwrap();
            println!("wrote {}", filename);
        }

        Scene {
            emit: vec![Vec3::zero(); planes.num_planes()],
            // rad_front: vec![Vec3::zero(); planes.num_planes()],
            // rad_back: vec![Vec3::zero(); planes.num_planes()],
            rad_front: RadBuffer::new(planes.num_planes()),
            rad_back: RadBuffer::new(planes.num_planes()),
            blocks: blocks,
            ff: formfactors,
            diffuse: vec![Vec3::new(1f32, 1f32, 1f32); planes.num_planes()],
            planes: planes,
            bitmap: bitmap,
            pints: 0,
        }
    }

    pub fn clear_emit(&mut self) {
        for v in self.emit.iter_mut() {
            *v = Vec3::zero();
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
            let diff_color = self.diffuse[i];
            if !occluded(light_pos, trace_pos, &self.bitmap) && dot > 0f32 {
                // println!("light");
                self.emit[i] +=
                    vec_mul(&diff_color, &color) * dot * (5f32 / (2f32 * 3.1415f32 * len * len));
            }
        }
    }

    pub fn do_rad(&mut self) {
        if true {
            self.do_rad_blocks();
        } else {
            std::mem::swap(&mut self.rad_front, &mut self.rad_back);

            for (i, ff_i) in self.ff.iter().enumerate() {
                // let mut rad = Vec3::zero();

                let mut rad_r = 0f32;
                let mut rad_g = 0f32;
                let mut rad_b = 0f32;
                let diffuse = self.diffuse[i as usize];
                let r = &self.rad_back.r[..];
                let g = &self.rad_back.g[..];
                let b = &self.rad_back.b[..];
                for (j, ff) in ff_i {
                    // unsafe {
                    rad_r += r[*j as usize] * diffuse.x * *ff;
                    rad_g += g[*j as usize] * diffuse.y * *ff;
                    rad_b += b[*j as usize] * diffuse.z * *ff;
                    // }
                }

                // self.rad_front[i as usize] = self.emit[i as usize] + rad;
                self.rad_front.r[i as usize] = self.emit[i as usize].x + rad_r;
                self.rad_front.g[i as usize] = self.emit[i as usize].y + rad_g;
                self.rad_front.b[i as usize] = self.emit[i as usize].z + rad_b;

                self.pints += ff_i.len();
            }
        }
    }

    // pub fn do_rad_blocks(&mut self) {
    //     std::mem::swap(&mut self.rad_front, &mut self.rad_back);
    //     // self.rad_front.copy
    //     let mut front = RadBuffer::new(0);
    //     std::mem::swap(&mut self.rad_front, &mut front);

    //     self.pints += self.do_rad_blocks_sub(&self.rad_back, &mut front, &self.blocks);
    //     std::mem::swap(&mut self.rad_front, &mut front);
    // }

    pub fn do_rad_blocks(&mut self) {
        std::mem::swap(&mut self.rad_front, &mut self.rad_back);
        // self.rad_front.copy

        assert!(self.rad_front.r.len() == self.ff.len());
        let mut front = RadBuffer::new(0);
        std::mem::swap(&mut self.rad_front, &mut front);

        self.pints += self.do_rad_blocks_sub(&self.rad_back, &mut front, &self.blocks);
        std::mem::swap(&mut self.rad_front, &mut front);
    }

    pub fn do_rad_blocks_sub(
        &self,
        src: &RadBuffer,
        dest: &mut RadBuffer,
        blocks: &Vec<Blocklist>,
    ) -> usize {
        let mut pints: usize = 0;
        for (i, ff_i) in blocks.iter().enumerate() {
            // let mut rad = Vec3::zero();

            let mut rad_r = 0f32;
            let mut rad_g = 0f32;
            let mut rad_b = 0f32;
            let diffuse = self.diffuse[i as usize];

            let r = &src.r[..];
            let g = &src.g[..];
            let b = &src.b[..];
            for (j, ff) in &ff_i.single {
                unsafe {
                    rad_r += r.get_unchecked(*j as usize) * diffuse.x * *ff;
                    rad_g += g.get_unchecked(*j as usize) * diffuse.y * *ff;
                    rad_b += b.get_unchecked(*j as usize) * diffuse.z * *ff;
                }
            }

            let vdiffuse_r = f32x2::splat(diffuse.x);
            let vdiffuse_g = f32x2::splat(diffuse.y);
            let vdiffuse_b = f32x2::splat(diffuse.z);
            let mut vsum_r = f32x2::splat(0f32);
            let mut vsum_g = f32x2::splat(0f32);
            let mut vsum_b = f32x2::splat(0f32);

            for (j, ff) in &ff_i.vec2 {
                let j = *j as usize;
                let jrange = j..j + 2;
                unsafe {
                    let vff = f32x2::from_slice_unaligned_unchecked(ff);
                    let vr = f32x2::from_slice_aligned_unchecked(r.get_unchecked(jrange.clone()));
                    let vg = f32x2::from_slice_aligned_unchecked(g.get_unchecked(jrange.clone()));
                    let vb = f32x2::from_slice_aligned_unchecked(b.get_unchecked(jrange.clone()));

                    vsum_r += vdiffuse_r * vff * vr;
                    vsum_g += vdiffuse_g * vff * vg;
                    vsum_b += vdiffuse_b * vff * vb;
                }
            }

            rad_r += vsum_r.sum();
            rad_g += vsum_g.sum();
            rad_b += vsum_b.sum();

            let vdiffuse_r = f32x4::splat(diffuse.x);
            let vdiffuse_g = f32x4::splat(diffuse.y);
            let vdiffuse_b = f32x4::splat(diffuse.z);

            let mut vsum_r = f32x4::splat(0f32);
            let mut vsum_g = f32x4::splat(0f32);
            let mut vsum_b = f32x4::splat(0f32);

            for (j, ff) in &ff_i.vec4 {
                // unsafe {
                let j = *j as usize;
                let jrange = j..j + 4;
                unsafe {
                    let vff = f32x4::from_slice_unaligned_unchecked(ff);
                    let vr = f32x4::from_slice_aligned_unchecked(r.get_unchecked(jrange.clone()));
                    let vg = f32x4::from_slice_aligned_unchecked(g.get_unchecked(jrange.clone()));
                    let vb = f32x4::from_slice_aligned_unchecked(b.get_unchecked(jrange.clone()));

                    vsum_r += vdiffuse_r * vff * vr;
                    vsum_g += vdiffuse_g * vff * vg;
                    vsum_b += vdiffuse_b * vff * vb;
                }
            }
            rad_r += vsum_r.sum();
            rad_g += vsum_g.sum();
            rad_b += vsum_b.sum();

            let vdiffuse_r = f32x8::splat(diffuse.x);
            let vdiffuse_g = f32x8::splat(diffuse.y);
            let vdiffuse_b = f32x8::splat(diffuse.z);

            let mut vsum_r = f32x8::splat(0f32);
            let mut vsum_g = f32x8::splat(0f32);
            let mut vsum_b = f32x8::splat(0f32);

            for (j, ff) in &ff_i.vec8 {
                // unsafe {
                let j = *j as usize;
                let jrange = j..j + 8;
                unsafe {
                    let vff = f32x8::from_slice_unaligned_unchecked(ff);
                    let vr = f32x8::from_slice_aligned_unchecked(r.get_unchecked(jrange.clone()));
                    let vg = f32x8::from_slice_aligned_unchecked(g.get_unchecked(jrange.clone()));
                    let vb = f32x8::from_slice_aligned_unchecked(b.get_unchecked(jrange.clone()));

                    vsum_r += vdiffuse_r * vff * vr;
                    vsum_g += vdiffuse_g * vff * vg;
                    vsum_b += vdiffuse_b * vff * vb;
                }
            }
            rad_r += vsum_r.sum();
            rad_g += vsum_g.sum();
            rad_b += vsum_b.sum();

            let vdiffuse_r = f32x16::splat(diffuse.x);
            let vdiffuse_g = f32x16::splat(diffuse.y);
            let vdiffuse_b = f32x16::splat(diffuse.z);

            let mut vsum_r = f32x16::splat(0f32);
            let mut vsum_g = f32x16::splat(0f32);
            let mut vsum_b = f32x16::splat(0f32);

            for (j, ff) in &ff_i.vec16 {
                // unsafe {
                let j = *j as usize;
                let jrange = j..j + 16;
                unsafe {
                    let vff = f32x16::from_slice_unaligned_unchecked(ff);
                    let vr = f32x16::from_slice_aligned_unchecked(r.get_unchecked(jrange.clone()));
                    let vg = f32x16::from_slice_aligned_unchecked(g.get_unchecked(jrange.clone()));
                    let vb = f32x16::from_slice_aligned_unchecked(b.get_unchecked(jrange.clone()));

                    vsum_r += vdiffuse_r * vff * vr;
                    vsum_g += vdiffuse_g * vff * vg;
                    vsum_b += vdiffuse_b * vff * vb;
                }
            }
            rad_r += vsum_r.sum();
            rad_g += vsum_g.sum();
            rad_b += vsum_b.sum();

            dest.r[i as usize] = self.emit[i as usize].x + rad_r;
            dest.g[i as usize] = self.emit[i as usize].y + rad_g;
            dest.b[i as usize] = self.emit[i as usize].z + rad_b;

            pints += ff_i.single.len()
                + ff_i.vec2.len() * 2
                + ff_i.vec4.len() * 4
                + ff_i.vec8.len() * 8
                + ff_i.vec16.len() * 16;
        }
        pints
    }

    pub fn print_stat(&self) {
        println!("write blocks");

        for blocklist in &self.blocks {
            blocklist.print_stat();
        }
    }
}
