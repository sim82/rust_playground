#[allow(unused_imports)]
use super::{Bitmap, BlockMap, DisplayWrap, Point3, Point3i, Vec3, Vec3i};
use super::{Dir, Plane, PlanesSep};
use cgmath::prelude::*;
use image::ImageBuffer;
use packed_simd::{f32x16, f32x2, f32x4, f32x8};

use crate::ffs;
use crate::util;
use std::cmp::Ordering;
use std::io::{BufReader, BufWriter};
use std::time::Instant;

use rayon::prelude::*;

pub struct RadBuffer {
    pub r: Vec<f32>,
    pub g: Vec<f32>,
    pub b: Vec<f32>,
}
type RadSlice<'a> = (&'a [f32], &'a [f32], &'a [f32]);
type MutRadSlice<'a> = (&'a mut [f32], &'a mut [f32], &'a mut [f32]);

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
            r: Self::aligned_vector_init(size, 64, 0f32),
            g: Self::aligned_vector_init(size, 64, 0f32),
            b: Self::aligned_vector_init(size, 64, 0f32),
        }
    }

    fn slice(&self, i: std::ops::Range<usize>) -> RadSlice<'_> {
        (&self.r[i.clone()], &self.g[i.clone()], &self.b[i.clone()])
    }
    fn slice_mut(&mut self, i: std::ops::Range<usize>) -> MutRadSlice<'_> {
        (
            &mut self.r[i.clone()],
            &mut self.g[i.clone()],
            &mut self.b[i.clone()],
        )
    }
    // this is a bit redundant, but found no better way since SliceIndex is non-copy and thus cannot be used for indexing multiple Vecs
    fn slice_full(&self) -> RadSlice<'_> {
        (&self.r[..], &self.g[..], &self.b[..])
    }
    fn slice_full_mut(&mut self) -> MutRadSlice<'_> {
        (&mut self.r[..], &mut self.g[..], &mut self.b[..])
    }

    fn chunks_mut(&mut self, size: usize) -> impl Iterator<Item = MutRadSlice<'_>> {
        itertools::izip!(
            self.r.chunks_mut(size),
            self.g.chunks_mut(size),
            self.b.chunks_mut(size)
        )
    }

    fn chunks_mut2(
        &mut self,
        size: usize,
    ) -> (
        impl Iterator<Item = &mut [f32]>,
        impl Iterator<Item = &mut [f32]>,
        impl Iterator<Item = &mut [f32]>,
    ) {
        (
            self.r.chunks_mut(size),
            self.g.chunks_mut(size),
            self.b.chunks_mut(size),
        )
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Blocklist {
    single: Vec<(u32, f32)>,
    vec2: Vec<(u32, [f32; 2])>,
    vec4: Vec<(u32, [f32; 4])>,
    vec8: Vec<(u32, [f32; 8])>,
    vec16: Vec<(u32, [f32; 16])>,
}

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

    fn num_formfactors(&self) -> usize {
        return self.single.len()
            + self.vec2.len() * 2
            + self.vec4.len() * 4
            + self.vec8.len() * 8
            + self.vec16.len() * 16;
    }
}

pub struct Scene {
    pub planes: PlanesSep,
    pub bitmap: BlockMap,
    pub emit: Vec<Vec3>,
    pub blocks: Vec<Blocklist>,
    pub extents: Vec<Vec<ffs::Extent>>,
    pub rad_front: RadBuffer,
    pub rad_back: RadBuffer,
    pub diffuse: Vec<Vec3>,
    pub pints: usize,
}

fn vec_mul(v1: &Vec3, v2: &Vec3) -> Vec3 {
    cgmath::vec3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z)
}

impl Scene {
    fn load_blocks(filename: &str, version: &str) -> Option<Vec<Blocklist>> {
        if let Ok(f) = std::fs::File::open(filename) {
            println!("read from {}", filename);
            let (file_version, blocks): (String, Vec<Blocklist>) =
                bincode::deserialize_from(BufReader::new(f)).unwrap();

            if file_version == version {
                println!("done");
                return Some(blocks);
            }
            println!("wrong version");
        }
        return None;
    }

    pub fn new(planes: PlanesSep, bitmap: BlockMap) -> Self {
        let filename = "blocks2.bin";
        let version = "v1";

        let blocks = if let Some(b) = Self::load_blocks(filename, version) {
            b
        } else {
            let formfactors = ffs::split_formfactors(ffs::setup_formfactors(&planes, &bitmap));
            let extents = ffs::to_extents(&formfactors);
            for (i, extlist) in extents.iter().enumerate() {
                println!("{}: {} ", i, extlist.len());

                for ext in extlist {
                    print!("{:?} ", ext);
                }

                println!("\n");
            }

            let blocks = formfactors
                .iter()
                .map(|x| Blocklist::new(x))
                .collect::<Vec<_>>();
            if let Ok(file) = std::fs::File::create(filename) {
                bincode::serialize_into(BufWriter::new(file), &(&version, &blocks)).unwrap();
            }
            println!("wrote {}", filename);
            blocks.clone() // this clone does something to the memory layout that improves performance by ~25%... don't know what it is
        };

        Scene {
            emit: vec![Vec3::zero(); planes.num_planes()],
            // rad_front: vec![Vec3::zero(); planes.num_planes()],
            // rad_back: vec![Vec3::zero(); planes.num_planes()],
            rad_front: RadBuffer::new(planes.num_planes()),
            rad_back: RadBuffer::new(planes.num_planes()),
            blocks: blocks,
            //ff: formfactors,
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
            let trace_pos = plane.cell + plane.dir.get_normal();

            let mut d = (pos
                - Point3::new(trace_pos.x as f32, trace_pos.y as f32, trace_pos.z as f32))
            .normalize();

            // normalize: make directional light
            let len = d.magnitude();
            // d /= len;
            let dot = cgmath::dot(d, plane.dir.get_normal());

            //self.emit[i] = Vec3::zero(); //new(0.2, 0.2, 0.2);
            let diff_color = self.diffuse[i];
            if !util::occluded(light_pos, trace_pos, &self.bitmap) && dot > 0f32 {
                // println!("light");
                self.emit[i] +=
                    vec_mul(&diff_color, &color) * dot * (5f32 / (2f32 * 3.1415f32 * len * len));
            }
        }
    }

    pub fn do_rad(&mut self) {
        self.do_rad_blocks();
    }

    pub fn do_rad_blocks(&mut self) {
        // let start = Instant::now();

        std::mem::swap(&mut self.rad_front, &mut self.rad_back);
        // self.rad_front.copy

        assert!(self.rad_front.r.len() == self.blocks.len());
        let mut front = RadBuffer::new(0);
        std::mem::swap(&mut self.rad_front, &mut front);

        let num_chunks = 32;
        let chunk_size = self.blocks.len() / num_chunks;
        let blocks_split = self.blocks.chunks(chunk_size).collect::<Vec<_>>();
        let emit_split = self.emit.chunks(chunk_size).collect::<Vec<_>>();
        let diffuse_split = self.diffuse.chunks(chunk_size).collect::<Vec<_>>();

        let (r_split, g_split, b_split) = front.chunks_mut2(chunk_size);
        let mut tmp = itertools::izip!(
            // front.chunks_mut(chunk_size),
            r_split,
            g_split,
            b_split,
            blocks_split,
            emit_split,
            diffuse_split
        )
        .collect::<Vec<_>>();

        self.pints += tmp
            .par_iter_mut()
            // .iter_mut()
            .map(|(ref mut r, ref mut g, ref mut b, blocks, emit, diffuse)| {
                RadWorkblock::new(self.rad_back.slice_full(), (r, g, b), blocks, emit, diffuse)
                    .do_iter()
            })
            .sum::<usize>();

        std::mem::swap(&mut self.rad_front, &mut front);
    }

    pub fn print_stat(&self) {
        // println!("write blocks");

        // for blocklist in &self.blocks {
        //     blocklist.print_stat();
        // }

        let ff_size: usize = self.blocks.iter().map(|x| x.num_formfactors() * 4).sum();
        let color_size = self.rad_front.r.len() * 3 * 4 * 2;

        println!("working set:\nff: {}\ncolor: {}", ff_size, color_size);
    }
}

struct RadWorkblock<'a> {
    src: RadSlice<'a>,
    dest: MutRadSlice<'a>,
    blocks: &'a [Blocklist],
    emit: &'a [Vec3],
    diffuse: &'a [Vec3],
}

impl RadWorkblock<'_> {
    pub fn new<'a>(
        src: RadSlice<'a>,
        dest: MutRadSlice<'a>,
        blocks: &'a [Blocklist],
        emit: &'a [Vec3],
        diffuse: &'a [Vec3],
    ) -> RadWorkblock<'a> {
        RadWorkblock {
            src: src,
            dest: dest,
            blocks: blocks,
            emit: emit,
            diffuse: diffuse,
        }
    }
    pub fn do_iter(&mut self) -> usize {
        let mut pints: usize = 0;
        for (i, ff_i) in self.blocks.iter().enumerate() {
            // let mut rad = Vec3::zero();

            let mut rad_r = 0f32;
            let mut rad_g = 0f32;
            let mut rad_b = 0f32;
            let diffuse = self.diffuse[i as usize];

            let (r, g, b) = self.src;
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

            self.dest.0[i as usize] = self.emit[i as usize].x + rad_r;
            self.dest.1[i as usize] = self.emit[i as usize].y + rad_g;
            self.dest.2[i as usize] = self.emit[i as usize].z + rad_b;

            pints += ff_i.single.len()
                + ff_i.vec2.len() * 2
                + ff_i.vec4.len() * 4
                + ff_i.vec8.len() * 8
                + ff_i.vec16.len() * 16;
        }
        pints
    }
}
