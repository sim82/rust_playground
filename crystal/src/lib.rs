extern crate cgmath;
extern crate ndarray;

use std::collections::hash_map;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::iter::Iterator;
use std::path::Path;

pub mod rad;

pub type BlockMap = ndarray::Array3<bool>;

pub type Vec2i = cgmath::Vector2<i32>;

pub type Vec3i = cgmath::Vector3<i32>;
pub type Vec3 = cgmath::Vector3<f32>;

pub type Point2i = cgmath::Point2<i32>;
pub type Point3i = cgmath::Point3<i32>;
pub type Point3 = cgmath::Point3<f32>;

const NUM_PLANE_CORNERS: usize = 4;

trait Bitmap {
    fn set(&mut self, p: Point3i, v: bool);

    fn get(&self, p: Point3i) -> bool;

    fn add(&mut self, slice: &MapSlice);

    fn print(&self);
    fn step(&self, p: Point3i, dir: &Dir) -> Option<Point3i>;
}

impl Bitmap for BlockMap {
    fn set(&mut self, p: Point3i, v: bool) {
        // let c = self.coord(&p);
        self[[p.x as usize, p.y as usize, p.z as usize]] = v;
    }

    fn get(&self, p: Point3i) -> bool {
        let (x, y, z) = self.dim();
        if p.x < 0
            || p.y < 0
            || p.z < 0
            || p.x as usize >= x
            || p.y as usize >= y
            || p.z as usize >= z
        {
            return false;
        }
        // self.bitmap[self.coord(&p)]
        self[[p.x as usize, p.y as usize, p.z as usize]]
    }

    fn add(&mut self, slice: &MapSlice) {
        let (sx, _, sz) = self.dim();

        let Vec2i { x: w, y: h } = slice.size();
        assert!(w >= sx as i32);
        assert!(h >= sz as i32);

        for ((x, y, z), v) in self.indexed_iter_mut() {
            *v = slice.get(Point2i::new(x as i32, z as i32)) >= y as i32;
        }
    }

    fn print(&self) {
        for xz_slice in self.axis_iter(ndarray::Axis(1)) {
            for x_slice in xz_slice.axis_iter(ndarray::Axis(0)) {
                for x in x_slice.iter() {
                    print!("{}", if *x { 1 } else { 0 });
                }
                println!();
            }
            println!("===================================");
        }
    }

    fn step(&self, p: Point3i, dir: &Dir) -> Option<Point3i> {
        let (x, y, z) = self.dim();
        let pnew = p + dir.get_normal::<i32>();
        if pnew.x < 0
            || pnew.y < 0
            || pnew.z < 0
            || pnew.x >= x as i32
            || pnew.y >= y as i32
            || pnew.z >= z as i32
        {
            None
        } else {
            Some(pnew)
        }
    }
}

pub struct MapSlice(Vec<Vec<i32>>);

impl MapSlice {
    fn print(&self) {
        let MapSlice(v) = self;

        for line in v {
            println!(
                "{}",
                line.iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<String>>()
                    .join(" ")
            );
        }
    }

    fn max(&self) -> &i32 {
        let MapSlice(v) = self;
        v.iter().map(|x| x.iter().max().unwrap()).max().unwrap()
    }

    fn size(&self) -> Vec2i {
        let MapSlice(v) = self;
        Vec2i::new(v[0].len() as i32, v.len() as i32)
    }
    fn get(&self, p: Point2i) -> i32 {
        let MapSlice(v) = self;
        v[p.y as usize][p.x as usize]
    }

    fn pumped(&self) -> MapSlice {
        let MapSlice(v) = self;

        let mut out = Vec::new();
        for line in v {
            let mut new_line = Vec::new();
            for c in line {
                new_line.push(*c);
                new_line.push(*c);
            }
            out.push(new_line.clone());
            out.push(new_line);
        }
        MapSlice(out)
    }
}

struct DisplayWrap<T>(T);

impl<T> From<T> for DisplayWrap<T> {
    fn from(t: T) -> Self {
        DisplayWrap(t)
    }
}

impl std::fmt::Display for DisplayWrap<Point3i> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        let DisplayWrap::<Point3i>(Point3i { x, y, z }) = self;

        write!(f, "[{} {} {}]", x, y, z)
    }
}

impl std::fmt::Display for DisplayWrap<[i32; 4]> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        let DisplayWrap::<[i32; 4]>([i1, i2, i3, i4]) = self;

        write!(f, "[{} {} {} {}]", i1, i2, i3, i4)
    }
}

#[derive(Copy, Clone, PartialEq)]
pub enum Dir {
    ZxPos,
    ZxNeg,
    YzPos,
    YzNeg,
    XyPos,
    XyNeg,
}

impl Dir {
    pub fn get_normal<T: num_traits::cast::FromPrimitive + num_traits::identities::Zero>(
        &self,
    ) -> cgmath::Vector3<T> {
        let fromi = |x, y, z| {
            if let (Some(fx), Some(fy), Some(fz)) = (T::from_i32(x), T::from_i32(y), T::from_i32(z))
            {
                cgmath::Vector3::<T>::new(fx, fy, fz)
            } else {
                cgmath::Vector3::<T>::new(T::zero(), T::zero(), T::zero())
            }
        };

        match self {
            Dir::ZxNeg => fromi(0, -1, 0),
            Dir::ZxPos => fromi(0, 1, 0),
            Dir::YzNeg => fromi(-1, 0, 0),
            Dir::YzPos => fromi(1, 0, 0),
            Dir::XyNeg => fromi(0, 0, -1),
            Dir::XyPos => fromi(0, 0, 1),
        }
    }

    fn get_corners(&self) -> [Vec3i; NUM_PLANE_CORNERS] {
        match self {
            Dir::ZxNeg => [
                Vec3i { x: 0, y: 0, z: 0 },
                Vec3i { x: 0, y: 0, z: 1 },
                Vec3i { x: 1, y: 0, z: 1 },
                Vec3i { x: 1, y: 0, z: 0 },
            ],
            Dir::ZxPos => [
                Vec3i { x: 0, y: 1, z: 0 },
                Vec3i { x: 1, y: 1, z: 0 },
                Vec3i { x: 1, y: 1, z: 1 },
                Vec3i { x: 0, y: 1, z: 1 },
            ],

            Dir::YzNeg => [
                Vec3i { x: 0, y: 0, z: 0 },
                Vec3i { x: 0, y: 1, z: 0 },
                Vec3i { x: 0, y: 1, z: 1 },
                Vec3i { x: 0, y: 0, z: 1 },
            ],
            Dir::YzPos => [
                Vec3i { x: 1, y: 0, z: 0 },
                Vec3i { x: 1, y: 0, z: 1 },
                Vec3i { x: 1, y: 1, z: 1 },
                Vec3i { x: 1, y: 1, z: 0 },
            ],

            Dir::XyNeg => [
                Vec3i { x: 0, y: 0, z: 0 },
                Vec3i { x: 1, y: 0, z: 0 },
                Vec3i { x: 1, y: 1, z: 0 },
                Vec3i { x: 0, y: 1, z: 0 },
            ],
            Dir::XyPos => [
                Vec3i { x: 0, y: 0, z: 1 },
                Vec3i { x: 0, y: 1, z: 1 },
                Vec3i { x: 1, y: 1, z: 1 },
                Vec3i { x: 1, y: 0, z: 1 },
            ],
        }
    }
}

trait Cell {
    fn get_plane(&self, dir: Dir) -> [Point3i; 4];
}

impl Cell for Point3i {
    fn get_plane(&self, dir: Dir) -> [Point3i; 4] {
        let points = dir.get_corners();
        [
            self + points[0],
            self + points[1],
            self + points[2],
            self + points[3],
        ]
    }
}

pub struct Plane {
    pub vertices: [i32; NUM_PLANE_CORNERS],
    pub dir: Dir,
    pub cell: Point3i,
}

impl Plane {
    fn new(vertices: [i32; NUM_PLANE_CORNERS], dir: Dir, cell: Point3i) -> Plane {
        Plane {
            vertices: vertices,
            dir: dir,
            cell: cell,
        }
    }
}

pub struct PlanesSep {
    vertices: Vec<Point3i>,
    planes: Vec<Plane>,
}

impl PlanesSep {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            planes: Vec::new(),
        }
    }
}

impl PlanesSep {
    pub fn create_planes(&mut self, bitmap: &BlockMap) {
        for ((x, y, z), v) in bitmap.indexed_iter() {
            // println!("{} {} {}", x, y, z);
            if !v {
                continue;
            }
            let this_point = Point3i::new(x as i32, y as i32, z as i32);
            for dir in [
                Dir::ZxNeg,
                Dir::ZxPos,
                Dir::XyNeg,
                Dir::XyPos,
                Dir::YzNeg,
                Dir::YzPos,
            ]
            .iter()
            {
                let create_plane = match bitmap.step(this_point, dir) {
                    Some(p) => !Bitmap::get(bitmap, p),
                    None => *dir != Dir::ZxNeg, // don't create downfacing planes on bottom
                };

                if create_plane {
                    let local_corners: [Vec3i; NUM_PLANE_CORNERS] = dir.get_corners();
                    assert!(local_corners.len() == 4);
                    let corners = local_corners.iter().map(|x| this_point + x);
                    let mut points = [0; NUM_PLANE_CORNERS];

                    for (i, c) in corners.enumerate() {
                        points[i] = self.vertices.len() as i32;
                        self.vertices.push(c);
                    }
                    self.planes.push(Plane::new(points, *dir, this_point));
                }
            }
        }
    }

    pub fn print(&self) {
        let mut x: Vec<(&Point3i, i32)> = self
            .vertices
            .iter()
            .enumerate()
            .map(|(i, p)| (p, i as i32))
            .collect();
        x.sort_by_key(|(_, v)| *v);

        for (k, v) in x.iter() {
            println!("{}: {}", v, DisplayWrap::from(**k));
        }

        for p in &self.planes {
            println!("{}", DisplayWrap::from(p.vertices));
        }
    }

    pub fn vertex_iter(&self) -> impl Iterator<Item = (&Point3i, i32)> + '_ {
        self.vertices.iter().enumerate().map(|(i, p)| (p, i as i32))
    }

    pub fn planes_iter(&self) -> impl Iterator<Item = &Plane> + '_ {
        self.planes.iter()
    }
    pub fn num_planes(&self) -> usize {
        self.planes.len()
    }
}

pub fn to_height(c: char) -> i32 {
    match c {
        x if x >= 'a' && x <= 'z' => 1 + x as i32 - 'a' as i32,
        x if x >= '0' && x <= '9' => 2 + 'z' as i32 - 'a' as i32 + c as i32 - '0' as i32,
        _ => 0,
    }
}
pub fn read_map_slice(reader: &mut std::io::BufRead, size: Vec2i) -> std::io::Result<MapSlice> {
    // let mut slice = vec![vec![width,

    let mut slice = Vec::new(); //vec![vec![0;0]];

    for _ in 0..size.y {
        let mut line = String::new();

        reader.read_line(&mut line)?;
        let line = line.trim();

        // println!("{} {}", line.len(), width);
        assert!(line.len() == size.x as usize);

        slice.push(line.chars().map(to_height).map(|x| x).collect());
    }
    Ok(MapSlice(slice))
}

pub fn read_map<P: AsRef<Path>>(filename: P) -> std::io::Result<BlockMap> {
    let file = File::open(filename)?;

    let mut reader = BufReader::new(file);

    let width;
    let height;
    {
        let mut line = String::new();
        reader.read_line(&mut line)?;

        let h: Vec<i32> = line
            .trim()
            .split_whitespace()
            .map(|x| x.parse::<i32>().unwrap())
            .collect();
        width = h[0];
        height = h[1];
    }

    let slice = read_map_slice(&mut reader, Vec2i::new(width, height))?;
    slice.print();

    let slice = slice.pumped().pumped();
    let max = slice.max();
    let real_size = slice.size();

    println!("real size: {:?}", real_size);
    let mut bm = BlockMap::default((real_size.x as usize, *max as usize, real_size.y as usize)); //Bitmap::new(width, *max, height);
    bm.add(&slice);
    Ok(bm)
}
