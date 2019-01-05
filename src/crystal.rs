extern crate cgmath;

use self::cgmath::Point3;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::iter::Iterator;
use std::path::Path;

pub struct Bitmap {
    sx: usize,
    sy: usize,
    sz: usize,
    bitmap: Vec<bool>,
}

impl Bitmap {
    fn new(sx: usize, sy: usize, sz: usize) -> Bitmap {
        Bitmap {
            sx: sx,
            sy: sy,
            sz: sz,
            bitmap: vec![false; sx * sy * sz],
        }
    }
    fn coord(&self, x: usize, y: usize, z: usize) -> usize {
        let slice = self.sx * self.sy;
        slice * z + self.sx * y + x
    }

    fn set(&mut self, x: usize, y: usize, z: usize, v: bool) {
        let c = self.coord(x, y, z);
        self.bitmap[c] = v;
    }

    fn get(&self, x: usize, y: usize, z: usize) -> bool {
        self.bitmap[self.coord(x, y, z)]
    }

    fn add(&mut self, slice: &MapSlice) {
        let (w, h) = slice.size();
        assert!(w >= self.sx);
        assert!(h >= self.sz);

        // let MapSlice(v) = slice;

        for y in (0..self.sy).rev() {
            for x in 0..self.sx {
                for z in 0..self.sz {
                    if slice.get(x, z) >= y {
                        self.set(x, y, z, true);
                    }
                }
            }
        }
        //assert!(self.sx ==
    }

    fn print(&self) {
        for y in (0..self.sy).rev() {
            for x in 0..self.sx {
                for z in 0..self.sz {
                    print!("{}", if self.get(x, y, z) { 1 } else { 0 });
                }
                println!();
            }

            println!("===================================");
        }
    }
}

pub struct MapSlice(Vec<Vec<usize>>);

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

    fn max(&self) -> &usize {
        let MapSlice(v) = self;
        v.iter().map(|x| x.iter().max().unwrap()).max().unwrap()
    }

    fn size(&self) -> (usize, usize) {
        let MapSlice(v) = self;
        (v[0].len(), v.len())
    }
    fn get(&self, x: usize, y: usize) -> usize {
        let MapSlice(v) = self;
        v[y][x]
    }
}

pub type Vec3i = cgmath::Vector3<i32>;
pub type Vec3 = cgmath::Vector3<f32>;

pub enum Dir {
    ZxPos,
    ZxNeg,
    YzPos,
    YzNeg,
    XyPos,
    XyNeg,
}

impl Dir {
    fn get_normal(&self) -> Vec3 {
        match self {
            Dir::ZxNeg => Vec3::new(0.0, -1.0, 0.0),
            Dir::ZxPos => Vec3::new(0.0, 1.0, 0.0),
            Dir::YzNeg => Vec3::new(-1.0, 0.0, 0.0),
            Dir::YzPos => Vec3::new(1.0, 0.0, 0.0),
            Dir::XyNeg => Vec3::new(0.0, 0.0, -1.0),
            Dir::XyPos => Vec3::new(0.0, 0.0, 1.0),
        }
    }

    fn get_corners(&self) -> (Vec3i, Vec3i, Vec3i, Vec3i) {
        match self {
            Dir::ZxNeg => (
                Vec3i { x: 0, y: 0, z: 0 },
                Vec3i { x: 0, y: 0, z: 1 },
                Vec3i { x: 1, y: 0, z: 1 },
                Vec3i { x: 1, y: 0, z: 0 },
            ),
            Dir::ZxPos => (
                Vec3i { x: 0, y: 1, z: 0 },
                Vec3i { x: 1, y: 1, z: 0 },
                Vec3i { x: 1, y: 1, z: 1 },
                Vec3i { x: 0, y: 1, z: 1 },
            ),

            Dir::YzNeg => (
                Vec3i { x: 0, y: 0, z: 0 },
                Vec3i { x: 0, y: 1, z: 0 },
                Vec3i { x: 0, y: 1, z: 1 },
                Vec3i { x: 0, y: 0, z: 1 },
            ),
            Dir::YzPos => (
                Vec3i { x: 1, y: 0, z: 0 },
                Vec3i { x: 1, y: 0, z: 1 },
                Vec3i { x: 1, y: 1, z: 1 },
                Vec3i { x: 1, y: 1, z: 0 },
            ),

            Dir::XyNeg => (
                Vec3i { x: 0, y: 0, z: 0 },
                Vec3i { x: 1, y: 0, z: 0 },
                Vec3i { x: 1, y: 1, z: 0 },
                Vec3i { x: 0, y: 1, z: 0 },
            ),
            Dir::XyPos => (
                Vec3i { x: 0, y: 0, z: 1 },
                Vec3i { x: 0, y: 1, z: 1 },
                Vec3i { x: 1, y: 1, z: 1 },
                Vec3i { x: 1, y: 0, z: 1 },
            ),
        }
    }
}

trait Cell {
    fn get_plane(&self, dir: Dir) -> [Point3<i32>; 4];
}

impl Cell for Point3<i32> {
    fn get_plane(&self, dir: Dir) -> [Point3<i32>; 4] {
        let points = dir.get_corners();
        [
            self + points.0,
            self + points.1,
            self + points.2,
            self + points.3,
        ]
    }
}

pub fn to_height(c: char) -> u32 {
    match c {
        x if x >= 'a' && x <= 'z' => 1 + x as u32 - 'a' as u32,
        x if x >= '0' && x <= '9' => 2 + 'z' as u32 - 'a' as u32 + c as u32 - '0' as u32,
        _ => 0,
    }
}

pub fn read_map_slice(
    reader: &mut std::io::BufRead,
    width: usize,
    height: usize,
) -> std::io::Result<MapSlice> {
    // let mut slice = vec![vec![width,

    let mut slice = Vec::new(); //vec![vec![0;0]];

    for _ in 0..height {
        let mut line = String::new();

        reader.read_line(&mut line)?;
        let line = line.trim();

        // println!("{} {}", line.len(), width);
        assert!(line.len() == width);

        slice.push(line.chars().map(to_height).map(|x| x as usize).collect());
    }
    Ok(MapSlice(slice))
}

pub fn read_map<P: AsRef<Path>>(filename: P) -> std::io::Result<Bitmap> {
    let file = File::open(filename)?;

    let mut reader = BufReader::new(file);

    let width;
    let height;
    {
        let mut line = String::new();
        reader.read_line(&mut line)?;

        // let h : Vec<usize> = header.trim().split_whitespace().map(|x| x.parse::<usize>().unwrap()).collect();
        let h: Vec<usize> = line
            .trim()
            .split_whitespace()
            .map(|x| x.parse::<usize>().unwrap())
            .collect();
        width = h[0];
        height = h[1];
    }
    // println!( "size: {} {}", h[0], h[1]);

    let slice = read_map_slice(&mut reader, width, height)?;
    slice.print();
    let max = slice.max();
    // println!( "max: {}", slice.max());
    // let mut bitmap = Bitmap::new(width, height,
    // for i in 0..height {

    // }
    let mut bm = Bitmap::new(width, *max, height);
    bm.add(&slice);
    bm.print();
    Ok(bm)
}
