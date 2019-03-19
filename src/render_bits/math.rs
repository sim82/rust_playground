use cgmath::prelude::*;
use cgmath::{Matrix4, Rad, Vector3};

pub fn perspective_projection(fov: Rad<f32>, a: f32, zn: f32, zf: f32) -> Matrix4<f32> {
    let f = Rad::cot(fov / 2f32);

    Matrix4::new(
        f / a,
        0f32,
        0f32,
        0f32,
        0f32,
        -f,
        0f32,
        0f32,
        0f32,
        0f32,
        zf / (zn - zf),
        -1f32,
        0f32,
        0f32,
        (zn * zf) / (zn - zf),
        0f32,
    )
}

pub fn orthograpic_projection(
    left_plane: f32,
    right_plane: f32,
    bottom_plane: f32,
    top_plane: f32,
    near_plane: f32,
    far_plane: f32,
) -> Matrix4<f32> {
    Matrix4::new(
        2.0f32 / (right_plane - left_plane),
        0.0f32,
        0.0f32,
        0.0f32,
        0.0f32,
        2.0f32 / (bottom_plane - top_plane),
        0.0f32,
        0.0f32,
        0.0f32,
        0.0f32,
        1.0f32 / (near_plane - far_plane),
        0.0f32,
        -(right_plane + left_plane) / (right_plane - left_plane),
        -(bottom_plane + top_plane) / (bottom_plane - top_plane),
        near_plane / (near_plane - far_plane),
        1.0f32,
    )
}

// projection matrix for a simple 2d canvas with origin in upper left corner, y-down and one pixel per x/y unit
// system stays right-handed (->z positive pointing inward) just like vulkan clip space
pub fn canvas_projection(width: u32, height: u32) -> Matrix4<f32> {
    // inverse of
    //  - translate origin to (1,1)
    //  - scale to screen resolution

    let scale = Matrix4::from_nonuniform_scale((width / 2) as f32, (height / 2) as f32, 1f32);
    let translate = Matrix4::from_translation(Vector3::<_>::new(1f32, 1f32, 0f32));

    (scale * translate).invert().unwrap()
    // scale.invert().unwrap()
}
