use cgmath::*;
use std::hash::{Hash, Hasher};

use crate::{animation::Lerp, pointcloud::Aabb};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerspectiveCamera {
    pub position: Point3<f32>,
    pub rotation: Quaternion<f32>,
    pub projection: PerspectiveProjection,
}

impl PerspectiveCamera {
    pub fn new(
        position: Point3<f32>,
        rotation: Quaternion<f32>,
        projection: PerspectiveProjection,
    ) -> Self {
        PerspectiveCamera {
            position,
            rotation,
            projection: projection,
        }
    }

    pub fn fit_near_far(&mut self, aabb: &Aabb<f32>) {
        // Get the view direction (forward vector) from the camera rotation
        let view_matrix = self.view_matrix();
        let forward = Vector3::new(-view_matrix[0][2], -view_matrix[1][2], -view_matrix[2][2]).normalize();
        
        // Get all 8 corners of the bounding box
        let corners = [
            Point3::new(aabb.min.x, aabb.min.y, aabb.min.z),
            Point3::new(aabb.max.x, aabb.min.y, aabb.min.z),
            Point3::new(aabb.min.x, aabb.max.y, aabb.min.z),
            Point3::new(aabb.max.x, aabb.max.y, aabb.min.z),
            Point3::new(aabb.min.x, aabb.min.y, aabb.max.z),
            Point3::new(aabb.max.x, aabb.min.y, aabb.max.z),
            Point3::new(aabb.min.x, aabb.max.y, aabb.max.z),
            Point3::new(aabb.max.x, aabb.max.y, aabb.max.z),
        ];
        
        // Project all corners along the view direction to find min/max depth
        let mut min_depth = f32::INFINITY;
        let mut max_depth = f32::NEG_INFINITY;
        
        for corner in &corners {
            let to_corner = corner - self.position;
            let depth = to_corner.dot(forward);
            min_depth = min_depth.min(depth);
            max_depth = max_depth.max(depth);
        }
        
        // Add some padding based on the field of view to account for splat extent
        // Larger FOV means splats can extend further from their centers
        let fov_factor = (self.projection.fovy.0.max(self.projection.fovx.0) * 0.5).tan();
        let size = aabb.size();
        let diagonal_length = (size.x * size.x + size.y * size.y + size.z * size.z).sqrt();
        let padding = diagonal_length * fov_factor * 0.1; // 10% of diagonal scaled by FOV
        
        // Ensure we have reasonable near/far values
        let znear = (min_depth - padding).max(0.001); // Minimum near plane
        let zfar = (max_depth + padding).max(znear + 0.01); // Ensure far > near
        
        // Apply a safety margin to prevent clipping at the edges
        let safety_margin = (zfar - znear) * 0.05; // 5% margin
        
        self.projection.znear = (znear - safety_margin).max(0.001);
        self.projection.zfar = zfar + safety_margin;
    }
}

impl Hash for PerspectiveCamera {
    fn hash<H: Hasher>(&self, state: &mut H) {
        bytemuck::bytes_of(&self.view_matrix()).hash(state);
        bytemuck::bytes_of(&self.proj_matrix()).hash(state);
    }
}

impl Lerp for PerspectiveCamera {
    // using SPLIT interpolation to interpolate between two cameras
    // see Kim et al. "A general construction scheme for unit quaternion curves with simple high order derivatives."
    fn lerp(&self, other: &Self, amount: f32) -> Self {
        PerspectiveCamera {
            position: Point3::from_vec(
                self.position.to_vec().lerp(other.position.to_vec(), amount),
            ),
            rotation: self.rotation.slerp(other.rotation, amount),
            projection: self.projection.lerp(&other.projection, amount),
        }
    }
}

impl Default for PerspectiveCamera {
    fn default() -> Self {
        Self {
            position: Point3::new(0., 0., -1.),
            rotation: Quaternion::new(1., 0., 0., 0.),
            projection: PerspectiveProjection {
                fovx: Deg(45.).into(),
                fovy: Deg(45.).into(),
                znear: 0.1,
                zfar: 100.,
                fov2view_ratio: 1.,
            },
        }
    }
}

impl Camera for PerspectiveCamera {
    fn view_matrix(&self) -> Matrix4<f32> {
        world2view(Matrix3::from(self.rotation), self.position.to_vec())
    }

    fn proj_matrix(&self) -> Matrix4<f32> {
        self.projection.projection_matrix()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerspectiveProjection {
    pub fovx: Rad<f32>,
    pub fovy: Rad<f32>,
    pub znear: f32,
    pub zfar: f32,
    /// fov ratio to viewport ratio
    /// needed for camera viewport resize
    pub(crate) fov2view_ratio: f32,
}

impl Hash for PerspectiveProjection {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.fovx.0.to_bits().hash(state);
        self.fovy.0.to_bits().hash(state);
        self.znear.to_bits().hash(state);
        self.zfar.to_bits().hash(state);
        self.fov2view_ratio.to_bits().hash(state);
    }
}

#[rustfmt::skip]
pub const VIEWPORT_Y_FLIP: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, -1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.,
    0.0, 0.0, 0., 1.0,
);

impl PerspectiveProjection {
    pub fn new<F: Into<Rad<f32>>>(
        viewport: Vector2<u32>,
        fov: Vector2<F>,
        znear: f32,
        zfar: f32,
    ) -> Self {
        let fov = fov.map(|v| v.into());
        let vr = viewport.x as f32 / viewport.y as f32;
        let fr = fov.x.0 / fov.y.0;
        Self {
            fovx: fov.x,
            fovy: fov.y,
            znear,
            zfar,
            fov2view_ratio: vr / fr,
        }
    }

    pub fn projection_matrix(&self) -> Matrix4<f32> {
        build_proj(self.znear, self.zfar, self.fovx, self.fovy)
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let ratio = width as f32 / height as f32;
        if width > height {
            self.fovy = self.fovx / ratio * self.fov2view_ratio;
        } else {
            self.fovx = self.fovy * ratio * self.fov2view_ratio;
        }
    }

    pub(crate) fn focal(&self, viewport: Vector2<u32>) -> Vector2<f32> {
        let viewport: Vector2<f32> = viewport.cast().unwrap();
        return Vector2::new(
            fov2focal(self.fovx, viewport.x),
            fov2focal(self.fovy, viewport.y),
        );
    }

    pub fn lerp(&self, other: &PerspectiveProjection, amount: f32) -> PerspectiveProjection {
        PerspectiveProjection {
            fovx: self.fovx * (1. - amount) + other.fovx * amount,
            fovy: self.fovy * (1. - amount) + other.fovy * amount,
            znear: self.znear * (1. - amount) + other.znear * amount,
            zfar: self.zfar * (1. - amount) + other.zfar * amount,
            fov2view_ratio: self.fov2view_ratio * (1. - amount) + other.fov2view_ratio * amount,
        }
    }
}

pub struct FrustumPlanes {
    pub near: Vector4<f32>,
    pub far: Vector4<f32>,
    pub left: Vector4<f32>,
    pub right: Vector4<f32>,
    pub top: Vector4<f32>,
    pub bottom: Vector4<f32>,
}

pub trait Camera {
    fn view_matrix(&self) -> Matrix4<f32>;
    fn proj_matrix(&self) -> Matrix4<f32>;

    fn position(&self) -> Point3<f32> {
        Point3::from_homogeneous(self.view_matrix().inverse_transform().unwrap().w)
    }

    fn frustum_planes(&self) -> FrustumPlanes {
        let p = self.proj_matrix();
        let v = self.view_matrix();
        let pv = p * v;
        let mut planes = [Vector4::zero(); 6];
        planes[0] = pv.row(3) + pv.row(0);
        planes[1] = pv.row(3) - pv.row(0);
        planes[2] = pv.row(3) + pv.row(1);
        planes[3] = pv.row(3) - pv.row(1);
        planes[4] = pv.row(3) + pv.row(2);
        planes[5] = pv.row(3) - pv.row(2);
        for i in 0..6 {
            planes[i] = planes[i].normalize();
        }
        return FrustumPlanes {
            near: planes[4],
            far: planes[5],
            left: planes[0],
            right: planes[1],
            top: planes[3],
            bottom: planes[2],
        };
    }
}

pub fn world2view(r: Matrix3<f32>, t: Vector3<f32>) -> Matrix4<f32> {
    let mut rt = Matrix4::from(r);
    rt[0].w = t.x;
    rt[1].w = t.y;
    rt[2].w = t.z;
    rt[3].w = 1.;
    return rt.inverse_transform().unwrap().transpose();
}

pub fn build_proj(znear: f32, zfar: f32, fov_x: Rad<f32>, fov_y: Rad<f32>) -> Matrix4<f32> {
    let tan_half_fov_y = (fov_y / 2.).tan();
    let tan_half_fov_x = (fov_x / 2.).tan();

    let top = tan_half_fov_y * znear;
    let bottom = -top;
    let right = tan_half_fov_x * znear;
    let left = -right;

    let mut p = Matrix4::zero();
    p[0][0] = 2.0 * znear / (right - left);
    p[1][1] = 2.0 * znear / (top - bottom);
    p[0][2] = (right + left) / (right - left);
    p[1][2] = (top + bottom) / (top - bottom);
    p[3][2] = 1.;
    p[2][2] = zfar / (zfar - znear);
    p[2][3] = -(zfar * znear) / (zfar - znear);
    return p.transpose();
}

pub fn focal2fov(focal: f32, pixels: f32) -> Rad<f32> {
    return Rad(2. * (pixels / (2. * focal)).atan());
}

pub fn fov2focal(fov: Rad<f32>, pixels: f32) -> f32 {
    pixels / (2. * (fov * 0.5).tan())
}
