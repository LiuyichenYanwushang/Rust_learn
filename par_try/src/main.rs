use ndarray::*;
use rayon::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use ndarray_linalg::{Eig};
use nalgebra::Complex;
use rayon::iter::ParallelIterator;
fn main() {
    let size=32;
    let a:Array3<f64> = Array::random((size,2, 2), Uniform::new(0., 1.));
    let mut band:Array2<Complex<f64>>=Array2::zeros((size,2));
    let (eval,evec):(Vec<_>,Vec<_>)=a.axis_iter(Axis(0))
    .into_par_iter()
    .map(|x| {let (eval,evec)=x.eig().unwrap();(eval.to_vec(),evec.into_raw_vec())}).collect();
    let array2 = Array2::from_shape_vec((size, 2), eval.into_iter().flatten().collect()).unwrap();
    let array3=Array3::from_shape_vec((size, 2,2), evec.into_iter().flatten().collect()).unwrap();
    println!("{:?}",array3)
}
