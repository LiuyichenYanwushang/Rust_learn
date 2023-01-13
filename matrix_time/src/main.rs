use ndarray::Array;
use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use num::complex::Complex;
use std::f64::consts::PI;
use ndarray_linalg::{Eig};
fn main() {
    let a:Array2<f64> = Array::random((2000, 2000), Uniform::new(0., 1.));
    let (eigvals,eigvecs)=a.eig().unwrap();
    //let c=a.dot(&eigvecs);
    //println!("{:8.4}", eigvecs);
}
