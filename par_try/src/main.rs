use ndarray::*;
use rayon::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use ndarray_linalg::{Eig};
fn main() {
    let a:Array3<f64> = Array::random((100,2000, 2000), Uniform::new(0., 1.));
    /*
    for i in 1..100{
        let (eigvals,eigvecs)=a.slice(s![i,..,..]).eig().unwrap();
    }
    */
    let results=a.axis_iter(Axis(0)).for_each(|x| {let (eval,evec)=x.eig().unwrap();
    });
}
