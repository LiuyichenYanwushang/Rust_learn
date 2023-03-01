use ndarray::{Array2,Axis,s,array,arr2,Array1};
use rayon::prelude::*;
use nalgebra::Complex;
use ndarray_linalg::*;
use ndarray_linalg::{Eigh, UPLO,Eig};
use gnuplot::*;
use std::f64::consts::PI;
use approx::assert_abs_diff_eq;
fn main() {
/*
    let mut a:Array2::<isize> = arr2(&[[1, 2, 3, 4, 5],[6,7,8,9,10]]);
    let mut R:Array1::<isize> = array![5,10];
    a=a.reversed_axes();
    println!("{:?}",a);
    println!("{}",find_R(&a,&R));
    println!("{}",index_R(&a,&R));
    let i0:Complex<f64>=1.0+1.0*1.0*Complex::i();
    println!("{}",i0.conj());
    let x = [1, 2, 3, 4, 5];
    let y = [1, 4, 9, 16, 25];

    let mut fg = Figure::new();
    fg.axes2d().lines(&x, &y, &[Caption("y = x^2"), Color("blue")]);
    fg.set_terminal("pdf", "test.pdf");
    fg.show();
    let a:Array2::<f64>=arr2(&[[1.0,1.0,1.0],[0.0,1.0,0.0],[1.0,0.0,1.0]]);
    println!("{:?}",a.view().mapv(f64::exp));
    println!("{}",i0.exp());
    let b=a.dot(&a.t());
    let c=b.det().unwrap();
    let a = array![
        [1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 10.],
    ];

    let a_inv = a.inv().unwrap();

    println!("Inverse of a is:\n{}", a_inv);
    println!("b and det of b is:\n{},{}",b, c);
    let a=array![1.0,2.0,3.0];
    println!("{}",a.dot(&a).sqrt());
    println!("{}",(a+a_inv));
    println!("{}",Array2::from_diag(&R));
    println!("{}",PI)
    let a: Array2<f64> = array![
        [2., 1.],
        [1., 2.],
    ];
    //let Ok((eigvals, eigvecs)) = a.eigh(UPLO::Lower);
    let (eigvals, eigvecs) = if let Ok((eigvals, eigvecs)) = a.eigh(UPLO::Lower) { (eigvals, eigvecs) } else { todo!() };
    assert_abs_diff_eq!(eigvals, array![1., 3.]);
    assert_abs_diff_eq!(
        a.dot(&eigvecs),
        eigvecs.dot(&Array2::from_diag(&eigvals)),);
*/
    let a=arr2(&[[1.0,3.0,2.0],[3.0,5.0,4.0],[2.0,4.0,2.0]]);
    let b=arr2(&[[0.0,1.0,3.0],[-1.0,0.0,6.0],[-3.0,-6.0,0.0]]);
    let b=b.map(|x| Complex::new(0.0,*x));
    let a=a.map(|x| Complex::new(*x,0.0));
    let a=a+b.clone();
    let (eval,evec)=a.eigh(UPLO::Lower).unwrap();
    println!("{}",eval);
    println!("{}",evec);
    println!("{}",evec.clone().t().dot(&(a.dot(&evec.map(|x| x.conj())))));
    println!("{}",a);
    println!("{}",b);
    let c=a.dot(&b);
    println!("{}",c);
    let c=c+b.dot(&a);
    println!("{}",c);
    println!("{:?}","a  b  c".trim().split_whitespace().nth(0));
    println!("{:?}","a  b  c".trim().split_whitespace().nth(1));
    println!("{:?}","a  b  c".trim().split_whitespace().nth(2));
}
fn find_R(hamR:&Array2::<isize>,R:&Array1::<isize>)->bool{
    let n_R:usize=hamR.len_of(Axis(0));
    let dim_R:usize=hamR.len_of(Axis(1));
    for i in 0..(n_R){

        let mut a=true;
        for j in 0..(dim_R){
            a=a&&( hamR[[i,j]]==R[[j]]);
        }
        if a{
            return true
        }
    }
    false
}
fn index_R(hamR:&Array2::<isize>,R:&Array1::<isize>)->usize{
    let n_R:usize=hamR.len_of(Axis(0));
    let dim_R:usize=hamR.len_of(Axis(1));
    for i in 0..(n_R-1){

        let mut a=true;
        for j in 0..(dim_R-1){
            a=a&&( hamR[[i,j]]==R[[j]])
        }
        if a{
            return i
        }
    }
    0
}
