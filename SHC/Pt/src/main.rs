//use super::*;
use Rustb::*;
use nalgebra::Complex;
use ndarray::prelude::*;
use ndarray::*;
use std::io::Write;
use std::fs::File;
fn main() {
    let mut model=Model::from_hr("/home/liuyichen/Rust_study/SHC/Pt/","wannier90",10.92);
    let mut new_ham=model.ham.clone();
    let nk:usize=1001;
    let T:f64=0.0;
    let eta:f64=0.001;
    let og:f64=0.0;
    let mu:f64=0.0;
    let dir_1=arr1(&[1.0,0.0,0.0]);
    let dir_2=arr1(&[0.0,1.0,0.0]);
    let spin:usize=3;
    let path=array![[0.50,0.25,0.75],[0.50,0.50,0.50],[0.00,0.00,0.00],[0.50,0.00,0.50],[0.50,0.25,0.75],[0.00,0.00,0.00]];
    let label=vec!["W","L","$\\Gamma$","X","W","$\\Gamma$"];
    /*
    model.show_band(&path,&label,nk,"tests/Bi");
    */
    let nk=3001;
    let (kvec,kdist,knode)=model.k_path(&path,nk);
    let omega=model.berry_curvature(&kvec,&dir_1,&dir_2,T,og,mu,spin,eta);
    let mut file=File::create("Omega.dat").expect("Unable to BAND.dat");
    for i in 0..nk{
        let mut s = String::new();
        let aa= format!("{:.6}", kdist[[i]]);
        s.push_str(&aa);
        if omega[[i]]>=0.0 {
            s.push_str("     ");
        }else{
            s.push_str("    ");
        }
        let aa= format!("{:.6}", omega[[i]]);
        s.push_str(&aa);
        writeln!(file,"{}",s).expect("Can't write");
    }
    let mut file=File::create("omega_klabel").expect("Unable to create KLBAELS");//写下高对称点的位置
    for i in 0..path.len_of(Axis(0)){
        let mut s=String::new();
        let aa= format!("{:.6}", knode[[i]]);
        s.push_str(&aa);
        s.push_str("      ");
        s.push_str(&label[i]);
        writeln!(file,"{}",s).expect("can not write");
    }
    let nk:usize=101;
    let kmesh=arr1(&[nk,nk,nk]);
    let conductivity=model.Hall_conductivity(&kmesh,&dir_1,&dir_2,T,og,mu,spin,eta);
    println!("{}",conductivity);
}
