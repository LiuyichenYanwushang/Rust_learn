use nalgebra::Complex;
use ndarray::prelude::*;
use ndarray::*;
use Rustb::*;
use std::f64::consts::PI;
use std::time::{Duration, Instant};
fn main() {
    let zero_energy:f64=17.9919;
    let mut  model=Model::from_hr("/home/liuyichen/Rust_study/SHC/Pt_1/data/","Pt",zero_energy);
    model.lat=model.lat*0.529117249;
    let nk:usize=101;
    let T:f64=0.0;
    let eta:f64=0.0001;
    let og:f64=0.0;
    let mu:f64=0.0;
    let dir_1=arr1(&[1.0,0.0,0.0]);
    let dir_2=arr1(&[0.0,1.0,0.0]);
    let spin:usize=3;
    let kmesh=arr1(&[nk,nk,nk]);
    let start = Instant::now();   // 开始计时
    let re_err=5e-2;
    let ab_err=5e-2;
    //let conductivity=model.Hall_conductivity_adapted(&kmesh,&dir_1,&dir_2,T,og,mu,spin,eta,re_err,ab_err);
    let conductivity=model.Hall_conductivity(&kmesh,&dir_1,&dir_2,T,og,mu,spin,eta);
    let end = Instant::now();    // 结束计时
    let duration = end.duration_since(start); // 计算执行时间
    println!("{}",conductivity);
    println!("function_a took {} seconds", duration.as_secs_f64());   // 输出执行时间

    model.show_band(&path,&label,nk,"./band");
}
