use nalgebra::Complex;
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::*;
use Rustb::*;
use std::f64::consts::PI;
use std::time::{Duration, Instant};
use gnuplot::{Figure, Caption, Color};
use std::fs::File;
fn main() {
    let zero_energy:f64=-2.0;
    let mut model=Model::from_hr("/home/liuyichen/Rust_study/SHC/MoS2/data/","wannier90",zero_energy);
    let nk:usize=301;
    let T:f64=0.0;
    let eta:f64=0.0001;
    let og:f64=0.0;
    let mu:f64=0.0;
    let dir_1=arr1(&[1.0,0.0,0.0]);
    let dir_2=arr1(&[0.0,1.0,0.0]);
    let spin:usize=3;
    let kmesh=arr1(&[nk,nk,1]);
    let start = Instant::now();   // 开始计时
    let re_err=5e-2;
    let ab_err=5e-2;
    //let conductivity=model.Hall_conductivity_adapted(&kmesh,&dir_1,&dir_2,T,og,mu,spin,eta,re_err,ab_err);
    let conductivity=model.Hall_conductivity(&kmesh,&dir_1,&dir_2,T,og,mu,spin,eta);
    let end = Instant::now();    // 结束计时
    let duration = end.duration_since(start); // 计算执行时间
    println!("{}",conductivity);
    println!("function_a took {} seconds", duration.as_secs_f64());   // 输出执行时间
    let kvec=gen_kmesh(&kmesh);
    let kvec=kvec-0.5;
    let kvec=kvec*0.5;
    let kvec=model.lat.dot(&(kvec.reversed_axes()));
    let kvec=kvec.reversed_axes();
    let berry_curv=model.berry_curvature(&kvec,&dir_1,&dir_2,T,og,mu,spin,eta);
    let data=berry_curv.into_shape((nk,nk,1)).unwrap();
    let data=data.slice(s![..,..,0]).to_owned();
    let max_data=find_max_value(&data);
    println!("{}",max_data);
    draw_heatmap(data,"heat_map.pdf");
    let nk=301;
    let path=arr2(&[[0.0,0.0,0.0],[0.5,0.0,0.0],[0.6666,0.3333,0.0],[0.0,0.0,0.0]]);
    let (kvec,kdist,knode)=model.k_path(&path,nk);
    let berry_curv=model.berry_curvature(&kvec,&dir_1,&dir_2,T,og,mu,spin,eta);
    let mut fg = Figure::new();
    let x:Vec<f64>=kdist.to_vec();
    let axes=fg.axes2d();
    let y:Vec<f64>=berry_curv.to_owned().to_vec();
    axes.lines(&x, &y, &[Color("black")]);
    fg.set_terminal("pdfcairo", "plot.pdf");
    fg.show();
    let label=vec!["G","M","K'","G"];
    model.show_band(&path,&label,nk,"./band");
    println!("{}",model.nsta);
    println!("{}",model.norb);


}

fn draw_heatmap(data: Array2<f64>,name:&str) {
    use gnuplot::{Figure, AxesCommon, AutoOption::Fix,HOT,RAINBOW};
    let mut fg = Figure::new();

    let (width, height) = (data.shape()[1], data.shape()[0]);
    let mut heatmap_data = vec![];

    for i in 0..height {
        for j in 0..width {
            heatmap_data.push(data[(i, j)]);
        }
    }

    let axes = fg.axes2d();
    axes.set_title("Heatmap", &[]);
    axes.set_cb_label("Values", &[]);
    axes.set_palette(RAINBOW);
    axes.image(heatmap_data.iter(), width, height,None, &[]);
    let size=data.shape();
    let axes=axes.set_x_range(Fix(0.0), Fix((size[0]-1) as f64));
    let axes=axes.set_y_range(Fix(0.0), Fix((size[1]-1) as f64));
    let axes=axes.set_aspect_ratio(Fix(1.0));
    fg.set_terminal("pdfcairo",name);
    fg.show().expect("Unable to draw heatmap");
}

fn find_max_value(data: &Array2<f64>) -> f64 {
    data.iter()
        .fold(f64::MIN, |max, &value| max.max(value))
}
