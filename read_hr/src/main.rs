use std::io::BufReader;
use std::io::BufRead;
use std::fs::File;
use std::path::Path;
use nalgebra::Complex;
use ndarray::*;
fn main() {
    from_hr("/home/liuyichen/Rust_study/read_hr/","wannier90");
}
pub fn from_hr(path:&str,file_name:&str){
//我们首先读 hr.dat 文件
    let mut file_path = path.to_string();
    file_path.push_str(file_name);
    let mut hr_path=file_path.clone();
    hr_path.push_str("_hr.dat");
    let path=Path::new(&hr_path);
    let hr=File::open(path).expect("Unable open the file, please check if have hr file");
    let reader = BufReader::new(hr);
    let mut reads:Vec<String>=Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        reads.push(line.clone());
    }
    let nsta=reads[1].trim().parse::<usize>().unwrap();
    let mut n_R=reads[2].trim().parse::<usize>().unwrap();
    let mut weights:Vec<usize>=Vec::new();
    let mut n_line:usize=0;
    for i in 3..reads.len(){
        let string:Vec<usize>=reads[i].trim().split_whitespace().map(|x| x.parse::<usize>().unwrap()).collect();
        weights.extend(string.clone());
        println!("{}",reads[i]);
        if string.len() !=15{
            n_line=i;
            break
        }
    }
/*
    for R in n_line..reads.len(){
        let mut string=reads[R].trim().split_whitespace();
        let a=string.next().unwrap().parse::<isize>().unwrap();
        let b=string.next().unwrap().parse::<isize>().unwrap();
        let c=string.next().unwrap().parse::<isize>().unwrap();
        if a==0 && b==0 && c==0{
            n_line=R;
            n_R-=R/nsta/nsta;
            break
        }
    }
*/
    let mut hamR=Array2::<isize>::zeros((1,3));
    let mut ham=Array3::<Complex<f64>>::zeros((1,nsta,nsta));
    for i in 0..n_R{
        let mut string=reads[i*nsta*nsta+n_line].trim().split_whitespace();
        let a=string.next().unwrap().parse::<isize>().unwrap();
        let b=string.next().unwrap().parse::<isize>().unwrap();
        let c=string.next().unwrap().parse::<isize>().unwrap();
        if (c>0) || (c==0 && b>0) ||(c==0 && b==0 && a>=0){
            println!("a={},b={},c={}",a,b,c);
            if a==0 && b==0 && c==0{
                hamR[[0,0]]=a;
                hamR[[0,1]]=b;
                hamR[[0,2]]=c;
                for ind_i in 0..nsta{
                    for ind_j in 0..nsta{
                        let mut string=reads[i*nsta*nsta+ind_i*nsta+ind_j+n_line].trim().split_whitespace();
                        let re=string.nth(5).unwrap().parse::<f64>().unwrap();
                        let im=string.next().unwrap().parse::<f64>().unwrap();
                        ham[[0,ind_i,ind_j]]=Complex::new(re,im);
                    }
                }
            }else{
                hamR.append(Axis(0),arr2(&[[a,b,c]]).view()).unwrap();
                let mut matrix=Array3::<Complex<f64>>::zeros((1,nsta,nsta));
                for ind_i in 0..nsta{
                    for ind_j in 0..nsta{
                        let mut string=reads[i*nsta*nsta+ind_i*nsta+ind_j+n_line].trim().split_whitespace();
                        let re=string.nth(5).unwrap().parse::<f64>().unwrap();
                        let im=string.next().unwrap().parse::<f64>().unwrap();
                        matrix[[0,ind_i,ind_j]]=Complex::new(re,im);
                    }
                }
                ham.append(Axis(0),matrix.view()).unwrap();
            }
        }
    }
    //文件读取完成
    println!("{:?}",hamR);
    for i in 0..hamR.len_of(Axis(0)){
        let R=-hamR.slice(s![i,..]).to_owned();
        let have=find_R(&hamR,&R);
        if have {
            println!("wrong, have conjugate item");
            println!("{:?},{:?}",R,hamR.slice(s![i,..]).to_owned())
        }
    }
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
