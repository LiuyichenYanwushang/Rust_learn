use std::io::BufReader;
use std::io::BufRead;
use std::fs::File;
use std::path::Path;
use nalgebra::Complex;
use ndarray::*;
fn main() {
    from_hr("./","wannier90");
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
    /*
    //文件读取完成
    println!("{:?}",hamR);
    for i in 1..hamR.len_of(Axis(0)){
        let R=-hamR.slice(s![i,..]).to_owned();
        let have=find_R(&hamR,&R);
        if have {
            println!("wrong, have conjugate item");
            println!("{:?},{:?}",R,hamR.slice(s![i,..]).to_owned())
        }
    }
    */
    //开始读取 .win 文件
    let mut reads:Vec<String>=Vec::new();
    let mut win_path=file_path.clone();
    win_path.push_str(".win"); //文件的位置
    let path=Path::new(&win_path); //转化为路径格式
    let hr=File::open(path).expect("Unable open the file, please check if have hr file");
    let reader = BufReader::new(hr);
    let mut reads:Vec<String>=Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        reads.push(line.clone());
    }
    let mut read_iter=reads.iter();
    let mut lat=Array2::<f64>::zeros((3,3)); //晶格轨道坐标初始化
    let mut spin:bool=false; //体系自旋初始化
    let mut natom:usize=0; //原子位置初始化
    let mut atom=Array2::<f64>::zeros((1,3)); //原子位置坐标初始化
    let mut proj_name:Vec<&str>=Vec::new();
    let mut proj_list:Vec<usize>=Vec::new();
    let mut atom_list:Vec<usize>=Vec::new();
    let mut atom_name:Vec<&str>=Vec::new();
    loop{
        let a=read_iter.next();
        if a==None{
            break;
        }else{
            let a=a.unwrap();
            if a.contains("begin unit_cell_cart") {
                let mut lat1=read_iter.next().unwrap().trim().split_whitespace(); //将数字放到
                let mut lat2=read_iter.next().unwrap().trim().split_whitespace();
                let mut lat3=read_iter.next().unwrap().trim().split_whitespace();
                for i in 0..3{
                    lat[[0,i]]=lat1.next().unwrap().parse::<f64>().unwrap();
                    lat[[1,i]]=lat2.next().unwrap().parse::<f64>().unwrap();
                    lat[[2,i]]=lat3.next().unwrap().parse::<f64>().unwrap();
                }
            } else if a.contains("spinors") && (a.contains("T") || a.contains("t")){
                spin=true;
            }else if a.contains("begin projections"){
                loop{
                    let string=read_iter.next().unwrap();
                    if string.contains("end projections"){
                        break
                    }else{ 
                        let prj:Vec<&str>=string.split(|c| c==',' || c==';' || c==':').collect();
                        let mut atom_orb_number:usize=0;
                        for item in prj[1..].iter(){
                            println!("{}",*item);
                            let aa:usize=match *item{
                                "s"=>1,
                                "p"=>3,
                                "d"=>5,
                                "f"=>7,
                                "sp3"=>4,
                                "sp2"=>3,
                                "sp"=>2,
                                "sp2d3"=>6,
                                "px"=>1,
                                "py"=>1,
                                "pz"=>1,
                                "dxy"=>1,
                                "dyz"=>1,
                                "dxz"=>1,
                                "dz2"=>1,
                                "dx2-y2"=>1,
                                &_=>panic!("Wrong, no matching"),
                            };
                            atom_orb_number+=aa;
                        }
                        proj_list.push(atom_orb_number);
                        proj_name.push(prj[0])
                    }
                }
            }else if a.contains("begin atoms_cart"){
                loop{
                    let string=read_iter.next().unwrap();
                    if string.contains("end atoms_cart"){
                        break
                    }else{       
                        let prj:Vec<&str>=string.split_whitespace().collect();
                        atom_name.push(prj[0])
                    }
                }
            }
        }
    }
    for name in atom_name.iter(){
        for (j,j_name) in proj_name.iter().enumerate(){
            if j_name==name{
                atom_list.push(proj_list[j])
            }
        }
    }
    natom=atom_list.len();
    println!("{:?}",atom_name);
    println!("{:?}",atom_list);
    println!("{}",atom_list.len());
    println!("{:?}",proj_list);
    println!("{:?}",proj_name);
    //开始读取 seedname_centres.xyz 文件
    let mut reads:Vec<String>=Vec::new();
    let mut xyz_path=file_path.clone();
    xyz_path.push_str("_centres.xyz");
    let path=Path::new(&xyz_path);
    let hr=File::open(path).expect("Unable open the file, please check if have hr file");
    let reader = BufReader::new(hr);
    let mut reads:Vec<String>=Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        reads.push(line.clone());
    }
    let nsta=reads[0].trim().parse::<usize>().unwrap()-natom;
    let norb=if spin{nsta/2}else{nsta};
    let mut orb=Array2::<f64>::zeros((norb,3));
    let mut atom=Array2::<f64>::zeros((natom,3));
    for i in 0..norb{
        let a:Vec<&str>=reads[i+2].trim().split_whitespace().collect();
        orb[[i,0]]=a[1].parse::<f64>().unwrap();
        orb[[i,1]]=a[2].parse::<f64>().unwrap();
        orb[[i,2]]=a[3].parse::<f64>().unwrap();
    }
    for i in 0..natom{
        let a:Vec<&str>=reads[i+2+nsta].trim().split_whitespace().collect();
        atom[[i,0]]=a[1].parse::<f64>().unwrap();
        atom[[i,1]]=a[2].parse::<f64>().unwrap();
        atom[[i,2]]=a[3].parse::<f64>().unwrap();
    }
    println!("{:?}",atom)
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
