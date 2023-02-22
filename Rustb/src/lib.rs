mod Rustb{
    use nalgebra::Complex;
    use ndarray::prelude::*;
    use ndarray::*;
    use ndarray_linalg::*;
    use std::f64::consts::PI;
    use std::fs::File;
    use std::io::Write;
    use ndarray_linalg::{Eigh, UPLO};
    use rayon::prelude::*;
    ///这个Model结构存储了一个TB模型所有的需要的信息
    pub struct Model{
        pub dim_r:usize,                    //模型的维度, 这里我们不区分 dim_k 和 dim_r, 默认是一致的, 请使用者自行设置二维体系.
        pub norb:usize,                     //模型的轨道数目
        pub nsta:usize,                     //模型的态的数目, 如果开启自旋, nsta=norb*2
        pub natom:usize,                    //模型的原子数目, 后面的 atom 和 atom_list 是用来存储原子位置, 以及 每一个原子对应的轨道数目
        pub spin:bool,                      //模型是否开启自旋, 若开启, spin=true
        pub lat:Array2::<f64>,              //模型的晶格矢量, 为一个 dim_r*dim_r大小的矩阵, axis0方向存储着1*dim_r的晶格矢量
        pub orb:Array2::<f64>,              //模型的轨道位置, 我们统一使用分数坐标
        pub atom:Array2::<f64>,             //模型的原子位置, 也是分数坐标
        pub atom_list:Vec<usize>,           //模型的原子中的轨道数目, 和原子位置的顺序一致.
        pub ham:Array3::<Complex<f64>>,     //模型的哈密顿量, <m0|H|nR>, 为一个 n_R*nsta*nsta的三维复数张量, 第一个nsta*nsta的矩阵对应的是原胞内的hopping,即 <m0|H|n0>, 后面对应的是 hamR中的hopping.
        pub hamR:Array2::<isize>,           //模型的原胞间haopping的距离, 即 <m0|H|nR> 中的 R
        pub rmatrix:Array4::<Complex<f64>>  //模型的位置矩阵, 即 <m0|r|nR>.
    }
    pub fn find_R(hamR:&Array2::<isize>,R:&Array1::<isize>)->bool{
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
    pub fn index_R(hamR:&Array2::<isize>,R:&Array1::<isize>)->usize{
        let n_R:usize=hamR.len_of(Axis(0));
        let dim_R:usize=hamR.len_of(Axis(1));
        for i in 0..(n_R-1){

            let mut a=true;
            for j in 0..(dim_R-1){
                a=a&&(hamR[[i,j]]==R[[j]]);
            }
            if a{
                return i
            }
        }
        0
    }
    impl Model{
        //!这个函数是用来创建一个 Model, 需要输入的变量意义为
        //!模型维度 dim_r
        //!轨道数目 norb
        //!晶格常数 lat
        //!轨道 orb
        //!原子数目 natom, 可以选择 None
        //!原子位置坐标 atom, 可以选择 None
        //!每个原子的轨道数目, atom_list
        //!是否考虑自旋 spin
        pub fn tb_model(dim_r:usize,norb:usize,lat:Array2::<f64>,orb:Array2::<f64>,natom:Option<usize>,atom:Option<Array2::<f64>>,atom_list:Option<Vec<usize>>,spin:bool)->Model{
            let mut nsta:usize=norb;
            if spin{
                nsta*=2;
            }
            let mut new_natom:usize=0;
            let mut new_atom_list:Vec<usize>=vec![1];
            let mut new_atom:Array2::<f64>=arr2(&[[0.0]]);
            if lat.len_of(Axis(1)) != dim_r{
                panic!("Wrong, the lat's second dimension's length must equal to dim_r") 
            }
            if lat.len_of(Axis(0))<lat.len_of(Axis(1)) {
                panic!("Wrong, the lat's second dimension's length must less than first dimension's length") 
            }
            if natom==None{
               if atom !=None && atom_list !=None{
                    let use_natom:usize=atom.as_ref().unwrap().len_of(Axis(0)).try_into().unwrap();
                    if use_natom != atom_list.as_ref().unwrap().len().try_into().unwrap(){
                        panic!("Wrong, the length of atom_list is not equal to the natom");
                    }
                    new_natom=use_natom;
                }else if atom_list !=None || atom != None{
                    panic!("Wrong, the atom and atom_list is not all None, please correspondence them");
                }else if atom_list==None && atom==None{
                    new_natom=norb.clone();
                    new_atom=orb.clone();
                    new_atom_list=vec![1;new_natom.try_into().unwrap()];
                } else{
                    new_natom=norb.clone();
                    println!("{:?},2",natom);
                };
            }else{
                new_natom=natom.unwrap();
                if atom_list==None || atom==None{
                    panic!("Wrong, the atom and atom_list is None but natom is not none")
                }else{
                    new_atom=atom.unwrap();
                    new_atom_list=atom_list.unwrap();
                }
            }
            let mut ham=Array3::<Complex<f64>>::zeros((1,nsta,nsta));
            let mut hamR=Array2::<isize>::zeros((1,dim_r));
            let mut rmatrix=Array4::<Complex<f64>>::zeros((1,dim_r,nsta,nsta));
            for i in 1..norb {
                for r in 1..dim_r{
                    rmatrix[[0,r,i,i]]=Complex::<f64>::from(orb[[i,r]]);
                    if spin{
                        rmatrix[[0,r,i+norb,i+norb]]=Complex::<f64>::from(orb[[i,r]]);
                    }
                }
            }
            let mut model=Model{
                dim_r,
                norb,
                nsta,
                natom:new_natom,
                spin,
                lat,
                orb,
                atom:new_atom,
                atom_list:new_atom_list,
                ham,
                hamR,
                rmatrix,
            };
            model
        }

        pub fn add_hop(&mut self,tmp:Complex<f64>,ind_i:usize,ind_j:usize,R:Array1::<isize>) {
            if R.len()!=self.dim_r{
                panic!("Wrong, the R length should equal to dim_r")
            }
            if ind_i>=self.norb ||ind_j>=self.norb{
                panic!("Wrong, ind_i and ind_j must less than norb, here norb is {}, but ind_i={} and ind_j={}",self.norb,ind_i,ind_j)
            }
            let R_exist=find_R(&self.hamR,&R);
            let negative_R=-R.clone();
            let negative_R_exist=find_R(&self.hamR,&negative_R);
            if R_exist {
                let index=index_R(&self.hamR,&R);
                if self.ham[[index,ind_i,ind_j]]!=Complex::new(0.0,0.0){
                    println!("Warning, the data of ham you input is not zero, I hope you know what you are doing. If you want to eliminate this warning, use del_add to remove hopping.")
                }
                self.ham[[index,ind_i,ind_j]]=tmp;
                if index==0{
                    self.ham[[index,ind_j,ind_i]]=tmp.conj();
                    if ind_i==ind_j && tmp.im !=0.0 {
                        panic!("Wrong, the onsite hopping must be real")
                    }
                }
            }else if negative_R_exist {
                let index=index_R(&self.hamR,&negative_R);
                if self.ham[[index,ind_j,ind_i]]!=Complex::new(0.0,0.0){
                    println!("Warning, the data of ham you input is not zero, I hope you know what you are doing. If you want to eliminate this warning, use del_add to remove hopping.")
                }
                self.ham[[index,ind_j,ind_i]]=tmp.conj();
            }else{
                let mut new_ham=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
                new_ham[[ind_i,ind_j]]=tmp;
                self.ham.push(Axis(0),new_ham.view()).unwrap();
                self.hamR.push(Axis(0),R.view()).unwrap();
            }
        }
        pub fn del_hop(&mut self,ind_i:usize,ind_j:usize,R:Array1::<isize>) {
            if R.len()!=self.dim_r{
                panic!("Wrong, the R length should equal to dim_r")
            }
            if ind_i>=self.norb ||ind_j>=self.norb{
                panic!("Wrong, ind_i and ind_j must less than norb, here norb is {}, but ind_i={} and ind_j={}",self.norb,ind_i,ind_j)
            }
            let R_exist=find_R(&self.hamR,&R);
            let negative_R=-R.clone();
            let negative_R_exist=find_R(&self.hamR,&negative_R);
            if R_exist {
                let index=index_R(&self.hamR,&R);
                self.ham[[index,ind_i,ind_j]]=Complex::new(0.0,0.0);
                if index==0{
                    self.ham[[index,ind_j,ind_i]]=Complex::new(0.0,0.0);
                }
            }else if negative_R_exist {
                let index=index_R(&self.hamR,&negative_R);
                self.ham[[index,ind_j,ind_i]]=Complex::new(0.0,0.0);
            }
        }
        pub fn k_path(&self,path:&Array2::<f64>,nk:usize)->(Array2::<f64>,Array1::<f64>,Array1::<f64>){
            if self.dim_r==0{
                panic!("the k dimension of the model is 0, do not use k_path")
            }
            let n_node:usize=path.len_of(Axis(0));
            if self.dim_r != path.len_of(Axis(1)){
                panic!("Wrong, the path's length along 1 dimension must equal to the model's dimension")
            }
            let k_metric=(self.lat.dot(&self.lat.t())).inv().unwrap();
            let mut k_node=Array1::<f64>::zeros(n_node);
            for n in 1..n_node{
                let dk=path.slice(s![n,..]).to_owned()-path.slice(s![n-1,..]).to_owned();
                let a=k_metric.dot(&dk);
                let dklen=dk.dot(&a).sqrt();
                k_node[[n]]=k_node[[n-1]]+dklen;
            }
            let mut node_index:Vec<usize>=vec![0];
            for n in 1..n_node-1{
                let frac=k_node[[n]]/k_node[[n_node-1]];
                let a=(frac*((nk-1) as f64).round()) as usize;
                node_index.push(a)
            }
            node_index.push(nk-1);
            let mut k_dist=Array1::<f64>::zeros(nk);
            let mut k_vec=Array2::<f64>::zeros((nk,self.dim_r));
            k_vec.slice_mut(s![0,..]).assign(&path.slice(s![0,..]));
            for n in 1..n_node {
                let n_i=node_index[n-1];
                let n_f=node_index[n];
                let kd_i=k_node[[n-1]];
                let kd_f=k_node[[n]];
                let k_i=path.slice(s![n-1,..]);
                let k_f=path.slice(s![n,..]);
                for j in n_i..n_f+1{
                    let frac:f64= ((j-n_i) as f64)/((n_f-n_i) as f64);
                    k_dist[[j]]=kd_i + frac*(kd_f-kd_i);
                    k_vec.slice_mut(s![j,..]).assign(&((1.0+frac)*k_i.to_owned() +frac*k_f.to_owned()));
                }
            }
            (k_vec,k_dist,k_node)
        }
        pub fn gen_ham(&self,kvec:&Array1::<f64>)->Array2::<Complex<f64>>{
            if kvec.len() !=self.dim_r{
                panic!("Wrong, the k-vector's length must equal to the dimension of model.")
            }
            let nR:usize=self.hamR.len_of(Axis(0));
            let U0=self.orb.dot(kvec);
            let U0=U0.map(|x| Complex::<f64>::new(*x,0.0));
            let U0=U0*Complex::new(0.0,2.0*PI);
            let U0=U0.mapv(Complex::exp);
            let U=Array2::from_diag(&U0);
            let Us=self.hamR.map(|x| *x as f64).dot(kvec).map(|x| Complex::<f64>::new(*x,0.0));
            let Us=Us*Complex::new(0.0,2.0*PI);
            let Us=Us.mapv(Complex::exp);
            let mut hamk=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
            let ham0=self.ham.slice(s![0,..,..]).to_owned();
            for i in 1..nR{
                hamk=hamk+self.ham.slice(s![i,..,..]).to_owned()*Us[[i]];
            }
            hamk=ham0+hamk.t()+hamk;
            hamk=hamk.dot(&U);
            let re_ham=U.t().mapv(|x| x.conj());
            re_ham
        }
        pub fn solve_onek(&self,kvec:&Array1::<f64>)->(Array1::<f64>,Array2::<Complex<f64>>){
            if kvec.len() !=self.dim_r{
                panic!("Wrong, the k-vector's length:k_len={} must equal to the dimension of model:{}.",kvec.len(),self.dim_r)
            } 
            let hamk=self.gen_ham(&kvec);
            let (eval, evec) = if let Ok((eigvals, eigvecs)) = hamk.eigh(UPLO::Lower) { (eigvals, eigvecs) } else { todo!() };
            let evec=evec.reversed_axes();
            (eval,evec)
        }
        pub fn solve_all(&self,kvec:&Array2::<f64>)->(Array2::<f64>,Array3::<Complex<f64>>){
            let nk=kvec.len_of(Axis(0));
            let mut band=Array2::<f64>::zeros((nk,self.nsta));
            let mut vectors=Array3::<Complex<f64>>::zeros((nk,self.nsta,self.nsta));
            for i in 0..nk{
                let k=kvec.slice(s![i,..]).to_owned();
                let (eval,evec)=self.solve_onek(&k);
                band.slice_mut(s![i,..]).assign(&eval);
                vectors.slice_mut(s![i,..,..]).assign(&evec);
            }
            (band,vectors)
        }
        pub fn solve_all_parallel(&self,kvec:&Array2::<f64>)->(Array2::<f64>,Array3::<Complex<f64>>){
            let nk=kvec.len_of(Axis(0));
            let (eval,evec):(Vec<_>,Vec<_>)=kvec.axis_iter(Axis(0)).into_par_iter().map(|x| {
                let (eval, evec) =self.solve_onek(&x.to_owned()); 
                (eval.to_vec(),evec.into_raw_vec())
                }).collect();
            let band = Array2::from_shape_vec((nk, self.nsta), eval.into_iter().flatten().collect()).unwrap();
            let vectors=Array3::from_shape_vec((nk, self.nsta,self.nsta), evec.into_iter().flatten().collect()).unwrap();
            (band,vectors)
        }
        pub fn show_band(&self,path:&Array2::<f64>,nk:usize)-> std::io::Result<()>{
            let (k_vec,k_dist,k_node)=self.k_path(&path,nk);
            let (eval,evec)=self.solve_all(&k_vec);
            let mut file=File::create("BAND.dat").expect("Unable to create file");
            for i in 0..nk{
                let mut s = String::new();
                let aa= format!("{:.6}", k_dist[[i]]);
                s.push_str(&aa);
                for j in 0..self.nsta{
                    if eval[[i,j]]>=0.0 {
                        s.push_str("     ");
                    }else{
                        s.push_str("    ");
                    }
                    let aa= format!("{:.6}", eval[[i,j]]);
                    s.push_str(&aa);
                }
                writeln!(file,"{}",s)?;
            }
            let mut file=File::create("KLABELS").expect("Unable to create file");
            for i in 0..path.len_of(Axis(0)){
                let mut s=String::new();
                let aa= format!("{:.6}", k_node[[i]]);
                s.push_str(&aa);
                writeln!(file,"{}",s)?;
            }
            Ok(())
        }
        pub fn from_hr(path:&str,file_name:&str)->Model{
            use std::io::BufReader;
            use std::io::BufRead;
            use std::fs::File;
            use std::path::Path;
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
            let n_R=reads[2].trim().parse::<usize>().unwrap();
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
            //开始尝试读取 _r.dat 文件


            let mut reads:Vec<String>=Vec::new();
            let mut r_path=file_path.clone();
            r_path.push_str("_r.dat");
            let mut rmatrix=Array4::<Complex<f64>>::zeros((1,3,nsta,nsta));
            let path=Path::new(&r_path);
            let hr=File::open(path);
             
            if hr.is_ok(){
                let hr=hr.unwrap();
                let reader = BufReader::new(hr);
                let mut reads:Vec<String>=Vec::new();
                for line in reader.lines() {
                    let line = line.unwrap();
                    reads.push(line.clone());
                }
                let n_R=reads[2].trim().parse::<usize>().unwrap();
                for i in 0..n_R{
                    let mut string=reads[i*nsta*nsta+3].trim().split_whitespace();
                    let a=string.next().unwrap().parse::<isize>().unwrap();
                    let b=string.next().unwrap().parse::<isize>().unwrap();
                    let c=string.next().unwrap().parse::<isize>().unwrap();
                    if (c>0) || (c==0 && b>0) ||(c==0 && b==0 && a>=0){
                        println!("a={},b={},c={}",a,b,c);
                        if a==0 && b==0 && c==0{
                            for ind_i in 0..nsta{
                                for ind_j in 0..nsta{
                                    let mut string=reads[i*nsta*nsta+ind_i*nsta+ind_j+3].trim().split_whitespace();
                                    string.nth(4);
                                    for r in 0..3{
                                        let re=string.next().unwrap().parse::<f64>().unwrap();
                                        let im=string.next().unwrap().parse::<f64>().unwrap();
                                        rmatrix[[0,r,ind_i,ind_j]]=Complex::new(re,im);
                                    }
                                }
                            }
                        }else{
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
            }else{
               for i in 1..norb {
                    for r in 1..3{
                        rmatrix[[0,r,i,i]]=Complex::<f64>::from(orb[[i,r]]);
                        if spin{
                            rmatrix[[0,r,i+norb,i+norb]]=Complex::<f64>::from(orb[[i,r]]);
                        }
                    }
                }
            }
            let mut model=Model{
                dim_r:3,
                norb,
                nsta,
                natom,
                spin,
                lat,
                orb,
                atom,
                atom_list,
                ham,
                hamR,
                rmatrix,
            };
            model
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Rustb::Model;
    use nalgebra::Complex;
    use ndarray::prelude::*;
    use ndarray::*;
    #[test]
    fn check_tb_model_function(){
        let li:Complex<f64>=1.0*Complex::i();
        let dim_r:usize=2;
        let norb:usize=2;
        let lat=arr2(&[[1.0,0.0],[3.0_f64.sqrt()/2.0,0.5]]);
        let orb=arr2(&[[0.0,0.0],[1.0/3.0,1.0/3.0]]);
        let mut model=Model::tb_model(dim_r,norb,lat,orb,None,None,None,false);
        println!("{}",model.lat);
        println!("{}",model.dim_r);
        println!("{}",dim_r);
        model.add_hop(3.0+1.0*li,0,1,array![0,0]);
        println!("{}",model.ham);
        let nk:usize=1000;
        let path=[[0.0,0.0],[1.0/3.0,1.0/3.0],[0.5,0.0],[0.0,0.0]];
        let path=arr2(&path);
        let (k_vec,k_dist,k_node)=model.k_path(&path,nk);
/*
        println!("{:?}",k_vec);
        println!("{:?}",k_dist);
        println!("{:?}",k_node);
*/
       // let kvec=k_vec.slice(s![0,..]).to_owned();
        let (eval,evec)=model.solve_all_parallel(&k_vec);
        println!("{:?}",eval);
        model.show_band(&path,nk);
    }
}

