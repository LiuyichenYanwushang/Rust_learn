use nalgebra::Complex;
use ndarray::linalg::kron;
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::*;
use std::f64::consts::PI;
use ndarray_linalg::{Eigh, UPLO};
use rayon::prelude::*;
use std::io::Write;
use std::fs::File;
///这个Model结构存储了一个TB模型所有的需要的信息
///
///dim_r:模型的维度, 这里我们不区分 dim_k 和 dim_r, 默认是一致的, 请使用者自行设置二维体系.
///
///norb:模型的轨道数目
///
///nsta:模型的态的数目, 如果开启自旋, nsta=norb*2
///
///natom:模型的原子数目, 后面的 atom 和 atom_list 是用来存储原子位置, 以及 每一个原子对应的轨道数目
///
///spin:模型是否开启自旋, 若开启, spin=true
///
///lat:模型的晶格矢量, 为一个 dim_r*dim_r大小的矩阵, axis0方向存储着1*dim_r的晶格矢量
///
///orb:模型的轨道位置, 我们统一使用分数坐标
///
///atom:模型的原子位置, 也是分数坐标
///
///atom_list:模型的原子中的轨道数目, 和原子位置的顺序一致.
///
///ham:模型的哈密顿量, <m0|H|nR>, 为一个 n_R*nsta*nsta的三维复数张量, 第一个nsta*nsta的矩阵对应的是原胞内的hopping,即 <m0|H|n0>, 后面对应的是 hamR中的hopping.
///
///hamR:模型的原胞间haopping的距离, 即 <m0|H|nR> 中的 R
///
///rmatrix:模型的位置矩阵, 即 <m0|r|nR>.
pub struct Model{
    pub dim_r:usize,                    //模型的实空间维度
    //pub dim_k:usize,                    //模型的k空间维度
    pub norb:usize,                     //模型的轨道数目
    pub nsta:usize,                     //模型的态的数目, 如果开启自旋, nsta=norb*2
    pub natom:usize,                    //模型的原子数目, 后面的 atom 和 atom_list 是用来存储原子位置, 以及 每一个原子对应的轨道数目
    pub spin:bool,                      //模型是否开启自旋, 若开启, spin=true
    pub lat:Array2::<f64>,              //模型的晶格矢量, 为一个 dim_r*dim_r大小的矩阵, axis0方向存储着1*dim_r的晶格矢量
    pub orb:Array2::<f64>,              //模型的轨道位置, 我们统一使用分数坐标
    pub atom:Array2::<f64>,             //模型的原子位置, 也是分数坐标.
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
    for i in 0..n_R{
        let mut a=true;
        for j in 0..dim_R{
            a=a&&(hamR[[i,j]]==R[[j]]);
        }
        if a{
            return i
        }
    }
    0
}
pub fn gen_kmesh(k_mesh:&Array1::<usize>)->Array2::<f64>{
    let dim:usize=k_mesh.len();
    let mut nk:usize=1;
    for i in 0..dim{
        nk*=k_mesh[[i]]-1;
    }
    fn gen_kmesh_arr(k_mesh:&Array1::<usize>,r0:usize,mut usek:Array1::<f64>)->Array2::<f64>{
        let dim:usize=k_mesh.len();
        let mut kvec=Array2::<f64>::zeros((0,dim));
        if r0==0{
            for i in 0..(k_mesh[[r0]]-1){
               let mut usek=Array1::<f64>::zeros(dim);
               usek[[r0]]=(i as f64)/((k_mesh[[r0]]-1) as f64);
               let k0:Array2::<f64>=gen_kmesh_arr(&k_mesh,r0+1,usek);
               kvec.append(Axis(0),k0.view()).unwrap();
            }
            return kvec
        }else if r0<k_mesh.len()-1{
            for i in 0..(k_mesh[[r0]]-1){
               let mut kk=usek.clone();
               kk[[r0]]=(i as f64)/((k_mesh[[r0]]-1) as f64);
               let k0:Array2::<f64>=gen_kmesh_arr(&k_mesh,r0+1,kk);
               kvec.append(Axis(0),k0.view()).unwrap();
            }
            return kvec
        }else{
            for i in 0..(k_mesh[[r0]]-1){
               usek[[r0]]=(i as f64)/((k_mesh[[r0]]-1) as f64);
               kvec.push_row(usek.view()).unwrap();
            }
            return kvec
        }
    }
    let mut usek=Array1::<f64>::zeros(dim);
    gen_kmesh_arr(&k_mesh,0,usek)
}
pub fn comm(A:&Array2::<Complex<f64>>,B:&Array2::<Complex<f64>>)->Array2::<Complex<f64>>{
    let A0=A.clone();
    let B0=B.clone();
    let C=A0.dot(&B0);
    let D=B0.dot(&A0);
    C-D
}
pub fn anti_comm(A:&Array2::<Complex<f64>>,B:&Array2::<Complex<f64>>)->Array2::<Complex<f64>>{
    let A0=A.clone();
    let B0=B.clone();
    let C=A0.dot(&B0)+B0.dot(&A0);
    C
}
impl Model{

    pub fn tb_model(dim_r:usize,norb:usize,lat:Array2::<f64>,orb:Array2::<f64>,spin:bool,natom:Option<usize>,atom:Option<Array2::<f64>>,atom_list:Option<Vec<usize>>)->Model{
        //!这个函数是用来初始化一个 Model, 需要输入的变量意义为

        //!模型维度 dim_r,
        //!轨道数目 norb,
        //!晶格常数 lat,
        //!轨道 orb,
        //!是否考虑自旋 spin,
        //!原子数目 natom, 可以选择 None,
        //!原子位置坐标 atom, 可以选择 None,
        //!每个原子的轨道数目, atom_list, 可以选择 None.
        //!
        //! 注意, 如果原子部分存在 None, 那么最好统一都是None.
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
        let ham=Array3::<Complex<f64>>::zeros((1,nsta,nsta));
        let hamR=Array2::<isize>::zeros((1,dim_r));
        let mut rmatrix=Array4::<Complex<f64>>::zeros((1,dim_r,nsta,nsta));
        for i in 0..norb {
            for r in 0..dim_r{
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
    pub fn set_hop(&mut self,tmp:Complex<f64>,ind_i:usize,ind_j:usize,R:Array1::<isize>,pauli:isize){
        if pauli != 0 && self.spin==false{
            println!("Wrong, if spin is Ture and pauli is not zero, the pauli is not use")
        }
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
                println!("Warning, the data of ham you input is {}, not zero, I hope you know what you are doing. If you want to eliminate this warning, use del_add to remove hopping.",self.ham[[index,ind_i,ind_j]])
            }
            if self.spin{
                match pauli{
                    0=>{self.ham[[index,ind_i,ind_j]]=tmp; self.ham[[index,ind_i+self.norb,ind_j+self.norb]]=tmp;},
                    1=>{self.ham[[index,ind_i+self.norb,ind_j]]=tmp; self.ham[[index,ind_i,ind_j+self.norb]]=tmp;},
                    2=>{self.ham[[index,ind_i+self.norb,ind_j]]=tmp*Complex::<f64>::i(); self.ham[[index,ind_i,ind_j+self.norb]]=-tmp*Complex::<f64>::i();},
                    3=>{self.ham[[index,ind_i,ind_j]]=tmp; self.ham[[index,ind_i+self.norb,ind_j+self.norb]]=-tmp;},
                    _ => todo!()
                }
            }else{
                self.ham[[index,ind_i,ind_j]]=tmp;
            }
            if index==0 && ind_i != ind_j{
                if self.spin{
                    match pauli{
                        0=>{self.ham[[0,ind_j,ind_i]]=tmp.conj(); self.ham[[0,ind_j+self.norb,ind_i+self.norb]]=tmp.conj();},
                        1=>{self.ham[[0,ind_j,ind_i+self.norb]]=tmp.conj(); self.ham[[0,ind_j+self.norb,ind_i]]=tmp.conj();},
                        2=>{self.ham[[0,ind_j,ind_i+self.norb]]=tmp.conj()*Complex::<f64>::i(); self.ham[[0,ind_j+self.norb,ind_i]]=-tmp.conj()*Complex::<f64>::i();},
                        3=>{self.ham[[0,ind_i,ind_j]]=tmp.conj(); self.ham[[0,ind_i+self.norb,ind_j+self.norb]]=-tmp.conj();},
                        _ => todo!()
                    }
                }else{
                    self.ham[[0,ind_j,ind_i]]=tmp.conj();
                }
            }
            if ind_i==ind_j && tmp.im !=0.0 && (pauli==0 ||pauli==3) && index==0{
                panic!("Wrong, the onsite hopping must be real, but here is {}",tmp)
            }
        }else if negative_R_exist {
            let index=index_R(&self.hamR,&negative_R);
            if self.ham[[index,ind_j,ind_i]]!=Complex::new(0.0,0.0){
                println!("Warning, the data of ham you input is {}, not zero, I hope you know what you are doing. If you want to eliminate this warning, use del_add to remove hopping.",self.ham[[index,ind_j,ind_i]])
            }
            if self.spin{
                match pauli{
                    0=>{self.ham[[index,ind_j,ind_i]]=tmp; self.ham[[index,ind_j+self.norb,ind_i+self.norb]]=tmp.conj();},
                    1=>{self.ham[[index,ind_j,ind_i+self.norb]]=tmp.conj(); self.ham[[index,ind_j+self.norb,ind_i]]=tmp.conj();},
                    2=>{self.ham[[index,ind_j,ind_i+self.norb]]=tmp.conj()*Complex::<f64>::i(); self.ham[[index,ind_j+self.norb,ind_i]]=-tmp.conj()*Complex::<f64>::i();},
                    3=>{self.ham[[index,ind_i,ind_j]]=tmp.conj(); self.ham[[index,ind_i+self.norb,ind_j+self.norb]]=-tmp.conj();},
                    _ => todo!()
                }
            }else{
                self.ham[[index,ind_j,ind_i]]=tmp.conj();
            }
        }else{
            let mut new_ham=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
            if self.spin{
                match pauli{
                    0=>{new_ham[[ind_i,ind_j]]=tmp; new_ham[[ind_i+self.norb,ind_j+self.norb]]=tmp;},
                    1=>{new_ham[[ind_i+self.norb,ind_j]]=tmp; new_ham[[ind_i,ind_j+self.norb]]=tmp;},
                    2=>{new_ham[[ind_i+self.norb,ind_j]]=tmp*Complex::<f64>::i(); new_ham[[ind_i,ind_j+self.norb]]=-tmp*Complex::<f64>::i();},
                    3=>{new_ham[[ind_i,ind_j]]=tmp; new_ham[[ind_i+self.norb,ind_j+self.norb]]=-tmp;},
                    _ => todo!()
                }
            }else{
                new_ham[[ind_i,ind_j]]=tmp;
            }
            self.ham.push(Axis(0),new_ham.view()).unwrap();
            self.hamR.push(Axis(0),R.view()).unwrap();
        }
    }

    pub fn add_hop(&mut self,tmp:Complex<f64>,ind_i:usize,ind_j:usize,R:Array1::<isize>,pauli:isize){
        if pauli != 0 && self.spin==false{
            println!("Wrong, if spin is Ture and pauli is not zero, the pauli is not use")
        }
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
            if self.spin{
                match pauli{
                    0=>{self.ham[[index,ind_i,ind_j]]+=tmp; self.ham[[index,ind_i+self.norb,ind_j+self.norb]]+=tmp;},
                    1=>{self.ham[[index,ind_i+self.norb,ind_j]]+=tmp; self.ham[[index,ind_i,ind_j+self.norb]]+=tmp;},
                    2=>{self.ham[[index,ind_i+self.norb,ind_j]]+=tmp*Complex::<f64>::i(); self.ham[[index,ind_i,ind_j+self.norb]]-=tmp*Complex::<f64>::i();},
                    3=>{self.ham[[index,ind_i,ind_j]]+=tmp; self.ham[[index,ind_i+self.norb,ind_j+self.norb]]-=tmp;},
                    _ => todo!()
                }
            }else{
                self.ham[[index,ind_i,ind_j]]+=tmp;
            }
            if index==0 && ind_i !=ind_j{
                if self.spin{
                    match pauli{
                        0=>{self.ham[[0,ind_j,ind_i]]+=tmp.conj(); self.ham[[0,ind_j+self.norb,ind_i+self.norb]]+=tmp.conj();},
                        1=>{self.ham[[0,ind_j,ind_i+self.norb]]+=tmp.conj(); self.ham[[0,ind_j+self.norb,ind_i]]+=tmp.conj();},
                        2=>{self.ham[[0,ind_j,ind_i+self.norb]]+=tmp.conj()*Complex::<f64>::i(); self.ham[[0,ind_j+self.norb,ind_i]]-=tmp.conj()*Complex::<f64>::i();},
                        3=>{self.ham[[0,ind_i,ind_j]]+=tmp.conj(); self.ham[[0,ind_i+self.norb,ind_j+self.norb]]-=tmp.conj();},
                        _ => todo!()
                    }
                }else{
                    self.ham[[0,ind_j,ind_i]]+=tmp.conj();
                }
            }
            if ind_i==ind_j && tmp.im !=0.0 && (pauli==0 ||pauli==3) && index==0{
                panic!("Wrong, the onsite hopping must be real, but here is {}",tmp)
            }
        }else if negative_R_exist {
            let index=index_R(&self.hamR,&negative_R);
            if self.ham[[index,ind_j,ind_i]]!=Complex::new(0.0,0.0){
                println!("Warning, the data of ham you input is {}, not zero, I hope you know what you are doing. If you want to eliminate this warning, use del_add to remove hopping.",self.ham[[index,ind_j,ind_i]])
            }
            if self.spin{
                match pauli{
                    0=>{self.ham[[index,ind_j,ind_i]]+=tmp; self.ham[[index,ind_j+self.norb,ind_i+self.norb]]+=tmp.conj();},
                    1=>{self.ham[[index,ind_j,ind_i+self.norb]]+=tmp.conj(); self.ham[[index,ind_j+self.norb,ind_i]]+=tmp.conj();},
                    2=>{self.ham[[index,ind_j,ind_i+self.norb]]+=tmp.conj()*Complex::<f64>::i(); self.ham[[index,ind_j+self.norb,ind_i]]-=tmp.conj()*Complex::<f64>::i();},
                    3=>{self.ham[[index,ind_i,ind_j]]+=tmp.conj(); self.ham[[index,ind_i+self.norb,ind_j+self.norb]]-=tmp.conj();},
                    _ => todo!()
                }
            }else{
                self.ham[[index,ind_j,ind_i]]+=tmp.conj();
            }
        }else{
            let mut new_ham=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
            if self.spin{
                match pauli{
                    0=>{new_ham[[ind_i,ind_j]]+=tmp; new_ham[[ind_i+self.norb,ind_j+self.norb]]+=tmp;},
                    1=>{new_ham[[ind_i+self.norb,ind_j]]+=tmp; new_ham[[ind_i,ind_j+self.norb]]+=tmp;},
                    2=>{new_ham[[ind_i+self.norb,ind_j]]+=tmp*Complex::<f64>::i(); new_ham[[ind_i,ind_j+self.norb]]-=tmp*Complex::<f64>::i();},
                    3=>{new_ham[[ind_i,ind_j]]+=tmp; new_ham[[ind_i+self.norb,ind_j+self.norb]]-=tmp;},
                    _ => todo!()
                }
            }else{
                new_ham[[ind_i,ind_j]]+=tmp;
            }
            self.ham.push(Axis(0),new_ham.view()).unwrap();
            self.hamR.push(Axis(0),R.view()).unwrap();
        }
    }
    pub fn set_onsite(&mut self, tmp:Array1::<f64>,pauli:isize){
        if tmp.len()!=self.norb{
            panic!("Wrong, the norb is {}, however, the onsite input's length is {}",self.norb,tmp.len())
        }
        for (i,item) in tmp.iter().enumerate(){
            self.set_onsite_one(*item,i,pauli)
        }
    }
    pub fn set_onsite_one(&mut self, tmp:f64,ind:usize,pauli:isize){
        let R=Array1::<isize>::zeros(self.dim_r);
        self.set_hop(Complex::new(tmp,0.0),ind,ind,R,pauli)
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
            //let dk=path.slice(s![n,..]).to_owned()-path.slice(s![n-1,..]).to_owned();
            let dk=path.row(n).to_owned()-path.slice(s![n-1,..]).to_owned();
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
        //k_vec.slice_mut(s![0,..]).assign(&path.slice(s![0,..]));
        k_vec.row_mut(0).assign(&path.row(0));
        for n in 1..n_node {
            let n_i=node_index[n-1];
            let n_f=node_index[n];
            let kd_i=k_node[[n-1]];
            let kd_f=k_node[[n]];
            //let k_i=path.slice(s![n-1,..]);
            //let k_f=path.slice(s![n,..]);
            let k_i=path.row(n-1);
            let k_f=path.row(n);
            for j in n_i..n_f+1{
                let frac:f64= ((j-n_i) as f64)/((n_f-n_i) as f64);
                k_dist[[j]]=kd_i + frac*(kd_f-kd_i);
                //k_vec.slice_mut(s![j,..]).assign(&((1.0-frac)*k_i.to_owned() +frac*k_f.to_owned()));
                k_vec.row_mut(j).assign(&((1.0-frac)*k_i.to_owned() +frac*k_f.to_owned()));

            }
        }
        (k_vec,k_dist,k_node)
    }
    ///这个是做傅里叶变换, 将实空间的哈密顿量变换到倒空间的哈密顿量
    pub fn gen_ham(&self,kvec:&Array1::<f64>)->Array2::<Complex<f64>>{
        if kvec.len() !=self.dim_r{
            panic!("Wrong, the k-vector's length must equal to the dimension of model.")
        }
        let nR:usize=self.hamR.len_of(Axis(0));
        let U0:Array1::<f64>=self.orb.dot(kvec);
        let U0:Array1::<Complex<f64>>=U0.map(|x| Complex::<f64>::new(*x,0.0));
        let U0=U0*Complex::new(0.0,2.0*PI);
        let mut U0:Array1::<Complex<f64>>=U0.mapv(Complex::exp);
        if self.spin{
            let UU=U0.clone();
            U0.append(Axis(0),UU.view()).unwrap();
        }
        let U=Array2::from_diag(&U0);
        let Us=(self.hamR.map(|x| *x as f64)).dot(kvec).map(|x| Complex::<f64>::new(*x,0.0));
        let Us=Us*Complex::new(0.0,2.0*PI);
        let Us=Us.mapv(Complex::exp);
        let mut hamk=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
        let ham0=self.ham.slice(s![0,..,..]).to_owned();
        for i in 1..nR{
            hamk=hamk+self.ham.slice(s![i,..,..]).to_owned()*Us[[i]];
        }
        hamk=ham0+hamk.map(|x| x.conj()).t()+hamk;
        hamk=hamk.dot(&U);
        let re_ham=U.map(|x| x.conj()).t().dot(&hamk);
        re_ham
    }
    pub fn gen_r(&self,kvec:&Array1::<f64>)->Array3::<Complex<f64>>{
        if kvec.len() !=self.dim_r{
            panic!("Wrong, the k-vector's length must equal to the dimension of model.")
        }
        let nR:usize=self.hamR.len_of(Axis(0));
        let U0=self.orb.dot(kvec);
        let U0=U0.map(|x| Complex::<f64>::new(*x,0.0));
        let U0=U0*Complex::new(0.0,2.0*PI);
        let mut U0=U0.mapv(Complex::exp);
        if self.spin{
            let UU=U0.clone();
            U0.append(Axis(0),UU.view()).unwrap();
        }
        let U=Array2::from_diag(&U0);
        let Us=(self.hamR.map(|x| *x as f64)).dot(kvec).map(|x| Complex::<f64>::new(*x,0.0));
        let Us=Us*Complex::new(0.0,2.0*PI);
        let Us=Us.mapv(Complex::exp);
        let mut rk=Array3::<Complex<f64>>::zeros((self.dim_r,self.nsta,self.nsta));
        let r0=self.rmatrix.slice(s![0,..,..,..]).to_owned();
        if self.rmatrix.len_of(Axis(0))==1{
            return self.rmatrix.slice(s![0,..,..,..]).to_owned()
        }else{
            for i in 1..nR{
                rk=rk+self.rmatrix.slice(s![i,..,..,..]).to_owned()*Us[[i]];
            }
            for i in 0..self.dim_r{
                let use_rk=rk.slice(s![i,..,..]).to_owned();
                let use_rk:Array2::<Complex<f64>>=r0.slice(s![i,..,..]).to_owned()+use_rk.map(|x| x.conj()).t()+use_rk;
                //接下来向位置算符添加轨道的位置项
                let use_rk=use_rk.dot(&U); //
                rk.slice_mut(s![i,..,..]).assign(&(U.map(|x| x.conj()).t().dot(&use_rk)));
            }
            return rk
        }
    }
    ///这个函数是用来生成速度算符的, 即 $<u_{mk}|\p_\alpha H_k|u_{nk}>$
    pub fn gen_v(&self,kvec:&Array1::<f64>)->Array3::<Complex<f64>>{
        if kvec.len() !=self.dim_r{
            panic!("Wrong, the k-vector's length must equal to the dimension of model.")
        }
        let nR:usize=self.hamR.len_of(Axis(0));
        let U0=self.orb.dot(kvec);
        let U0=U0.map(|x| Complex::<f64>::new(*x,0.0));
        let U0=U0*Complex::new(0.0,2.0*PI);
        let mut U0=U0.mapv(Complex::exp);
        if self.spin{
            let UU=U0.clone();
            U0.append(Axis(0),UU.view()).unwrap();
        }
        let U=Array2::from_diag(&U0);
        let Us=(self.hamR.map(|x| *x as f64)).dot(kvec).map(|x| Complex::<f64>::new(*x,0.0));
        let Us=Us*Complex::new(0.0,2.0*PI);
        let Us=Us.mapv(Complex::exp);
        let mut UU=Array3::<f64>::zeros((self.dim_r,self.nsta,self.nsta));
        let orb_real=self.orb.dot(&self.lat);
        for r in 0..self.dim_r{
            for i in 0..self.norb{
                for j in 0..self.norb{
                    UU[[r,i,j]]=-orb_real[[i,r]]+orb_real[[j,r]];
                    if self.spin{
                        UU[[r,i+self.norb,j]]=-orb_real[[i,r]]+orb_real[[j,r]];
                        UU[[r,i,j+self.norb]]=-orb_real[[i,r]]+orb_real[[j,r]];
                        UU[[r,i+self.norb,j+self.norb]]=-orb_real[[i,r]]+orb_real[[j,r]];
                    }
                }
            }
        }
        let UU=UU.map(|x| Complex::<f64>::new(0.0,*x)); //UU[i,j]=-tau[i]+tau[j] 
        let mut v=Array3::<Complex<f64>>::zeros((self.dim_r,self.nsta,self.nsta));//定义一个初始化的速度矩阵
        let ham0=self.ham.slice(s![0,..,..]).to_owned();
        let R0=self.hamR.clone().map(|x| Complex::<f64>::new((*x) as f64,0.0));
        let R0=R0.dot(&self.lat.map(|x| Complex::new(*x,0.0)));
        for i0 in 0..self.dim_r{
            let mut hamk=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
            let mut vv=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
            for i in 1..nR{
                vv=vv+self.ham.slice(s![i,..,..]).to_owned()*Us[[i]]*R0[[i,i0]]*Complex::i(); //这一步对 R 求和
                hamk=hamk+self.ham.slice(s![i,..,..]).to_owned()*Us[[i]];
            }
            vv=vv.clone().reversed_axes().map(|x| x.conj())+vv;
            let hamk0=hamk.clone();
            let hamk=hamk+self.ham.slice(s![0,..,..]).to_owned()+hamk0.map(|x| x.conj()).t();
            vv=vv+hamk.clone()*UU.slice(s![i0,..,..]).to_owned();
            let vv=vv.dot(&U); //加下来两步填上轨道坐标导致的相位
            let vv=U.map(|x| x.conj()).t().dot(&vv);
            v.slice_mut(s![i0,..,..]).assign(&vv);
        }
        //到这里, 我们完成了 sum_{R} iR H_{mn}(R) e^{ik(R+tau_n-tau_m)} 的计算
        //接下来, 我们计算贝利联络 A_\alpha=\sum_R r(R)e^{ik(R+tau_n-tau_m)}-tau
        if self.rmatrix.len_of(Axis(0))!=1 {
            let hamk=self.gen_ham(&kvec);
            let rk=self.gen_r(&kvec); 
            for i in 0..self.dim_r{
                let mut UU=self.orb.slice(s![..,i]).to_owned().clone(); //我们首先提取出alpha方向的轨道的位置
                if self.spin{
                    let UUU=UU.clone();
                    UU.append(Axis(0),UUU.view()).unwrap();
                }
                let mut UU=Array2::from_diag(&UU); //将其化为矩阵
                let A=rk.slice(s![i,..,..]).to_owned()-UU;
                let A=comm(&hamk,&A)*Complex::i();
                let vv=v.slice(s![i,..,..]).to_owned().clone();
                v.slice_mut(s![i,..,..]).assign(&(vv+A));
            }
        }
        v
    }

    pub fn solve_band_onek(&self,kvec:&Array1::<f64>)->Array1::<f64>{
        if kvec.len() !=self.dim_r{
            panic!("Wrong, the k-vector's length:k_len={} must equal to the dimension of model:{}.",kvec.len(),self.dim_r)
        } 
        let hamk=self.gen_ham(&kvec);
        let hamk_conj=hamk.clone().map(|x| x.conj());
        let hamk_conj=hamk_conj.t();
        let sum0=(hamk.clone()-hamk_conj).sum();
        if sum0.im()> 1e-8 || sum0.re() >1e-8{
            panic!("Wrong, hamiltonian is not hamilt");
        }
        let eval = if let Ok(eigvals) = hamk.eigvalsh(UPLO::Lower) { eigvals } else { todo!() };
        eval
    }
    pub fn solve_band_all(&self,kvec:&Array2::<f64>)->Array2::<f64>{
        let nk=kvec.len_of(Axis(0));
        let mut band=Array2::<f64>::zeros((nk,self.nsta));
        for i in 0..nk{
            //let k=kvec.slice(s![i,..]).to_owned();
            let k=kvec.row(i).to_owned();
            let eval=self.solve_band_onek(&k);
            band.slice_mut(s![i,..]).assign(&eval);
        }
        band
    }
    pub fn solve_band_all_parallel(&self,kvec:&Array2::<f64>)->Array2::<f64>{
        let nk=kvec.len_of(Axis(0));
        let eval:Vec<_>=kvec.axis_iter(Axis(0)).into_par_iter().map(|x| {
            let eval=self.solve_band_onek(&x.to_owned()); 
            eval.to_vec()
            }).collect();
        let band = Array2::from_shape_vec((nk, self.nsta), eval.into_iter().flatten().collect()).unwrap();
        band
    }
    pub fn solve_onek(&self,kvec:&Array1::<f64>)->(Array1::<f64>,Array2::<Complex<f64>>){
        if kvec.len() !=self.dim_r{
            panic!("Wrong, the k-vector's length:k_len={} must equal to the dimension of model:{}.",kvec.len(),self.dim_r)
        } 
        let hamk=self.gen_ham(&kvec);
        let hamk_conj=hamk.clone().map(|x| x.conj());
        let hamk_conj=hamk_conj.t();
        let sum0=(hamk.clone()-hamk_conj).sum();
        if sum0.im()> 1e-8 || sum0.re() >1e-8{
            panic!("Wrong, hamiltonian is not hamilt");
        }
        let (eval, evec) = if let Ok((eigvals, eigvecs)) = hamk.eigh(UPLO::Lower) { (eigvals, eigvecs) } else { todo!() };
        let evec=evec.reversed_axes().map(|x| x.conj());
        (eval,evec)
    }
    pub fn solve_all(&self,kvec:&Array2::<f64>)->(Array2::<f64>,Array3::<Complex<f64>>){
        let nk=kvec.len_of(Axis(0));
        let mut band=Array2::<f64>::zeros((nk,self.nsta));
        let mut vectors=Array3::<Complex<f64>>::zeros((nk,self.nsta,self.nsta));
        for i in 0..nk{
            //let k=kvec.slice(s![i,..]).to_owned();
            let k=kvec.row(i).to_owned();
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
    /*
    pub fn cut_piece(&self,num:usize,dir:usize)->Model{
        if num<1{
            panic!("Wrong, the num={} is less than 1");
        }
        let new_orb=Array2::<f64>::zeros((self.norb*num,self.dim_r));
        let new_norb=self.norb*num;
        let new_nsta=self.nsta*num;
        let mut new_lat=self.lat;
        new_lat.slice_mut(s![dir,..]).assign(self.lat.slice(s![dir,..])*(num as f64));
        for i in 0..num{
            for n in 0..self.norb{
                let mut use_orb=self.orb.slice(s![n,..]);
                use_orb[[dir]]*=1.0/(num as f64);
                new_orb.slice_mut(s![i*self.norb+n,..]).assign(use_orb);
            }
        }

    }
    */
    pub fn dos(&self,k_mesh:&Array1::<usize>,E_min:f64,E_max:f64,E_n:usize,sigma:f64)->(Array1::<f64>,Array1::<f64>){
        let kvec:Array2::<f64>=gen_kmesh(&k_mesh);
        let band=self.solve_band_all_parallel(&kvec);
        let E0=Array1::linspace(E_min,E_max,E_n);
        let mut dos=Array1::<f64>::zeros(E_n);
        let mut nk:usize=1;
        let dim:usize=k_mesh.len();
        for i in 0..dim{
            nk*=k_mesh[[i]]-1;
        }
        let mut centre=Array1::<f64>::zeros(0);
        for i in 0..nk{
            centre.append(Axis(0),band.row(i)).unwrap();
        }
        let sigma0=1.0/sigma;
        let pi0=1.0/(2.0*PI).sqrt();
        for i in centre.iter(){
            dos.map(|x| x+pi0*sigma0*(-((x-i)*sigma0).powi(2)/2.0).exp());
        }
        dos=dos/(nk as f64);
        (E0,dos)
    }

    pub fn berry_curvature_onek(&self,k_vec:&Array1::<f64>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,T:f64,og:f64,mu:f64,spin:usize,eta:f64)->f64{
        let li:Complex<f64>=1.0*Complex::i();
        let (band,evec)=self.solve_onek(&k_vec);
        let mut v:Array3::<Complex<f64>>=self.gen_v(k_vec);
        let mut J:Array3::<Complex<f64>>=v.clone();
        if self.spin {
            let mut X:Array2::<Complex<f64>>=Array2::eye(self.nsta);
            let pauli:Array2::<Complex<f64>>= match spin{
                0=> arr2(&[[1.0+0.0*li,0.0+0.0*li],[0.0+0.0*li,1.0+0.0*li]]),
                1=> arr2(&[[0.0+0.0*li,1.0+0.0*li],[1.0+0.0*li,0.0+0.0*li]])/2.0,
                2=> arr2(&[[0.0+0.0*li,0.0-1.0*li],[0.0+1.0*li,0.0+0.0*li]])/2.0,
                3=> arr2(&[[1.0+0.0*li,0.0+0.0*li],[0.0+0.0*li,-1.0+0.0*li]])/2.0,
                _=>panic!("Wrong, spin should be 0, 1, 2, 3, but you input {}",spin),
            };
            X=kron(&pauli,&Array2::eye(self.norb));
            for i in 0..self.dim_r{
                let j=J.slice(s![i,..,..]).to_owned().clone();
                let j=anti_comm(&X,&j)/2.0; //这里做反对易
                J.slice_mut(s![i,..,..]).assign(&(j*dir_1[[i]]));
                let v0=v.slice(s![i,..,..]).to_owned().clone();
                v.slice_mut(s![i,..,..]).assign(&(v0*dir_2[[i]]));
            }
        }else{ 
            if spin !=0{
                println!("Warning, the model haven't got spin, so the spin input will be ignord");
            }
            for i in 0..self.dim_r{
                let j=J.slice(s![i,..,..]).to_owned().clone();
                J.slice_mut(s![i,..,..]).assign(&(j*dir_1[[i]]));
                let v0=v.slice(s![i,..,..]).to_owned().clone();
                v.slice_mut(s![i,..,..]).assign(&(v0*dir_2[[i]]));
            }
        };

        let J:Array2::<Complex<f64>>=J.sum_axis(Axis(0));
        let v:Array2::<Complex<f64>>=v.sum_axis(Axis(0));
        let evec_conj:Array2::<Complex<f64>>=evec.clone().map(|x| x.conj()).to_owned();
        let A1=J.dot(&evec.clone().reversed_axes());
        let A1=evec_conj.clone().dot(&A1);
        let A2=v.dot(&evec.reversed_axes());
        let A2=evec_conj.dot(&A2);
 //       let A1=J.dot(&evec_conj.clone().reversed_axes());
 //       let A1=evec.clone().dot(&A1);
 //       let A2=v.dot(&evec_conj.reversed_axes());
 //       let A2=evec.dot(&A2);
        let mut U0=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
        for i in 0..self.nsta{
            for j in 0..self.nsta{
                if i != j{
                    U0[[i,j]]=1.0/((band[[i]]-band[[j]]).powi(2)-(og+li*eta).powi(2));
                }else{
                    U0[[i,j]]=Complex::new(0.0,0.0);
                }
            }
        }
        let omega_n:Array1::<f64>=(-Complex::new(2.0,0.0)*(A1*U0).dot(&A2)).diag().map(|x| x.im).to_owned();
        let mut omega:f64=0.0;
        if T==0.0{
            for i in 0..self.nsta{
                omega+= if band[[i]]> mu {0.0} else {omega_n[[i]]};
            }
        }else{
            let beta=1.0/(T*8.617e-5);
            let fermi_dirac=band.map(|x| 1.0/((beta*(x-mu)).exp()+1.0));
            omega=(omega_n*fermi_dirac).sum();
        }
        omega
    }
    //这个是用来并行计算大量k点的贝利曲率
    pub fn berry_curvature(&self,k_vec:&Array2::<f64>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,T:f64,og:f64,mu:f64,spin:usize,eta:f64)->Array1::<f64>{
        if dir_1.len() !=self.dim_r || dir_2.len() != self.dim_r{
            panic!("Wrong, the dir_1 or dir_2 you input has wrong length, it must equal to dim_r={}, but you input {},{}",self.dim_r,dir_1.len(),dir_2.len())
        }
        let nk=k_vec.len_of(Axis(0));
        let omega:Vec<f64>=k_vec.axis_iter(Axis(0)).into_par_iter().map(|x| {
            let omega_one=self.berry_curvature_onek(&x.to_owned(),&dir_1,&dir_2,T,og,mu,spin,eta); 
            omega_one
            }).collect();
        let omega=arr1(&omega);
        omega
    }
    pub fn Hall_conductivity(&self,k_mesh:&Array1::<usize>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,T:f64,og:f64,mu:f64,spin:usize,eta:f64)->f64{
        let kvec:Array2::<f64>=gen_kmesh(&k_mesh);
        let nk:usize=kvec.len_of(Axis(0));
        let omega=self.berry_curvature(&kvec,&dir_1,&dir_2,T,og,mu,spin,eta);
        let conductivity:f64=omega.sum()/(nk as f64)*(2.0*PI).powi(self.dim_r as i32)/self.lat.det().unwrap();
        conductivity
    }
    pub fn show_band(&self,path:&Array2::<f64>,label:&Vec<&str>,nk:usize,name:&str)-> std::io::Result<()>{
        use std::fs::create_dir_all;
        use std::path::Path;
        if path.len_of(Axis(0))!=label.len(){
            panic!("Error, the path's length {} and label's length {} must be equal!",path.len_of(Axis(0)),label.len())
        }
        let (k_vec,k_dist,k_node)=self.k_path(&path,nk);
        let eval=self.solve_band_all_parallel(&k_vec);
        create_dir_all(name).expect("can't creat the file");
        let mut name0=String::new();
        name0.push_str("./");
        name0.push_str(&name);
        let name=name0;
        let mut band_name=name.clone();
        band_name.push_str("/BAND.dat");
        let band_name=Path::new(&band_name);
        let mut file=File::create(band_name).expect("Unable to BAND.dat");
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
        let mut k_name=name.clone();
        k_name.push_str("/KLABELS");
        let k_name=Path::new(&k_name);
        let mut file=File::create(k_name).expect("Unable to create KLBAELS");//写下高对称点的位置
        for i in 0..path.len_of(Axis(0)){
            let mut s=String::new();
            let aa= format!("{:.6}", k_node[[i]]);
            s.push_str(&aa);
            s.push_str("      ");
            s.push_str(&label[i]);
            writeln!(file,"{}",s)?;
        }
        let mut py_name=name.clone();
        py_name.push_str("/print.py");
        let py_name=Path::new(&py_name);
        let mut file=File::create(py_name).expect("Unable to create print.py");
        writeln!(file,"import numpy as np\nimport matplotlib.pyplot as plt\ndata=np.loadtxt('BAND.dat')\nk_nodes=[]\nlabel=[]\nf=open('KLABELS')\nfor i in f.readlines():\n    k_nodes.append(float(i.split()[0]))\n    label.append(i.split()[1])\nfig,ax=plt.subplots()\nax.plot(data[:,0],data[:,1:])\nfor x in k_nodes:\n    ax.axvline(x,c='k')\nax.set_xticks(k_nodes)\nax.set_xticklabels(label)\nax.set_xlim([0,k_nodes[-1]])\nfig.savefig('band.pdf')");
        Ok(())
    }
    pub fn from_hr(path:&str,file_name:&str,zero_energy:f64)->Model{
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
            if string.len() !=15{
                n_line=i+1;
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
                if a==0 && b==0 && c==0{
                    for ind_i in 0..nsta{
                        for ind_j in 0..nsta{
                            let mut string=reads[i*nsta*nsta+ind_i*nsta+ind_j+n_line].trim().split_whitespace();
                            let re=string.nth(5).unwrap().parse::<f64>().unwrap();
                            let im=string.next().unwrap().parse::<f64>().unwrap();
                            ham[[0,ind_j,ind_i]]=Complex::new(re,im)/(weights[i] as f64);
                        }
                    }
                }else{
                    let mut matrix=Array3::<Complex<f64>>::zeros((1,nsta,nsta));
                    for ind_i in 0..nsta{
                        for ind_j in 0..nsta{
                            let mut string=reads[i*nsta*nsta+ind_i*nsta+ind_j+n_line].trim().split_whitespace();
                            let re=string.nth(5).unwrap().parse::<f64>().unwrap();
                            let im=string.next().unwrap().parse::<f64>().unwrap();
                            matrix[[0,ind_j,ind_i]]=Complex::new(re,im)/(weights[i] as f64); //wannier90 里面是按照纵向排列的矩阵
                        }
                    }
                    ham.append(Axis(0),matrix.view()).unwrap();
                    hamR.append(Axis(0),arr2(&[[a,b,c]]).view()).unwrap();
                }
            }
        }
        for i in 0..nsta{
            ham[[0,i,i]]-=Complex::new(zero_energy,0.0);
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
                                let aa:usize=match *item{
                                    "s"=>1,
                                    "p"=>3,
                                    "d"=>5,
                                    "f"=>7,
                                    "sp3"=>4,
                                    "sp2"=>3,
                                    "sp"=>2,
                                    "sp3d2"=>6,
                                    "px"=>1,
                                    "py"=>1,
                                    "pz"=>1,
                                    "dxy"=>1,
                                    "dxz"=>1,
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
        //let nsta=reads[0].trim().parse::<usize>().unwrap()-natom;
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
        orb=orb.dot(&lat.inv().unwrap());
        atom=atom.dot(&lat.inv().unwrap());
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
                        let mut matrix=Array4::<Complex<f64>>::zeros((1,3,nsta,nsta));
                        for ind_i in 0..nsta{
                            for ind_j in 0..nsta{
                                let mut string=reads[i*nsta*nsta+ind_i*nsta+ind_j+3].trim().split_whitespace();
                                string.nth(4);
                                for r in 0..3{
                                    let re=string.next().unwrap().parse::<f64>().unwrap();
                                    let im=string.next().unwrap().parse::<f64>().unwrap();
                                    matrix[[0,r,ind_i,ind_j]]=Complex::new(re,im);
                                }
                            }
                        }
                        rmatrix.append(Axis(0),matrix.view()).unwrap();
                    }
                }
            }
        }else{
           for i in 0..norb {
                for r in 0..3{
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

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Complex;
    use ndarray::prelude::*;
    use ndarray::*;
    #[test]
    fn anti_comm_test(){
        let a=array![[1.0,2.0,3.0],[0.0,1.0,0.0],[0.0,0.0,0.0]];
        let b=array![[1.0,0.0,0.0],[1.0,1.0,0.0],[2.0,0.0,1.0]];
        let c=a.dot(&b)+b.dot(&a);
        println!("{}",c)
    }
    #[test]
    fn Haldan_model(){
        let li:Complex<f64>=1.0*Complex::i();
        let t=-1.0+0.0*li;
        let t2=1.0+0.0*li;
        let delta=0.7;
        let dim_r:usize=2;
        let norb:usize=2;
        let lat=arr2(&[[1.0,0.0],[0.5,3.0_f64.sqrt()/2.0]]);
        let orb=arr2(&[[1.0/3.0,1.0/3.0],[2.0/3.0,2.0/3.0]]);
        let mut model=Model::tb_model(dim_r,norb,lat,orb,false,None,None,None);
        model.set_onsite(arr1(&[-delta,delta]),0);
        let R0:Array2::<isize>=arr2(&[[0,0],[-1,0],[0,-1]]);
        for (i,R) in R0.axis_iter(Axis(0)).enumerate(){
            let R=R.to_owned();
            model.add_hop(t,0,1,R,0);
        }
        let R0:Array2::<isize>=arr2(&[[1,0],[-1,1],[0,-1]]);
        for (i,R) in R0.axis_iter(Axis(0)).enumerate(){
            let R=R.to_owned();
            model.add_hop(t2*li,0,0,R,0);
        }
        let R0:Array2::<isize>=arr2(&[[-1,0],[1,-1],[0,1]]);
        for (i,R) in R0.axis_iter(Axis(0)).enumerate(){
            let R=R.to_owned();
            model.add_hop(t2*li,1,1,R,0);
        }
        let nk:usize=1001;
        let path=[[0.0,0.0],[2.0/3.0,1.0/3.0],[0.5,0.5],[1.0/3.0,2.0/3.0],[0.0,0.0]];
        let path=arr2(&path);
        let (k_vec,k_dist,k_node)=model.k_path(&path,nk);
        let (eval,evec)=model.solve_all_parallel(&k_vec);
        let label=vec!["G","K","M","K'","G"];
        model.show_band(&path,&label,nk,"tests/Haldan");
        /////开始计算体系的霍尔电导率//////
        let nk:usize=101;
        let T:f64=0.0;
        let eta:f64=0.001;
        let og:f64=0.0;
        let mu:f64=0.0;
        let dir_1=arr1(&[1.0,0.0]);
        let dir_2=arr1(&[0.0,1.0]);
        let spin:usize=0;
        let kmesh=arr1(&[nk,nk]);
        let (eval,evec)=model.solve_onek(&arr1(&[0.3,0.5]));
        let conductivity=model.Hall_conductivity(&kmesh,&dir_1,&dir_2,T,og,mu,spin,eta);
        println!("{}",conductivity/(2.0*PI));

    }
    #[test]
    fn graphene(){
        let li:Complex<f64>=1.0*Complex::i();
        let t1=1.0+0.0*li;
        let t2=0.1+0.0*li;
        let t3=0.0+0.0*li;
        let delta=0.0;
        let dim_r:usize=2;
        let norb:usize=2;
        let lat=arr2(&[[2.0,0.0],[1.0,3.0_f64.sqrt()]]);
        let orb=arr2(&[[0.0,0.0],[1.0/3.0,1.0/3.0]]);
        let mut model=Model::tb_model(dim_r,norb,lat,orb,false,None,None,None);
        model.set_onsite(arr1(&[delta,-delta]),0);
        model.add_hop(t1,0,1,array![0,0],0);
        model.add_hop(t1,0,1,array![-1,0],0);
        model.add_hop(t1,0,1,array![0,-1],0);
        model.add_hop(t2,0,0,array![1,0],0);
        model.add_hop(t2,1,1,array![1,0],0);
        model.add_hop(t2,0,0,array![0,1],0);
        model.add_hop(t2,1,1,array![0,1],0);
        model.add_hop(t2,0,0,array![1,-1],0);
        model.add_hop(t2,1,1,array![1,-1],0);
        model.add_hop(t3,0,1,array![1,-1],0);
        model.add_hop(t3,0,1,array![-1,1],0);
        model.add_hop(t3,0,1,array![-1,-1],0);
        let nk:usize=1001;
        let path=[[0.0,0.0],[2.0/3.0,1.0/3.0],[0.5,0.5],[0.0,0.0]];
        let path=arr2(&path);
        let (k_vec,k_dist,k_node)=model.k_path(&path,nk);
        let (eval,evec)=model.solve_all_parallel(&k_vec);
        let label=vec!["G","K","M","G"];
        model.show_band(&path,&label,nk,"tests/graphene");
        /////开始计算体系的霍尔电导率//////
        let nk:usize=101;
        let T:f64=0.0;
        let eta:f64=0.001;
        let og:f64=0.0;
        let mu:f64=0.0;
        let dir_1=arr1(&[1.0,0.0]);
        let dir_2=arr1(&[0.0,1.0]);
        let spin:usize=0;
        let kmesh=arr1(&[nk,nk]);
        let (eval,evec)=model.solve_onek(&arr1(&[0.3,0.5]));
        let conductivity=model.Hall_conductivity(&kmesh,&dir_1,&dir_2,T,og,mu,spin,eta);
        println!("{}",conductivity/(2.0*PI));
    }

    #[test]
    fn kane_mele(){
        let li:Complex<f64>=1.0*Complex::i();
        let delta=0.7;
        let t=-1.0+0.0*li;
        let rashba=0.0+0.0*li;
        let soc=-0.24+0.0*li;
        let dim_r:usize=2;
        let norb:usize=2;
        let lat=arr2(&[[1.0,0.0],[0.5,3.0_f64.sqrt()/2.0]]);
        let orb=arr2(&[[1.0/3.0,1.0/3.0],[2.0/3.0,2.0/3.0]]);
        let mut model=Model::tb_model(dim_r,norb,lat,orb,true,None,None,None);
        model.set_onsite(arr1(&[delta,-delta]),0);
        let R0:Array2::<isize>=arr2(&[[0,0],[-1,0],[0,-1]]);
        for (i,R) in R0.axis_iter(Axis(0)).enumerate(){
            let R=R.to_owned();
            model.set_hop(t,0,1,R,0);
        }
        let R0:Array2::<isize>=arr2(&[[1,0],[-1,1],[0,-1]]);
        for (i,R) in R0.axis_iter(Axis(0)).enumerate(){
            let R=R.to_owned();
            model.set_hop(soc*li,0,0,R,3);
        }
        let R0:Array2::<isize>=arr2(&[[-1,0],[1,-1],[0,1]]);
        for (i,R) in R0.axis_iter(Axis(0)).enumerate(){
            let R=R.to_owned();
            model.set_hop(soc*li,1,1,R,3);
        }
        /*
        model.add_hop(li*rashba*0.5,0,1,array![0,0],1);
        model.add_hop(-li*rashba*3.0_f64.sqrt()/2.0,0,1,array![0,0],2);
        model.add_hop(li*rashba*0.5,0,1,array![-2,0],1);
        model.add_hop(li*rashba*3.0_f64.sqrt()/2.0,0,1,array![-2,0],2);
        model.add_hop(-li*rashba,0,1,array![0,-2],1);
        */
        let nk:usize=1001;
        let path=[[0.0,0.0],[2.0/3.0,1.0/3.0],[0.5,0.5],[1.0/3.0,2.0/3.0],[0.0,0.0]];
        let path=arr2(&path);
        let (k_vec,k_dist,k_node)=model.k_path(&path,nk);
        let (eval,evec)=model.solve_all_parallel(&k_vec);
        let label=vec!["G","K","M","K'","G"];
        model.show_band(&path,&label,nk,"tests/kane");
        /////开始计算体系的霍尔电导率//////
        let nk:usize=101;
        let T:f64=0.0;
        let eta:f64=0.001;
        let og:f64=0.0;
        let mu:f64=0.0;
        let dir_1=arr1(&[3.0_f64.sqrt()/2.0,-0.5]);
        let dir_2=arr1(&[0.0,1.0]);
        let spin:usize=3;
        let kmesh=arr1(&[nk,nk]);
        let (eval,evec)=model.solve_onek(&arr1(&[0.3,0.5]));
        let conductivity=model.Hall_conductivity(&kmesh,&dir_1,&dir_2,T,og,mu,spin,eta);
        println!("{}",conductivity/(2.0*PI));
    }
}

