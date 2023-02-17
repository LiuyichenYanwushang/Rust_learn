mod Rustb{
    use nalgebra::Complex;
    use ndarray::prelude::*;
    use ndarray::*;
    use ndarray_linalg::*;
    use std::f64::consts::PI;
    pub struct Model{
        pub dim_r:usize,
        pub norb:usize,
        pub nsta:usize,
        pub natom:usize,
        pub spin:bool,
        pub lat:Array2::<f64>,
        pub orb:Array2::<f64>,
        pub atom:Array2::<f64>,
        pub atom_list:Vec<usize>,
        pub ham:Array3::<Complex<f64>>,
        pub hamR:Array2::<isize>,
        pub rmatrix:Array3::<Complex<f64>>
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
            let mut rmatrix=Array3::<Complex<f64>>::zeros((dim_r,nsta,nsta));
            for i in 1..norb {
                for r in 1..dim_r{
                    rmatrix[[r,i,i]]=Complex::<f64>::from(orb[[i,r]]);
                    if spin{
                        rmatrix[[r,i+norb,i+norb]]=Complex::<f64>::from(orb[[i,r]]);
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
                self.ham[[index,ind_i,ind_j]]=tmp;
                if index==0{
                    self.ham[[index,ind_j,ind_i]]=tmp.conj();
                    if ind_i==ind_j && tmp.im !=0.0 {
                        panic!("Wrong, the onsite hopping must be real")
                    }
                }
            }else if negative_R_exist {
                let index=index_R(&self.hamR,&negative_R);
                self.ham[[index,ind_j,ind_i]]=tmp.conj();
            }else{
                let mut new_ham=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
                new_ham[[ind_i,ind_j]]=tmp;
                self.ham.push(Axis(0),new_ham.view()).unwrap();
                self.hamR.push(Axis(0),R.view()).unwrap();
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
            let U0=(self.orb.dot(kvec)*Complex::new(0.0,2.0*PI)).mapv(f64::exp);
            let U=Array2::from_diag(&U0);
            let ham0=self.ham.slice(s![0,..,..]).to_owned();
            let hams=self.ham.slice(s![1..nR,..,..]).to_owned();
            let Us=(self.hamR.dot(kvec)*Complex::new(0.0,2.0*PI)).mapv(f64::exp);
            let mut hamk=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
            for i in 1..nR{
                hamk+=self.hamR.slice(s![i,..,..]).to_owned()*Us[[i]]
            }
            hamk+=ham0+hamk.t();
            hamk=hamk.dot(&U);
            let re_ham=U.t().mapv(|x| x.conj());
            re_ham
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
        let nk:usize=100;
        let path=[[0.0,0.0],[1.0/3.0,1.0/3.0],[0.5,0.0],[0.0,0.0]];
        let path=arr2(&path);
        let (k_vec,k_dist,k_node)=model.k_path(&path,nk);
        println!("{:?}",k_vec);
        println!("{:?}",k_dist);
        println!("{:?}",k_node);
    }
}

