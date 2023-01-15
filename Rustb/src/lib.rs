mod Rustb{
    use nalgebra::Complex;
    use ndarray::prelude::*;
    use ndarray::*;

    pub struct Model{
        pub dim_r:u64,
        pub dim_k:u64,
        pub norb:u64,
        pub natom:u64,
        pub spin:bool,
        pub lat:Array2::<f64>,
        pub orb:Array2::<f64>,
        pub atom:Option<Array2::<f64>>,
        pub atom_list:Option<Vec<u64>>,
        pub ham:Option<Array3::<Complex<f64>>>,
        pub hamR:Option<Array2::<i64>>
    }
    impl Model{
        pub fn tb_model(dim_r:u64,dim_k:u64,norb:u64,natom:u64,lat:Array2::<f64>,orb:Array2::<f64>,atom:Option<Array2::<f64>>,atom_list:Option<Vec<u64>>,spin:bool)->Model{
            let li=1.0*Complex::i();
            let a:u64=0;
            let c0=0.0+0.0*li;
            if spin{
                let nsta:u64=norb*2;
            }else{
                let nsta:u64=norb;
            }
            //let ham=Array::zeros((0,nsta,nsta))+Array::zeros((0,nsta,nsta))*li;
            let mut model=Model{
                dim_r,
                dim_k,
                norb,
                natom,
                spin,
                lat,
                orb,
                atom,
                atom_list,
                ham:None,
                hamR:None,
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
    fn check_model_struct(){
        let model=Model{
            dim_r:2,
            dim_k:2,
            norb:2,
            natom:1,
            spin:false,
            lat:arr2(&[[1.2,1.3],[2.1,2.3]]),
            orb:arr2(&[[1.2,1.3],[2.1,2.3]]),
            atom:None,
            atom_list:None,
            ham:Some(arr3(&[[[1.+0.*Complex::i(),0.0*Complex::i()],[0.0*Complex::i(),Complex::i()]]])),
            hamR:Some(arr2(&[[0,0]]))
        };
        println!("{}",model.ham.as_ref().expect("REASON").slice(s![0,..,..]));
        println!("{:?}",model.ham.as_ref().expect("REASON").shape());
    }
    #[test]
    fn check_tb_model_function(){
        let dim_r:u64=3;
        let dim_k:u64=2;
        let norb:u64=5;
        let natom:u64=5;
        let lat=arr2(&[[1.0,0.0,0.0],[3.0_f64.sqrt()/2.0,0.5,0.0],[0.0,0.0,1.0]]);
        let orb=arr2(&[[0.0,0.0,0.0],[0.5,0.0,0.5],[0.5,0.5,0.0],[0.0,0.5,0.5],[0.5,0.5,0.5]]);
        let model=Model::tb_model(dim_r,dim_k,norb,natom,lat,orb,None,None,false);
        println!("{}",model.lat);
    }
}

