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
use std::thread;
use std::thread::spawn;
use partial_application::partial;
use std::time::{Duration, Instant};
fn main() {
    //let k_range=arr2(&[[0.0,1.0],[0.0,1.0]]);
    let k_range=arr2(&[[0.0,1.0],[0.0,1.0],[0.0,1.0]]);
    let re_err=5e-3;
    let ab_err=5e-3;
    /*
    let start = Instant::now();   // 开始计时
    let a:f64=adapted_integrate(&test_func,&k_range,re_err,ab_err);
    let end = Instant::now();    // 结束计时
    let duration = end.duration_since(start); // 计算执行时间
    println!("function_a took {} seconds", duration.as_secs_f64());   // 输出执行时间
    println!("{}",a);

    let start = Instant::now();   // 开始计时
    let a:f64=adapted_integrate_loop(&test_func,&k_range,re_err,ab_err);
    let end = Instant::now();    // 结束计时
    let duration = end.duration_since(start); // 计算执行时间
    println!("function_b took {} seconds", duration.as_secs_f64());   // 输出执行时间
    println!("{}",a);
    */
    let start = Instant::now();   // 开始计时
    let a:f64=adapted_integrate_quick(&test_func,&k_range,re_err,ab_err);
    let end = Instant::now();    // 结束计时
    let duration = end.duration_since(start); // 计算执行时间
    println!("function_c took {} seconds", duration.as_secs_f64());   // 输出执行时间
    println!("{}",a);
}
pub fn adapted_integrate(f0:&dyn Fn(&Array1::<f64>)->f64,k_range:&Array2::<f64>,re_err:f64,ab_err:f64)->f64{
    ///对于任意维度的积分 n, 我们的将区域刨分成 n+1面体的小块, 然后用线性插值来近似这个n+1的积分结果
    ///
    ///积分的公式为: 
    ///
    ///设被积函数为 f(x1,x2,...,xn), 存在n+1个点 (y01,y02,...y0n)...(yn1,yn2... ynn), 对应的值为 z0,z1,...,zn
    ///
    ///这样我们就能得到 1/(n+1)! *\sum_{i}^n z_i *dV, dV 是正 n+1面体的体积.

    let dim=k_range.len_of(Axis(0));
    if dim==1{
        //对于一维情况, 我们就是用梯形算法的 (a+b)*h/2, 这里假设的是函数的插值为线性插值.
        let kvec_l:Array1::<f64>=arr1(&[k_range[[0,0]]]);
        let kvec_r:Array1::<f64>=arr1(&[k_range[[0,1]]]);
        let kvec_m:Array1::<f64>=arr1(&[(k_range[[0,1]]+k_range[[0,0]])/2.0]);
        let dk:f64=k_range[[0,1]]-k_range[[0,0]];
        let y_l:f64=f0(&kvec_l);
        let y_r:f64=f0(&kvec_r);
        let y_m:f64=f0(&kvec_m);
        let all:f64=(y_l+y_r)*dk/2.0;
        let all_1=(y_l+y_m)*dk/4.0;
        let all_2=(y_r+y_m)*dk/4.0;
        let err=all_1+all_2-all;
        let abs_err= if ab_err>all*re_err{ab_err} else {re_err};
        if err< abs_err{
            return all_1+all_2;
        }else{
            let k_range_l=arr2(&[[kvec_l[[0]],kvec_m[[0]]]]);
            let k_range_r=arr2(&[[kvec_m[[0]],kvec_r[[0]]]]);
            let all_1=adapted_integrate(f0,&k_range_l,re_err/2.0,ab_err/2.0);
            let all_2=adapted_integrate(f0,&k_range_r,re_err/2.0,ab_err/2.0);
            return all_1+all_2;
        }
    }else if dim==2{
    //对于二维, 我们依旧假设线性插值, 这样我们考虑的就是二维平面上的三角形上的任意一点的值是到其余三个点的距离的加权系数的平均值, 我们将四边形变成两个三角形来考虑.
        let area_1:Array2::<f64>=arr2(&[[k_range.row(0)[0],k_range.row(1)[0]],[k_range.row(0)[1],k_range.row(1)[0]],[k_range.row(0)[0],k_range.row(1)[1]]]);//第一个三角形
        let area_2:Array2::<f64>=arr2(&[[k_range.row(0)[1],k_range.row(1)[1]],[k_range.row(0)[1],k_range.row(1)[0]],[k_range.row(0)[0],k_range.row(1)[1]]]);//第二个三角形
        fn cal_integrate_2D(f0:&dyn Fn(&Array1::<f64>)->f64,kvec:&Array2::<f64>)->f64{
            //这个是用来进行线性插值积分的结果, 给出三个点和函数, 计算得到对应的插值积分结果
            let mut S:Array2::<f64>=kvec.clone();
            S.push(Axis(1),Array1::ones(3).view());
            let S:f64=S.det().expect("Wrong, S'det is 0").abs();
            let mut all:f64=0.0;
            for i in 0..kvec.len_of(Axis(0)){
                all+=f0(&kvec.row(i).to_owned())
            }
            all*=S;
            all=all/6.0;
            all
        }
        fn adapt_integrate_triangle(f0:&dyn Fn(&Array1::<f64>)->f64,kvec:&Array2::<f64>,re_err:f64,ab_err:f64)->f64{
            //这个函数是用来进行自适应算法的
            let mut S:Array2::<f64>=kvec.clone();
            S.push(Axis(1),Array1::ones(3).view());
            let S:f64=S.det().expect("Wrong, S'det is 0").abs();

            let all=cal_integrate_2D(f0,&kvec);
            let kvec_m=kvec.mean_axis(Axis(0)).unwrap();
            let mut kvec_1=Array2::<f64>::zeros((0,2));
            kvec_1.push_row(kvec.row(0));
            kvec_1.push_row(kvec.row(1));
            kvec_1.push_row(kvec_m.view());
            let mut kvec_2=Array2::<f64>::zeros((0,2));
            kvec_2.push_row(kvec.row(0));
            kvec_2.push_row(kvec_m.view());
            kvec_2.push_row(kvec.row(2));
            let mut kvec_3=Array2::<f64>::zeros((0,2));
            kvec_3.push_row(kvec_m.view());
            kvec_3.push_row(kvec.row(1));
            kvec_3.push_row(kvec.row(2));
            let all_1=cal_integrate_2D(f0,&kvec_1);
            let all_2=cal_integrate_2D(f0,&kvec_2);
            let all_3=cal_integrate_2D(f0,&kvec_3);
            let all_new=all_1+all_2+all_3;
            let abs_err:f64= if ab_err>all*re_err{ab_err} else {re_err};
            /*
            println!("kvec={}",kvec);
            println!("all={}",all);
            println!("all_1={}",all_1);
            println!("all_2={}",all_2);
            println!("all_3={}",all_3);
            println!("all_new={}",all_new);
            println!("err={}",abs_err);
            println!("real_err={}",(all_new-all));
            */
            if (all_new-all).abs()> abs_err && S>1e-8{
               //let all_1=adapt_integrate_triangle(f0,&kvec_1,re_err/3.0,ab_err/3.0);
               //let all_2=adapt_integrate_triangle(f0,&kvec_2,re_err/3.0,ab_err/3.0);
               //let all_3=adapt_integrate_triangle(f0,&kvec_3,re_err/3.0,ab_err/3.0);
               //all_new=all_1+all_2+all_3;
               return adapt_integrate_triangle(f0,&kvec_1,re_err/3.0,ab_err/3.0)+adapt_integrate_triangle(f0,&kvec_2,re_err/3.0,ab_err/3.0)+adapt_integrate_triangle(f0,&kvec_3,re_err/3.0,ab_err/3.0);
            }else{
                return all_new;
            }
        }
        let all_1=adapt_integrate_triangle(f0,&area_1,re_err/2.0,ab_err/2.0);
        let all_2=adapt_integrate_triangle(f0,&area_2,re_err/2.0,ab_err/2.0);
        return all_1+all_2;
    }else if dim==3{
    //对于三位情况, 需要用到四面体, 所以需要先将6面体变成6个四面体
        fn cal_integrate_3D(f0:&dyn Fn(&Array1::<f64>)->f64,kvec:&Array2::<f64>)->f64{
            //这个是用来进行线性插值积分的结果, 给出三个点和函数, 计算得到对应的插值积分结果
            let mut S=kvec.clone();
            S.push(Axis(1),Array1::ones(4).view());
            let S:f64=S.det().expect("Wrong, S'det is 0").abs();
            let mut all:f64=0.0;
            for i in 0..kvec.len_of(Axis(0)){
                all+=f0(&kvec.row(i).to_owned())
            }
            all*=S;
            all=all/24.0;
            all
        }
        fn adapt_integrate_tetrahedron(f0:&dyn Fn(&Array1::<f64>)->f64,kvec:&Array2::<f64>,re_err:f64,ab_err:f64)->f64{
            //这个函数是用来进行自适应算法的
            let mut S=kvec.clone();
            S.push(Axis(1),Array1::ones(4).view());
            let S:f64=S.det().expect("Wrong, S'det is 0").abs();//先求一下体积

            let all=cal_integrate_3D(f0,&kvec);
            let kvec_m=kvec.mean_axis(Axis(0)).unwrap();
            /////////////////////////
            let mut kvec_1=Array2::<f64>::zeros((0,3));
            kvec_1.push_row(kvec.row(0));
            kvec_1.push_row(kvec.row(1));
            kvec_1.push_row(kvec.row(2));
            kvec_1.push_row(kvec_m.view());
            
            let mut kvec_2=Array2::<f64>::zeros((0,3));
            kvec_2.push_row(kvec.row(0));
            kvec_2.push_row(kvec.row(1));
            kvec_2.push_row(kvec_m.view());
            kvec_2.push_row(kvec.row(3));
            let mut kvec_3=Array2::<f64>::zeros((0,3));
            kvec_3.push_row(kvec.row(0));
            kvec_3.push_row(kvec_m.view());
            kvec_3.push_row(kvec.row(2));
            kvec_3.push_row(kvec.row(3));
            let mut kvec_4=Array2::<f64>::zeros((0,3));
            kvec_4.push_row(kvec_m.view());
            kvec_4.push_row(kvec.row(1));
            kvec_4.push_row(kvec.row(2));
            kvec_4.push_row(kvec.row(3));
            let all_1=cal_integrate_3D(f0,&kvec_1);
            let all_2=cal_integrate_3D(f0,&kvec_2);
            let all_3=cal_integrate_3D(f0,&kvec_3);
            let all_4=cal_integrate_3D(f0,&kvec_4);
            let mut all_new=all_1+all_2+all_3+all_4;
            let abs_err= if ab_err>all*re_err{ab_err} else {re_err};
            if (all_new-all).abs()> abs_err && S > 1e-9{
                let all_1=adapt_integrate_tetrahedron(f0,&kvec_1,re_err/4.0,ab_err/4.0);
                let all_2=adapt_integrate_tetrahedron(f0,&kvec_2,re_err/4.0,ab_err/4.0);
                let all_3=adapt_integrate_tetrahedron(f0,&kvec_3,re_err/4.0,ab_err/4.0);
                let all_4=adapt_integrate_tetrahedron(f0,&kvec_4,re_err/4.0,ab_err/4.0);
                all_new=all_1+all_2+all_3+all_4;
            }
            all_new
        }
        let area_1:Array2::<f64>=arr2(&[[k_range.row(0)[0],k_range.row(1)[0],k_range.row(2)[0]],
                                        [k_range.row(0)[1],k_range.row(1)[0],k_range.row(2)[0]],
                                        [k_range.row(0)[0],k_range.row(1)[1],k_range.row(2)[0]],
                                        [k_range.row(0)[0],k_range.row(1)[0],k_range.row(2)[1]]]);//第一个四面体
        let area_2:Array2::<f64>=arr2(&[[k_range.row(0)[1],k_range.row(1)[1],k_range.row(2)[0]],
                                        [k_range.row(0)[1],k_range.row(1)[0],k_range.row(2)[0]],
                                        [k_range.row(0)[0],k_range.row(1)[1],k_range.row(2)[0]],
                                        [k_range.row(0)[1],k_range.row(1)[1],k_range.row(2)[1]]]);//第二个四面体
        let area_3:Array2::<f64>=arr2(&[[k_range.row(0)[1],k_range.row(1)[0],k_range.row(2)[1]],
                                        [k_range.row(0)[1],k_range.row(1)[0],k_range.row(2)[0]],
                                        [k_range.row(0)[0],k_range.row(1)[0],k_range.row(2)[1]],
                                        [k_range.row(0)[1],k_range.row(1)[1],k_range.row(2)[1]]]);//第三个四面体
        let area_4:Array2::<f64>=arr2(&[[k_range.row(0)[0],k_range.row(1)[1],k_range.row(2)[1]],
                                        [k_range.row(0)[0],k_range.row(1)[0],k_range.row(2)[1]],
                                        [k_range.row(0)[0],k_range.row(1)[1],k_range.row(2)[0]],
                                        [k_range.row(0)[1],k_range.row(1)[1],k_range.row(2)[1]]]);//第一个四面体
        let area_5:Array2::<f64>=arr2(&[[k_range.row(0)[0],k_range.row(1)[0],k_range.row(2)[1]],
                                        [k_range.row(0)[1],k_range.row(1)[1],k_range.row(2)[1]],
                                        [k_range.row(0)[0],k_range.row(1)[1],k_range.row(2)[0]],
                                        [k_range.row(0)[1],k_range.row(1)[0],k_range.row(2)[0]]]);//第一个四面体
        let all_1=adapt_integrate_tetrahedron(f0,&area_1,re_err/5.0,ab_err/5.0);
        let all_2=adapt_integrate_tetrahedron(f0,&area_2,re_err/5.0,ab_err/5.0);
        let all_3=adapt_integrate_tetrahedron(f0,&area_3,re_err/5.0,ab_err/5.0);
        let all_4=adapt_integrate_tetrahedron(f0,&area_4,re_err/5.0,ab_err/5.0);
        let all_5=adapt_integrate_tetrahedron(f0,&area_5,re_err/5.0,ab_err/5.0);
        return all_1+all_2+all_3+all_4+all_5
    }else{
        panic!("wrong, the row_dim if k_range must be 1,2 or 3, but you's give {}",dim);
    }
}

pub fn adapted_integrate_loop(f0:&dyn Fn(&Array1::<f64>)->f64,k_range:&Array2::<f64>,re_err:f64,ab_err:f64)->f64{
    ///对于任意维度的积分 n, 我们的将区域刨分成 n+1面体的小块, 然后用线性插值来近似这个n+1的积分结果
    ///
    ///积分的公式为: 
    ///
    ///设被积函数为 f(x1,x2,...,xn), 存在n+1个点 (y01,y02,...y0n)...(yn1,yn2... ynn), 对应的值为 z0,z1,...,zn
    ///
    ///这样我们就能得到 1/(n+1)! *\sum_{i}^n z_i *dV, dV 是正 n+1面体的体积.

    let dim=k_range.len_of(Axis(0));
    if dim==1{
        //对于一维情况, 我们就是用梯形算法的 (a+b)*h/2, 这里假设的是函数的插值为线性插值.
        let mut use_range=vec![(k_range.clone(),re_err,ab_err)];
        let mut result=0.0;
        while let Some((k_range,re_err,ab_err))=use_range.pop() {
            let kvec_l:Array1::<f64>=arr1(&[k_range[[0,0]]]);
            let kvec_r:Array1::<f64>=arr1(&[k_range[[0,1]]]);
            let kvec_m:Array1::<f64>=arr1(&[(k_range[[0,1]]+k_range[[0,0]])/2.0]);
            let dk:f64=k_range[[0,1]]-k_range[[0,0]];
            let y_l:f64=f0(&kvec_l);
            let y_r:f64=f0(&kvec_r);
            let y_m:f64=f0(&kvec_m);
            let all:f64=(y_l+y_r)*dk/2.0;
            let all_1=(y_l+y_m)*dk/4.0;
            let all_2=(y_r+y_m)*dk/4.0;
            let err=all_1+all_2-all;
            let abs_err= if ab_err>all*re_err{ab_err} else {re_err};
            if err< abs_err{
                result+=all_1+all_2;
            }else{
                let k_range_l=arr2(&[[kvec_l[[0]],kvec_m[[0]]]]);
                let k_range_r=arr2(&[[kvec_m[[0]],kvec_r[[0]]]]);
                use_range.push((k_range_l.clone(),re_err/2.0,ab_err/2.0));
                use_range.push((k_range_r.clone(),re_err/2.0,ab_err/2.0));
            }
        }
        return result;
    }else if dim==2{
    //对于二维, 我们依旧假设线性插值, 这样我们考虑的就是二维平面上的三角形上的任意一点的值是到其余三个点的距离的加权系数的平均值, 我们将四边形变成两个三角形来考虑.
        let area_1:Array2::<f64>=arr2(&[[k_range.row(0)[0],k_range.row(1)[0]],[k_range.row(0)[1],k_range.row(1)[0]],[k_range.row(0)[0],k_range.row(1)[1]]]);//第一个三角形
        let area_2:Array2::<f64>=arr2(&[[k_range.row(0)[1],k_range.row(1)[1]],[k_range.row(0)[1],k_range.row(1)[0]],[k_range.row(0)[0],k_range.row(1)[1]]]);//第二个三角形
        fn cal_integrate_2D(f0:&dyn Fn(&Array1::<f64>)->f64,kvec:&Array2::<f64>)->f64{
            //这个是用来进行线性插值积分的结果, 给出三个点和函数, 计算得到对应的插值积分结果
            let mut S:Array2::<f64>=kvec.clone();
            S.push(Axis(1),Array1::ones(3).view());
            let S:f64=S.det().expect("Wrong, S'det is 0").abs();
            let mut all:f64=0.0;
            for i in 0..kvec.len_of(Axis(0)){
                all+=f0(&kvec.row(i).to_owned())
            }
            all*=S;
            all=all/6.0;
            all
        }
        fn adapt_integrate_triangle(f0:&dyn Fn(&Array1::<f64>)->f64,kvec:&Array2::<f64>,re_err:f64,ab_err:f64)->f64{
            //这个函数是用来进行自适应算法的
            let mut use_kvec=vec![(kvec.clone(),re_err,ab_err)];
            let mut result=0.0;
            while let Some((kvec,re_err,ab_err))=use_kvec.pop() {
                let mut S=kvec.clone();
                S.push(Axis(1),Array1::ones(3).view());
                let S:f64=S.det().expect("Wrong, S'det is 0").abs();//先求一下体积


                let all=cal_integrate_2D(f0,&kvec);
                let kvec_m=kvec.mean_axis(Axis(0)).unwrap();
                let mut kvec_1=Array2::<f64>::zeros((0,2));
                kvec_1.push_row(kvec.row(0));
                kvec_1.push_row(kvec.row(1));
                kvec_1.push_row(kvec_m.view());
                let mut kvec_2=Array2::<f64>::zeros((0,2));
                kvec_2.push_row(kvec.row(0));
                kvec_2.push_row(kvec_m.view());
                kvec_2.push_row(kvec.row(2));
                let mut kvec_3=Array2::<f64>::zeros((0,2));
                kvec_3.push_row(kvec_m.view());
                kvec_3.push_row(kvec.row(1));
                kvec_3.push_row(kvec.row(2));
                let all_1=cal_integrate_2D(f0,&kvec_1);
                let all_2=cal_integrate_2D(f0,&kvec_2);
                let all_3=cal_integrate_2D(f0,&kvec_3);
                let all_new=all_1+all_2+all_3;
                let abs_err:f64= if ab_err>all*re_err{ab_err} else {re_err};
                /*
                println!("kvec={}",kvec);
                println!("all={}",all);
                println!("all_1={}",all_1);
                println!("all_2={}",all_2);
                println!("all_3={}",all_3);
                println!("all_new={}",all_new);
                println!("err={}",abs_err);
                println!("real_err={}",(all_new-all));
                */
                if (all_new-all).abs()> abs_err && S>1e-8{
                   //let all_1=adapt_integrate_triangle(f0,&kvec_1,re_err/3.0,ab_err/3.0);
                   //let all_2=adapt_integrate_triangle(f0,&kvec_2,re_err/3.0,ab_err/3.0);
                   //let all_3=adapt_integrate_triangle(f0,&kvec_3,re_err/3.0,ab_err/3.0);
                   //all_new=all_1+all_2+all_3;
                   //return all_new;
                   use_kvec.push((kvec_1.clone(),re_err/3.0,ab_err/3.0));
                   use_kvec.push((kvec_2.clone(),re_err/3.0,ab_err/3.0));
                   use_kvec.push((kvec_3.clone(),re_err/3.0,ab_err/3.0));
                }else{
                    result+=all_new; 
                }
            }
            result
        }
        let all_1=adapt_integrate_triangle(f0,&area_1,re_err/2.0,ab_err/2.0);
        let all_2=adapt_integrate_triangle(f0,&area_2,re_err/2.0,ab_err/2.0);
        return all_1+all_2;
    }else if dim==3{
    //对于三位情况, 需要用到四面体, 所以需要先将6面体变成6个四面体
        fn cal_integrate_3D(f0:&dyn Fn(&Array1::<f64>)->f64,kvec:&Array2::<f64>)->f64{
            //这个是用来进行线性插值积分的结果, 给出三个点和函数, 计算得到对应的插值积分结果
            let mut S=kvec.clone();
            S.push(Axis(1),Array1::ones(4).view());
            let S:f64=S.det().expect("Wrong, S'det is 0").abs();
            let mut all:f64=0.0;
            for i in 0..kvec.len_of(Axis(0)){
                all+=f0(&kvec.row(i).to_owned())
            }
            all*=S;
            all=all/24.0;
            all
        }
        fn adapt_integrate_tetrahedron(f0:&dyn Fn(&Array1::<f64>)->f64,kvec:&Array2::<f64>,re_err:f64,ab_err:f64)->f64{
            //这个函数是用来进行自适应算法的
            let mut use_kvec=vec![(kvec.clone(),re_err,ab_err)];
            let mut result=0.0;
            while let Some((kvec,re_err,ab_err))=use_kvec.pop() {
                let mut S=kvec.clone();
                S.push(Axis(1),Array1::ones(4).view());
                let S:f64=S.det().expect("Wrong, S'det is 0").abs();//先求一下体积

                let all=cal_integrate_3D(f0,&kvec);
                let kvec_m=kvec.mean_axis(Axis(0)).unwrap();
                /////////////////////////
                let mut kvec_1=Array2::<f64>::zeros((0,3));
                kvec_1.push_row(kvec.row(0));
                kvec_1.push_row(kvec.row(1));
                kvec_1.push_row(kvec.row(2));
                kvec_1.push_row(kvec_m.view());
                
                let mut kvec_2=Array2::<f64>::zeros((0,3));
                kvec_2.push_row(kvec.row(0));
                kvec_2.push_row(kvec.row(1));
                kvec_2.push_row(kvec_m.view());
                kvec_2.push_row(kvec.row(3));
                let mut kvec_3=Array2::<f64>::zeros((0,3));
                kvec_3.push_row(kvec.row(0));
                kvec_3.push_row(kvec_m.view());
                kvec_3.push_row(kvec.row(2));
                kvec_3.push_row(kvec.row(3));
                let mut kvec_4=Array2::<f64>::zeros((0,3));
                kvec_4.push_row(kvec_m.view());
                kvec_4.push_row(kvec.row(1));
                kvec_4.push_row(kvec.row(2));
                kvec_4.push_row(kvec.row(3));
                let all_1=cal_integrate_3D(f0,&kvec_1);
                let all_2=cal_integrate_3D(f0,&kvec_2);
                let all_3=cal_integrate_3D(f0,&kvec_3);
                let all_4=cal_integrate_3D(f0,&kvec_4);
                let all_new=all_1+all_2+all_3+all_4;
                let abs_err= if ab_err>all*re_err{ab_err} else {re_err};
                if (all_new-all).abs()> abs_err && S > 1e-9{
                    use_kvec.push((kvec_1.clone(),re_err*0.25,ab_err*0.25));
                    use_kvec.push((kvec_2.clone(),re_err*0.25,ab_err*0.25));
                    use_kvec.push((kvec_3.clone(),re_err*0.25,ab_err*0.25));
                    use_kvec.push((kvec_4.clone(),re_err*0.25,ab_err*0.25));
                }else{
                    result+=all_new;
                }
            }
            result
        }
        let area_1:Array2::<f64>=arr2(&[[k_range.row(0)[0],k_range.row(1)[0],k_range.row(2)[0]],
                                        [k_range.row(0)[1],k_range.row(1)[0],k_range.row(2)[0]],
                                        [k_range.row(0)[0],k_range.row(1)[1],k_range.row(2)[0]],
                                        [k_range.row(0)[0],k_range.row(1)[0],k_range.row(2)[1]]]);//第一个四面体
        let area_2:Array2::<f64>=arr2(&[[k_range.row(0)[1],k_range.row(1)[1],k_range.row(2)[0]],
                                        [k_range.row(0)[1],k_range.row(1)[0],k_range.row(2)[0]],
                                        [k_range.row(0)[0],k_range.row(1)[1],k_range.row(2)[0]],
                                        [k_range.row(0)[1],k_range.row(1)[1],k_range.row(2)[1]]]);//第二个四面体
        let area_3:Array2::<f64>=arr2(&[[k_range.row(0)[1],k_range.row(1)[0],k_range.row(2)[1]],
                                        [k_range.row(0)[1],k_range.row(1)[0],k_range.row(2)[0]],
                                        [k_range.row(0)[0],k_range.row(1)[0],k_range.row(2)[1]],
                                        [k_range.row(0)[1],k_range.row(1)[1],k_range.row(2)[1]]]);//第三个四面体
        let area_4:Array2::<f64>=arr2(&[[k_range.row(0)[0],k_range.row(1)[1],k_range.row(2)[1]],
                                        [k_range.row(0)[0],k_range.row(1)[0],k_range.row(2)[1]],
                                        [k_range.row(0)[0],k_range.row(1)[1],k_range.row(2)[0]],
                                        [k_range.row(0)[1],k_range.row(1)[1],k_range.row(2)[1]]]);//第一个四面体
        let area_5:Array2::<f64>=arr2(&[[k_range.row(0)[0],k_range.row(1)[0],k_range.row(2)[1]],
                                        [k_range.row(0)[1],k_range.row(1)[1],k_range.row(2)[1]],
                                        [k_range.row(0)[0],k_range.row(1)[1],k_range.row(2)[0]],
                                        [k_range.row(0)[1],k_range.row(1)[0],k_range.row(2)[0]]]);//第一个四面体
        let all_1=adapt_integrate_tetrahedron(f0,&area_1,re_err/5.0,ab_err/5.0);
        let all_2=adapt_integrate_tetrahedron(f0,&area_2,re_err/5.0,ab_err/5.0);
        let all_3=adapt_integrate_tetrahedron(f0,&area_3,re_err/5.0,ab_err/5.0);
        let all_4=adapt_integrate_tetrahedron(f0,&area_4,re_err/5.0,ab_err/5.0);
        let all_5=adapt_integrate_tetrahedron(f0,&area_5,re_err/5.0,ab_err/5.0);
        return all_1+all_2+all_3+all_4+all_5
    }else{
        panic!("wrong, the row_dim if k_range must be 1,2 or 3, but you's give {}",dim);
    }
}
pub fn adapted_integrate_quick(f0:&dyn Fn(&Array1::<f64>)->f64,k_range:&Array2::<f64>,re_err:f64,ab_err:f64)->f64{
    ///对于任意维度的积分 n, 我们的将区域刨分成 n+1面体的小块, 然后用线性插值来近似这个n+1的积分结果
    ///
    ///积分的公式为: 
    ///
    ///设被积函数为 f(x1,x2,...,xn), 存在n+1个点 (y01,y02,...y0n)...(yn1,yn2... ynn), 对应的值为 z0,z1,...,zn
    ///
    ///这样我们就能得到 1/(n+1)! *\sum_{i}^n z_i *dV, dV 是正 n+1面体的体积.

    let dim=k_range.len_of(Axis(0));
    if dim==1{
        //对于一维情况, 我们就是用梯形算法的 (a+b)*h/2, 这里假设的是函数的插值为线性插值.
        let mut use_range=vec![(k_range.clone(),re_err,ab_err)];
        let mut result=0.0;
        while let Some((k_range,re_err,ab_err))=use_range.pop() {
            let kvec_l:Array1::<f64>=arr1(&[k_range[[0,0]]]);
            let kvec_r:Array1::<f64>=arr1(&[k_range[[0,1]]]);
            let kvec_m:Array1::<f64>=arr1(&[(k_range[[0,1]]+k_range[[0,0]])/2.0]);
            let dk:f64=k_range[[0,1]]-k_range[[0,0]];
            let y_l:f64=f0(&kvec_l);
            let y_r:f64=f0(&kvec_r);
            let y_m:f64=f0(&kvec_m);
            let all:f64=(y_l+y_r)*dk/2.0;
            let all_1=(y_l+y_m)*dk/4.0;
            let all_2=(y_r+y_m)*dk/4.0;
            let err=all_1+all_2-all;
            let abs_err= if ab_err>all*re_err{ab_err} else {re_err};
            if err< abs_err{
                result+=all_1+all_2;
            }else{
                let k_range_l=arr2(&[[kvec_l[[0]],kvec_m[[0]]]]);
                let k_range_r=arr2(&[[kvec_m[[0]],kvec_r[[0]]]]);
                use_range.push((k_range_l.clone(),re_err/2.0,ab_err/2.0));
                use_range.push((k_range_r.clone(),re_err/2.0,ab_err/2.0));
            }
        }
        return result;
    }else if dim==2{
    //对于二维, 我们依旧假设线性插值, 这样我们考虑的就是二维平面上的三角形上的任意一点的值是到其余三个点的距离的加权系数的平均值, 我们将四边形变成两个三角形来考虑.
        let area_1:Array2::<f64>=arr2(&[[k_range.row(0)[0],k_range.row(1)[0]],[k_range.row(0)[1],k_range.row(1)[0]],[k_range.row(0)[0],k_range.row(1)[1]]]);//第一个三角形
        let area_2:Array2::<f64>=arr2(&[[k_range.row(0)[1],k_range.row(1)[1]],[k_range.row(0)[1],k_range.row(1)[0]],[k_range.row(0)[0],k_range.row(1)[1]]]);//第二个三角形
        fn adapt_integrate_triangle(f0:&dyn Fn(&Array1::<f64>)->f64,kvec:&Array2::<f64>,re_err:f64,ab_err:f64)->f64{
            //这个函数是用来进行自适应算法的
            let mut result=0.0;
            let s1=f0(&kvec.row(0).to_owned());
            let s2=f0(&kvec.row(1).to_owned());
            let s3=f0(&kvec.row(2).to_owned());
            let mut use_kvec=vec![(kvec.clone(),re_err,ab_err,s1,s2,s3)];
            while let Some((kvec,re_err,ab_err,s1,s2,s3))=use_kvec.pop() {
                //let mut S=kvec.clone();
                //S.push(Axis(1),Array1::ones(3).view());
                //let S:f64=S.det().expect("Wrong, S'det is 0").abs();//先求一下体积
                let S:f64=((kvec[[1,0]]*kvec[[2,1]]-kvec[[2,0]]*kvec[[1,1]])-(kvec[[0,0]]*kvec[[2,1]]-kvec[[0,1]]*kvec[[2,0]])+(kvec[[0,0]]*kvec[[1,1]]-kvec[[1,0]]*kvec[[0,1]])).abs();
                let kvec_m=kvec.mean_axis(Axis(0)).unwrap();
                let sm:f64=f0(&kvec_m.to_owned());

                let mut kvec_1=Array2::<f64>::zeros((0,2));
                kvec_1.push_row(kvec.row(0));
                kvec_1.push_row(kvec.row(1));
                kvec_1.push_row(kvec_m.view());

                let mut kvec_2=Array2::<f64>::zeros((0,2));
                kvec_2.push_row(kvec.row(0));
                kvec_2.push_row(kvec_m.view());
                kvec_2.push_row(kvec.row(2));

                let mut kvec_3=Array2::<f64>::zeros((0,2));
                kvec_3.push_row(kvec_m.view());
                kvec_3.push_row(kvec.row(1));
                kvec_3.push_row(kvec.row(2));

                let all:f64=(s1+s2+s3)*S/6.0;
                let all_new:f64=all/3.0*2.0+sm*S/6.0;
                let abs_err:f64= if ab_err>all*re_err{ab_err} else {re_err};
                if (all_new-all).abs() > abs_err && S>1e-8{
                   use_kvec.push((kvec_1.clone(),re_err/3.0,ab_err/3.0,s1,s2,sm));
                   use_kvec.push((kvec_2.clone(),re_err/3.0,ab_err/3.0,s1,sm,s3));
                   use_kvec.push((kvec_3.clone(),re_err/3.0,ab_err/3.0,sm,s2,s3));
                }else{
                    result+=all_new; 
                }
            }
            result
        }
        let all_1=adapt_integrate_triangle(f0,&area_1,re_err/2.0,ab_err/2.0);
        let all_2=adapt_integrate_triangle(f0,&area_2,re_err/2.0,ab_err/2.0);
        return all_1+all_2;
    }else if dim==3{
    //对于三位情况, 需要用到四面体, 所以需要先将6面体变成6个四面体
        fn adapt_integrate_tetrahedron(f0:&dyn Fn(&Array1::<f64>)->f64,kvec:&Array2::<f64>,re_err:f64,ab_err:f64,S:f64)->f64{
        /*
            fn det3(a:&Array1::<f64>,b:&Array1::<f64>,c:&Array1::<f64>)->f64{
                return -a[[2]]*b[[1]]*c[[0]]+a[[1]]*b[[2]]*c[[0]]+a[[2]]*b[[0]]*c[[1]]-a[[0]]*b[[2]]*c[[1]]-a[[1]]*b[[0]]*c[[2]]+a[[0]]*b[[1]]*c[[2]];
            }
            */
            //这个函数是用来进行自适应算法的
            let mut result=0.0;
            let s1=f0(&kvec.row(0).to_owned());
            let s2=f0(&kvec.row(1).to_owned());
            let s3=f0(&kvec.row(2).to_owned());
            let s4=f0(&kvec.row(3).to_owned());
            let mut use_kvec=vec![(kvec.clone(),re_err,ab_err,s1,s2,s3,s4,S)];
            while let Some((kvec,re_err,ab_err,s1,s2,s3,s4,S))=use_kvec.pop() {
                let kvec_m=kvec.mean_axis(Axis(0)).unwrap();
                let sm=f0(&kvec_m.to_owned());
                /////////////////////////
                let mut kvec_1=Array2::<f64>::zeros((0,3));
                kvec_1.push_row(kvec.row(0));
                kvec_1.push_row(kvec.row(1));
                kvec_1.push_row(kvec.row(2));
                kvec_1.push_row(kvec_m.view());
                
                let mut kvec_2=Array2::<f64>::zeros((0,3));
                kvec_2.push_row(kvec.row(0));
                kvec_2.push_row(kvec.row(1));
                kvec_2.push_row(kvec_m.view());
                kvec_2.push_row(kvec.row(3));

                let mut kvec_3=Array2::<f64>::zeros((0,3));
                kvec_3.push_row(kvec.row(0));
                kvec_3.push_row(kvec_m.view());
                kvec_3.push_row(kvec.row(2));
                kvec_3.push_row(kvec.row(3));

                let mut kvec_4=Array2::<f64>::zeros((0,3));
                kvec_4.push_row(kvec_m.view());
                kvec_4.push_row(kvec.row(1));
                kvec_4.push_row(kvec.row(2));
                kvec_4.push_row(kvec.row(3));

                let all=(s1+s2+s3+s4)*S/24.0;
                let all_new=all/4.0*3.0+sm*S/24.0;
                let abs_err= if ab_err>all*re_err{ab_err} else {re_err};
                let S0=S/4.0;
                if (all_new-all).abs()> abs_err && S > 1e-9{
                    use_kvec.push((kvec_1.clone(),re_err*0.25,ab_err*0.25,s1,s2,s3,sm,S0));
                    use_kvec.push((kvec_2.clone(),re_err*0.25,ab_err*0.25,s1,s2,sm,s4,S0));
                    use_kvec.push((kvec_3.clone(),re_err*0.25,ab_err*0.25,s1,sm,s3,s4,S0));
                    use_kvec.push((kvec_4.clone(),re_err*0.25,ab_err*0.25,sm,s2,s3,s4,S0));
                }else{
                    result+=all_new;
                }
            }
            result
        }
        let area_1:Array2::<f64>=arr2(&[[k_range.row(0)[0],k_range.row(1)[0],k_range.row(2)[0]],
                                        [k_range.row(0)[1],k_range.row(1)[0],k_range.row(2)[0]],
                                        [k_range.row(0)[0],k_range.row(1)[1],k_range.row(2)[0]],
                                        [k_range.row(0)[0],k_range.row(1)[0],k_range.row(2)[1]]]);//第一个四面体
        let area_2:Array2::<f64>=arr2(&[[k_range.row(0)[1],k_range.row(1)[1],k_range.row(2)[0]],
                                        [k_range.row(0)[1],k_range.row(1)[0],k_range.row(2)[0]],
                                        [k_range.row(0)[0],k_range.row(1)[1],k_range.row(2)[0]],
                                        [k_range.row(0)[1],k_range.row(1)[1],k_range.row(2)[1]]]);//第二个四面体
        let area_3:Array2::<f64>=arr2(&[[k_range.row(0)[1],k_range.row(1)[0],k_range.row(2)[1]],
                                        [k_range.row(0)[1],k_range.row(1)[0],k_range.row(2)[0]],
                                        [k_range.row(0)[0],k_range.row(1)[0],k_range.row(2)[1]],
                                        [k_range.row(0)[1],k_range.row(1)[1],k_range.row(2)[1]]]);//第三个四面体
        let area_4:Array2::<f64>=arr2(&[[k_range.row(0)[0],k_range.row(1)[1],k_range.row(2)[1]],
                                        [k_range.row(0)[0],k_range.row(1)[0],k_range.row(2)[1]],
                                        [k_range.row(0)[0],k_range.row(1)[1],k_range.row(2)[0]],
                                        [k_range.row(0)[1],k_range.row(1)[1],k_range.row(2)[1]]]);//第一个四面体
        let area_5:Array2::<f64>=arr2(&[[k_range.row(0)[0],k_range.row(1)[0],k_range.row(2)[1]],
                                        [k_range.row(0)[1],k_range.row(1)[1],k_range.row(2)[1]],
                                        [k_range.row(0)[0],k_range.row(1)[1],k_range.row(2)[0]],
                                        [k_range.row(0)[1],k_range.row(1)[0],k_range.row(2)[0]]]);//第一个四面体
        let V=(k_range[[0,1]]-k_range[[0,0]])*(k_range[[1,1]]-k_range[[1,0]])*(k_range[[2,1]]-k_range[[2,0]]);
        let all_1=adapt_integrate_tetrahedron(f0,&area_1,re_err,ab_err/6.0,V);
        let all_2=adapt_integrate_tetrahedron(f0,&area_2,re_err,ab_err/6.0,V);
        let all_3=adapt_integrate_tetrahedron(f0,&area_3,re_err,ab_err/6.0,V);
        let all_4=adapt_integrate_tetrahedron(f0,&area_4,re_err,ab_err/6.0,V);
        let all_5=adapt_integrate_tetrahedron(f0,&area_5,re_err,ab_err/3.0,V*2.0);
        return all_1+all_2+all_3+all_4+all_5
    }else{
        panic!("wrong, the row_dim if k_range must be 1,2 or 3, but you's give {}",dim);
    }
}

fn test_func(k:&Array1::<f64>)->f64{
    let k0=k.clone();
    //k0.dot(&k0)//.powi(5).sqrt().sin()
    k0.sum().powi(2)
    //k0.sum()
}
