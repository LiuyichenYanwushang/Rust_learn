use ndarray::{Array2, Array1, arr1};
use ndarray_linalg::*;
use std::time::{Duration, Instant};
use nalgebra::Complex;

// 定义常见的公差取值
const DEFAULT_TOLERANCE: f64 = 1e-10;

// 定义共轭梯度法结构体
struct ConjugateGradient {
    max_iterations: usize,
    tolerance: f64
}

impl ConjugateGradient {
    fn new(max_iterations: usize, tolerance: f64) -> ConjugateGradient {
        ConjugateGradient{ max_iterations, tolerance }
    }

    // 定义共轭梯度法迭代求解函数，带入系数矩阵 A 和右侧向量 b
    fn solve(&self, a: &Array2::<Complex<f64>>, b: &Array1::<Complex<f64>>) -> (Array1<Complex<f64>>, usize) {
        let n = a.shape()[0];
        let mut x = Array1::zeros(n);
        let mut r = b - &a.dot(&x);          // 初始残差
        let mut p = r.clone();              // 初始搜索方向
        let mut iterations = 0;

        while iterations < self.max_iterations && r.norm() > self.tolerance { // 迭代条件
            let ap = &a.dot(&p);
            let alpha = r.dot(&r) / p.dot(ap);
            x += &(alpha * &p);              // 更新迭代估计值
            r -= &(alpha * ap);             // 计算残差
            let beta = r.dot(&r) / p.dot(ap);
            p = r.clone() + &(beta * &p);    // 更新搜索方向
            iterations += 1;
        }

        (x, iterations)
    }
}

fn main() {
    // 生成一个随机的对称正定矩阵，用于本例演示
    const n:usize = 5000;
    let mut a = Array2::<Complex<f64>>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let val = (i - j) as f64/(1000000 as f64);
            a[[i, j]] = Complex::new(val,val);
            a[[j, i]] = Complex::new(val,-val);
        }
    }

    // 求解本征值
    let cg = ConjugateGradient::new(10000, DEFAULT_TOLERANCE);

    let start = Instant::now();   // 开始计时
    let r=Array1::<Complex<f64>>::ones(n);
    let (eigenvalues, num_iterations) = cg.solve(&a, &r);
    let end = Instant::now();    // 结束计时
    let duration = end.duration_since(start); // 计算执行时间
    println!("function_a took {} seconds", duration.as_secs_f64());   // 输出执行时间
    // 输出本征值及迭代次数
    println!("Number of iterations: {}", num_iterations);
    let start = Instant::now();   // 开始计时
    let eigvals=a.eigvals().unwrap();
    let end = Instant::now();    // 结束计时
    let duration = end.duration_since(start); // 计算执行时间
    println!("function_a took {} seconds", duration.as_secs_f64());   // 输出执行时间
}
