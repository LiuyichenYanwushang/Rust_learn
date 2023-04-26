use ndarray::*;
use ndarray::linalg::*;
use ndarray_linalg::*;
use num_complex::Complex;
use nalgebra_sparse::{coo::CooMatrix, csc::CscMatrix, csr::CsrMatrix};
use num_traits::identities::Zero;
use num_traits::One;

// Lanczos algorithm for symmetric matrices
// https://en.wikipedia.org/wiki/Lanczos_algorithm
fn lanczos(A: &Array2::<Complex<f64>>, k: usize) -> (Array1::<Complex<f64>>, Array1::<Complex<f64>>) {
    // Initialize variables
    let n = A.shape()[0];
    //let mut alpha = Vec::with_capacity(k);
    let mut alpha=Array1::<Complex<f64>>::zeros(k);
    let mut beta:Vec<Complex<f64>> = Vec::with_capacity(k);
    let mut q = Vec::with_capacity(k + 1);
    let mut v = Array1::from_elem(n, Complex::<f64>::new(0.0,0.0));

    // Choose an arbitrary vector q[0] with ||q[0]|| = 1 and v = 0
    //q.push(Array1::from_elem(n, 1.0 / (n as f64).sqrt()));
    //let mut first_q=Array1::<f64>::zeros(n);
    //first_q[[0]]=1.0;
    //
    let mut first_q=Array1::<Complex<f64>>::ones(n)/(n as f64).sqrt();
    q.push(first_q);


    for i in 0..k {
        // v = A * q[i] - beta[i-1] * q[i-1]
        v.assign(&A.dot(&q[i]));
        if i > 0 {
            v.scaled_add(-beta[i - 1], &q[i - 1]);
        }

        // alpha[i] = q[i]^T * v
        alpha[i]=q[i].map(|x| x.conj()).dot(&v);

        // v = v - alpha[i] * q[i]
        v.scaled_add(-alpha[i], &q[i]);

        // beta[i] = ||v||
        beta.push(Complex::new(v.norm(),0.0));

        // Orthogonalize v against the previous vectors
        for j in 0..=i {
            let r = q[j].dot(&v);
            v.scaled_add(-r, &q[j]);
        }
        let b = v.norm();
        if b > 0.0 {
            // q[i+1] = v / b
            q.push(v.mapv(|x| x / b));
        } else {
            // q[i+1] = some arbitrary vector orthogonal to q[0], ..., q[i]
            let mut w = Array1::zeros(n);
            w[0] = -q[0][n - 1];
            w[n - 1] = q[0][0];
            for j in 1..n - 1 {
                w[j] = q[0][j + 1] - q[0][j - 1];
            }
            let c = w.norm();
            q.push(w.mapv(|x| x / c));
        }
    }
    beta.pop();
    let  beta=Array1::<Complex<f64>>::from_vec(beta);
    (alpha,beta)
}
pub fn trig_solve(
    diag: &Array1<Complex<f64>>,
    upper: &Array1<Complex<f64>>)->(Array1<Complex<f64>>, Array2<Complex<f64>>){
    

    

}
pub fn thomas_algorithm_eigenvalues(
    diag: &Array1<Complex<f64>>,
    upper: &Array1<Complex<f64>>,
    lower: &Array1<Complex<f64>>,
) -> (Array1<Complex<f64>>, Array2<Complex<f64>>) {
    let n = diag.len();
    let mut eigenvalues = Array1::zeros(n);
    let mut eigenvectors = Array2::zeros((n, n));

    // 初始化第一个本征矢量为(1, 0, ..., 0)
    eigenvectors[[0, 0]] = Complex::one();

    // 迭代求解本征值和本征矢量
    for i in 0..n - 1 {
        let alpha:Complex<f64> = diag[[i]];
        let beta:Complex<f64> = upper[[i]];
        let gamma:Complex<f64> = lower[[i]];

        // 求解对角化矩阵Q
        let q:Complex<f64>= if gamma.is_zero() {
            Complex::<f64>::new(1.0,0.0)
        } else {
            alpha - eigenvalues[[i]]
        };
        let q:Complex<f64>=q.clone()/(alpha - eigenvalues[[i]] - beta * eigenvalues[[i + 1]] / gamma);

        // 更新本征值和本征矢量
        let diff_eig=eigenvalues[[i]] - eigenvalues[[i + 1]];
        eigenvalues[[i]] += q * diff_eig;
        eigenvalues[[i + 1]] += -q * diff_eig;

        let old_evec=eigenvectors.row(i + 1).to_owned();
        let old_evec_1=eigenvectors.row(i).to_owned();
        eigenvectors.row_mut(i).zip_mut_with(&old_evec, |v1, v2| {
            *v1 = q * (*v1 - v2.conj() * beta / gamma);
        });
        eigenvectors.row_mut(i + 1)
            .zip_mut_with(&old_evec_1, |v1, v2| {
                *v1 = q * (*v1 - v2 * gamma / beta);
            });
    }

    (eigenvalues, eigenvectors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};
    use approx::AbsDiffEq;
    #[test]
    fn test_lanczos() {
        // Example from https://en.wikipedia.org/wiki/Lanczos_algorithm#Example
        let A = arr2(&[
            [4.0, 1.0, -2.0, 2.0],
            [1.0, 2.0, 0.0, 1.0],
            [-2.0, 0.0, 3.0, -2.0],
            [2.0, 1.0, -2.0, -1.0],
        ]);
        let A=A.map(|x| Complex::new(*x,0.0));
        let k = 4;
        let (lambda, x) = lanczos(&A, k);
        let diag=lambda;
        let upper=x;
        let lower=upper.clone().map(|x| x.conj());
        println!("{},{},{}",diag,upper,lower);
        let (eval,evec)=thomas_algorithm_eigenvalues(&diag, &upper, &lower);
        println!("{}",eval);
    }
}
