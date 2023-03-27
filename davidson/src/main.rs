use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::Norm;
use num_complex::Complex;
use ndarray_linalg::{Eigh, UPLO};
use ndarray_linalg::conjugate;
use ndarray_linalg::*;

struct ComplexMatrix {
    data: Array2<Complex<f64>>,
}

impl ComplexMatrix {
    fn new(data: Array2<Complex<f64>>) -> Self {
        Self { data }
    }

    fn dim(&self) -> usize {
        self.data.shape()[0]
    }

    fn eigenvalues(&self) -> Array1<Complex<f64>> {
        // TODO: compute eigenvalues using LAPACK or other libraries
        unimplemented!()
    }

    fn diagonalize(&mut self) -> Result<(), &'static str> {
        // TODO: diagonalize the matrix using LAPACK or other libraries
        unimplemented!()
    }

    fn apply(&self, x: &Array1<Complex<f64>>) -> Array1<Complex<f64>> {
        self.data.dot(x)
    }
}

struct DavidsonSolver {
    matrix: ComplexMatrix,
    num_eigenvectors: usize,
    num_iterations: usize,
    tolerance: f64,
}

impl DavidsonSolver {
    fn new(matrix: ComplexMatrix, num_eigenvectors: usize, num_iterations: usize, tolerance: f64) -> Self {
        Self { matrix, num_eigenvectors, num_iterations, tolerance }
    }

    fn solve(&mut self) -> Result<Array2<Complex<f64>>, &'static str> {
        let n = self.matrix.dim();
        let mut v = Array2::zeros((n, self.num_eigenvectors));
        let mut av = Array2::zeros((n, self.num_eigenvectors));
        let mut a = Array2::zeros((self.num_eigenvectors, self.num_eigenvectors));
        let mut d = Array1::zeros(self.num_eigenvectors);

        // initialize v using a random unit vector
        v.column_mut(0).assign(&Array1::from_shape_fn(n, |_| Complex::new(1.0, 0.0)));
        let v0=v.column_mut(0).to_owned();
        v.column_mut(0).assign(&v0.map(|x| x/v0.norm()));

        for i in 0..self.num_iterations {
            // compute av = A * v
            for j in 0..self.num_eigenvectors {
                av.column_mut(j).assign(&self.matrix.apply(&v.column(j).to_owned()));
            }

            // build the projection matrix P = I - V * V^T
            let p = Array2::eye(n) - v.dot(&v.map(|x| x.conj()).t());

            // compute the subspace matrix B = V^T * A * V
            a.assign(&v.t().map(|x| x.conj()).dot(&av));
            a /= Complex::new(self.num_eigenvectors as f64,0.0);

            // diagonalize the subspace matrix B
            //let eigen = a.eig();
            let eigen = if let Ok(eigvals) = a.eigvals() { eigvals } else {todo!()};
            println!("{}",a);
            println!("{}",eigen);
            if eigen.iter().any(|&e| e.im.abs() > 1e-6) {
                return Err("eigenvalues are complex");
            }
            d.assign(&eigen.map(|e| e.re));
            let mut q = a.clone() - Array2::<Complex<f64>>::eye(self.num_eigenvectors);
            for j in 0..self.num_eigenvectors {
                let q0=q.column(j).to_owned();
                let norm = q0.norm();
                if norm > 0.0 {
                    q.column_mut(j).assign(&(q0/(Complex::new(norm as f64,0.0))));
                }
            }
            //v.assign(&(v.dot(&p) + q.dot(&av)));
            v.assign(&(p.dot(&v) + av.dot(&q)));

            // check convergence
            let max_error = d.iter().zip(d.iter().skip(1)).map(|(&d1, &d2)| (d2 - d1).abs()).fold(0.0, f64::max);
            if max_error < self.tolerance {
                break;
            }
        }

        Ok(v)
    }
}

fn main() {
    let data = Array2::from_shape_fn((3, 3), |(i, j)| {
        match (i, j) {
            (0, 0) => Complex::new(1.0, 0.0),
            (0, 1) => Complex::new(2.0, -1.0),
            (1, 0) => Complex::new(2.0, 1.0),
            (1, 1) => Complex::new(3.0, 0.0),
            (2, 2) => Complex::new(4.0, 0.0),
            _ => Complex::new(0.0, 0.0),
        }
    });
    let matrix = ComplexMatrix::new(data);
    let mut solver = DavidsonSolver::new(matrix, 2, 100, 1e-6);
    let result = solver.solve().unwrap();
    println!("{:?}", result);
}
