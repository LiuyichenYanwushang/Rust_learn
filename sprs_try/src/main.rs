use sprs::{CsMatSymM};
use sprs::kernels::eigen::SymEigen;
fn main() {
    // 创建一个稀疏矩阵
    let data = vec![1.0, 2.0, 1.0, 3.0, 2.0, 3.0];
    let indices = vec![0, 1, 2, 0, 1, 2];
    let indptr = vec![0, 2, 4, 6];
    let mat = CsMatSymM::new((3, 3), indptr, indices, data);

    // 计算稀疏矩阵的本征值和特征向量
    let eig = SymEigen::new(&mat);
    let values = eig.values();
    let vectors = eig.vectors();

    println!("Eigenvalues: {:?}", values);
    println!("Eigenvectors: {:?}", vectors);
}
