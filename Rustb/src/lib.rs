use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

pub fn add(left: f64, right: f64) -> f64 {
    return left + right
}

fn sum_as_string(a:usize,b:usize)->PyResult<String>{
    Ok((a+b).to_string())
}

fn test(_py: Python<'_>,m: &PyModule)->PyResult<()> {
    m.add_function(wrap_pyfunction!(add,m)?)?;
    m.add_function(wrap_pyfunction!(sum_as_string,m)?)?;
    OK(())
}
