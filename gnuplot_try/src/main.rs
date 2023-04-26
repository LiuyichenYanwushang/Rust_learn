use gnuplot::{Figure, Caption, Color};
use ndarray::{Array, Array2};

fn main() {
//!首先，我们定义了二维函数的定义域和取值范围。然后，我们使用 ndarray 库中的 Array::range 方法生成相应的坐标轴范围。接下来，我们使用嵌套循环计算每个坐标点上的函数值，并将这些函数值保存在一个二维数组中。最后，我们使用 gnuplot 库来绘制生成的函数值数组。在绘制函数时，我们使用 gnuplot 中的 image 函数来绘制颜色图，并使用其他函数来添加标签、标题和网格线。

//!请注意，此代码假定您已经在您的 Cargo.toml 文件中包含了 ndarray 和 gnuplot 库的依赖项，如下所示：
    let x_min = -5.0;
    let x_max = 5.0;
    let y_min = -5.0;
    let y_max = 5.0;
    let num_points = 100;

    let x_range = Array::range(x_min, x_max, (x_max - x_min) / num_points as f64);
    let y_range = Array::range(y_min, y_max, (y_max - y_min) / num_points as f64);

    let mut z = Array2::<f64>::zeros((num_points, num_points));
    for (i, x) in x_range.iter().enumerate() {
        for (j, y) in y_range.iter().enumerate() {
            let numerator = x.powi(2) + y.powi(2);
            let denominator = x * x.sin() + y * y.sin();
            z[[j, i]] = numerator / denominator;
        }
    }

    let mut fg = Figure::new();
    fg.axes2d()
        .image(z.t().to_owned().as_slice(), x_min, y_min, x_max, y_max, &[Color("white")])
        .xlabel("X")
        .ylabel("Y")
        .set_title("z=(x^2+y^2)/(x*sin(x)+y*sin(y))", &[])
        .set_grid_options(false, &[("xtics", "10"), ("ytics", "10")]);
    fg.show().unwrap();
}





