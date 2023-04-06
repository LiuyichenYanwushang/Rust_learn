extern crate gnuplot;

use gnuplot::{Figure, Caption, Color};

fn main() {
    // 创建一个新的 figure
    let mut fg = Figure::new();

    // 绘制正弦函数
    let x = (-100..100).step_by(1).map(|i| i as f64 / 10.0);
    let y = x.clone().map(|x| x.sin());
    fg.axes2d().lines(x, y, &[Color("red"), Caption("sin(x)")]);

    // 保存图像为 PDF 文件
    fg.set_terminal("pdf", "output.pdf");
    fg.show();
}
