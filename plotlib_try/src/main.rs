use ndarray::{Array, Array2};
use plotly::{Bar, Scatter, ScatterMode, Plot, Layout};


fn main() {
// 创建一个二维数组，用于构建折线图
let data = Array2::from_shape_vec((2, 5), vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).unwrap();

// 使用 Plotly Rust 创建折线图
let mut plot = Plot::new();

// 添加折线图数据
plot.add_trace(Scatter::new(data.row(0).to_vec(), data.row(1).to_vec(), ScatterMode::Lines));

// 设置图表标题
plot.set_layout(Layout::new().title("Simple Line Chart"));

// 生成图表
plot.show();
}
