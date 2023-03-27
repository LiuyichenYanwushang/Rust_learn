use plotters::prelude::*;
use ndarray::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 创建一个三维数组，包含 100x100 的网格数据
    let mut data = Array::zeros((100, 100));

    // 在数组中设置一些值，这里以随机数为例
    for i in 0..100 {
        for j in 0..100 {
            data[[i, j]] = rand::random();
        }
    }

    // 创建一个新的绘图区域
    let root = BitMapBackend::new("output.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    // 创建一个新的三维坐标轴，并指定 x 轴、y 轴和 z 轴的范围
    let mut chart = ChartBuilder::on(&root)
        .caption("3D Heatmap", ("sans-serif", 30))
        .x_label_area_size(40)
        .y_label_area_size(40)
        .margin(5)
        .build_cartesian_3d(-10.0..10.0, -10.0..10.0, 0.0..1.0)?;

    // 绘制三维热度图
    chart
        .surface_layer(
            data.slice(s![.., ..]).iter().cloned(),
            100,
            100,
            None,
            &BLUE.mix(0.8),
        )?;

    Ok(())
}
