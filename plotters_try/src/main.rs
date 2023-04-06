/*
use ndarray::{Array, ArrayView, Axis, Ix1, Ix2};
use plotters::prelude::*;

fn main() {
    // 创建一个二维数组
    let data = Array::from_shape_fn((10, 2), |(i, j)| (i + j) as f64);

    // 创建一个视图来查看第一列
    let column_view: ArrayView<f64, Ix1> = data.column(0);

    // 将视图中的数据传递给绘图库
    let mut points = Vec::new();
    for (i, &value) in column_view.iter().enumerate() {
        points.push((i as i32, value));
    }

    // 创建一个新的绘图
    let root = BitMapBackend::new("plot.pdf", (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    // 创建一个折线图
    let mut chart = ChartBuilder::on(&root)
        .caption("My Plot", ("sans-serif", 20))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..10, 0.0..20.0)
        .unwrap();

    chart
        .draw_series(LineSeries::new(points, &BLUE))
        .unwrap();

    // 保存图像
    root.present().unwrap();
}
*/
use plotters::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 创建一个新的图像文件
    let root = BitMapBackend::new("heatmap.png", (640, 480)).into_drawing_area();

    // 定义图像的范围和数据
    let x_start = -3.0;
    let x_end = 3.0;
    let y_start = -3.0;
    let y_end = 3.0;
    let num_points = 100;
    let mut data = vec![0.0; num_points * num_points];
    for i in 0..num_points {
        let x = x_start + (x_end - x_start) / (num_points - 1) as f64 * i as f64;
        for j in 0..num_points {
            let y = y_start + (y_end - y_start) / (num_points - 1) as f64 * j as f64;
            data[i * num_points + j] = x * x + y * y;
        }
    }

    // 绘制热图
    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .margin(5)
        .caption("Heatmap Example", ("Arial", 20))
        .build_ranged(x_start..x_end, y_start..y_end)?;

    chart.configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .draw()?;

    chart.draw_series(
        data.iter().enumerate().map(|(i, &val)| {
            let x = i % num_points;
            let y = i / num_points;
            Rectangle::new(
                [(x_start + (x_end - x_start) / (num_points - 1) as f64 * x as f64, y_start + (y_end - y_start) / (num_points - 1) as f64 * y as f64), (x_start + (x_end - x_start) / (num_points - 1) as f64 * (x + 1) as f64, y_start + (y_end - y_start) / (num_points - 1) as f64 * (y + 1) as f64)],
                HSLColor(
                    val / (x_end * x_end + y_end * y_end) * 0.8 + 0.2,
                    1.0,
                    0.5,
                ).filled(),
            )
        })
    )?;

    Ok(())
}
