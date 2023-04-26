use ndarray::Array2;

fn draw_heatmap(data: Array2<f64>) {
    use gnuplot::{Figure, AxesCommon, AutoOption::Fix,HOT};
    let mut fg = Figure::new();

    let (width, height) = (data.shape()[1], data.shape()[0]);
    let mut heatmap_data = vec![];

    for i in 0..height {
        for j in 0..width {
            heatmap_data.push(data[(i, j)]);
        }
    }

    let axes = fg.axes2d();
    axes.set_title("Heatmap", &[]);
    axes.set_cb_label("Values", &[]);
    axes.set_palette(HOT);
    axes.image(heatmap_data.iter(), width, height,None, &[]);
    fg.set_terminal("pngcairo","out.png");

    fg.show().expect("Unable to draw heatmap");
}

fn main() {
    let data = Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .expect("Unable to create 2D array");
    draw_heatmap(data);
}

