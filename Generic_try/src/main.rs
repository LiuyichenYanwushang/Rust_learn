struct Point<T> {
x: T,
y: T,
}

impl<T> Point<T> {
    fn new(x: T, y: T) -> Self {
        Self {
            x,
            y,
        }
    }
}

fn main() {
let point = Point::new(vec![0.0,1.0],vec![1.0,2.0]);
println!("Point x: {:?}, y: {:?}", point.x, point.y);
}
