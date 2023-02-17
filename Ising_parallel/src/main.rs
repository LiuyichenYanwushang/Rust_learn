// 三维 Ising 模型模拟

use std::sync::{Arc, Mutex};
use std::thread;
use rand::random;

// 定义模拟参数
const J: f64 = 1.0;
const BETA: f64 = 1.0;
const N: usize = 1000;

// 定义网格状态
struct Grid {
state: Vec<Vec<Vec<i32>>>,
energy: f64,
magnetization: f64
}

// 初始化网格
fn init_grid() -> Grid {
let mut state = vec![vec![vec![0; N]; N]; N];
let mut energy = 0.0;
let mut magnetization = 0.0;
// 初始化网格状态
for i in 0..N {
    for j in 0..N {
        for k in 0..N {
            let r: f64 = rand::random();
            if r > 0.5 {
                state[i][j][k] = 1;
                magnetization += 1.0;
            } else {
                state[i][j][k] = -1;
                magnetization -= 1.0;
            }
        }
    }
}

// 计算初始能量
for i in 0..N {
    for j in 0..N {
        for k in 0..N {
            energy += -J * (state[i][j][k] * state[(i + 1) % N][j][k] + state[i][(j + 1) % N][k] + state[i][j][(k + 1) % N]) as f64 ;
        }
    }
}

Grid {
    state,
    energy,
    magnetization
}
}

// 模拟步骤
fn step(grid: &mut Grid) {
// 随机选择一个网格点
let i: usize = rand::random::<usize>() % N;
let j: usize = rand::random::<usize>() % N;
let k: usize = rand::random::<usize>() % N;

// 计算翻转点之前和之后的能量差
let delta_e = 2.0 * J * ((grid.state[i][j][k] * (grid.state[(i + 1) % N][j][k] + grid.state[i][(j + 1) % N][k] + grid.state[i][j][(k + 1) % N])) as f64);

// 根据能量差计算接受概率
let p_accept = (-delta_e * BETA).exp();

// 根据接受概率决定是否接受翻转
let r: f64 = rand::random();
if r < p_accept {
    grid.state[i][j][k] *= -1;
    grid.energy += delta_e;
    grid.magnetization += 2.0 * grid.state[i][j][k] as f64;
}
}

fn main() {
let mut grid = init_grid();

// 将网格状态封装成Arc类型，用于多线程访问
let grid = Arc::new(Mutex::new(grid));

// 创建线程池
let mut threads = vec![];
for _ in 0..16 {
    let grid = grid.clone();
    threads.push(thread::spawn(move || {
        for _ in 0..10000 {
            let mut grid = grid.lock().unwrap();
            step(&mut grid);
        }
    }));
}

// 等待线程池完成
for t in threads {
    t.join().unwrap();
}

// 输出结果
let grid = grid.lock().unwrap();
println!("Energy: {}, Magnetization: {}", grid.energy, grid.magnetization);
}





