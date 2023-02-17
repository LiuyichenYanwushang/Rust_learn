use std::f64;
use rand::random;

// 定义结构体
struct Ising3D {
    size: usize,
    energy: f64,
    magnetization: f64,
    spins: Vec<i32>,
    temperature: f64,
    j: f64
}

impl Ising3D {
// 初始化函数
    fn new(size: usize, temperature: f64, j: f64) -> Ising3D {
        let mut spins = Vec::with_capacity(size * size * size);
        for _ in 0..size * size * size {
            spins.push(1);
        }
        Ising3D {
            size: size,
            energy: 0.0,
            magnetization: 0.0,
            spins: spins,
            temperature: temperature,
            j: j
        }
    }
    fn calculate_energy(&mut self) -> f64 {
        let mut energy = 0.0;
        for x in 0..self.size {
            for y in 0..self.size {
                for z in 0..self.size {
                    let left = self.spins[(x + 1) % self.size + (y * self.size) + (z * self.size * self.size)];
                    let right = self.spins[(x - 1 + self.size) % self.size + (y * self.size) + (z * self.size * self.size)];
                    let up = self.spins[x + ((y + 1) % self.size * self.size) + (z * self.size * self.size)];
                    let down = self.spins[x + ((y - 1 + self.size) % self.size * self.size) + (z * self.size * self.size)];
                    let front = self.spins[x + (y * self.size) + ((z + 1) % self.size * self.size * self.size)];
                    let back = self.spins[x + (y * self.size) + ((z - 1 + self.size) % self.size * self.size * self.size)];
                    energy += (self.spins[x + (y * self.size) + (z * self.size * self.size)]
                        * (left + right + up + down + front + back)) as f64;
                }
            }
        }
        self.energy = -self.j * energy;
        self.energy
    }

    // 计算磁化函数
    fn calculate_magnetization(&mut self) -> f64 {
        let mut magnetization = 0.0;
        for x in 0..self.size {
            for y in 0..self.size {
                for z in 0..self.size {
                    magnetization += (self.spins[x + (y * self.size) + (z * self.size * self.size)]) as f64;
                }
            }
        }
        self.magnetization = magnetization;
        self.magnetization
    }

    // 模拟函数
    fn simulate(&mut self, steps: usize) {
        for _ in 0..steps {
            for x in 0..self.size {
                for y in 0..self.size {
                    for z in 0..self.size {
                        let i = x + (y * self.size) + (z * self.size * self.size);
                        let left = self.spins[(x + 1) % self.size + (y * self.size) + (z * self.size * self.size)];
                        let right = self.spins[(x - 1 + self.size) % self.size + (y * self.size) + (z * self.size * self.size)];
                        let up = self.spins[x + ((y + 1) % self.size * self.size) + (z * self.size * self.size)];
                        let down = self.spins[x + ((y - 1 + self.size) % self.size * self.size) + (z * self.size * self.size)];
                        let front = self.spins[x + (y * self.size) + ((z + 1) % self.size * self.size * self.size)];
                        let back = self.spins[x + (y * self.size) + ((z - 1 + self.size) % self.size * self.size * self.size)];
                        let delta_energy = 2.0 * (self.spins[i] * (left + right + up + down + front + back)) as f64;
                        let x: f64 = rand::random();
                        if delta_energy <= 0.0 || (f64::exp(-delta_energy / self.temperature)) > x {
                            self.spins[i] *= -1;
                        }
                    }
                }
            }
        }
    }
}
fn main() {
// 初始化模型
let mut ising = Ising3D::new(100, 10.0, 1.0);
// 模拟1000步
ising.simulate(1000);
// 计算能量
println!("Energy: {}", ising.calculate_energy());
// 计算磁化
println!("Magnetization: {}", ising.calculate_magnetization());
}
