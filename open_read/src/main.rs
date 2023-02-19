use ndarray::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use rand_isaac::isaac64::Isaac64Rng;
use std::fs::File;
use std::io::Write;

fn main() -> std::io::Result<()> {
    let seed = 42;
    let mut rng = Isaac64Rng::seed_from_u64(seed);

    let array = Array::random_using((1000, 5), Uniform::new(0., 10.), &mut rng);
    let mut file = File::create("data.txt")?;
    for i in 0..1000{
        //writeln!(file,"  {}", array.slice(s![i,..]))?;
        let mut s = String::new();
        for j in 1..5{
            //s.push_str(&array[[i,j]].to_string());
            let aa= format!("{:.3}", array[[i,j]]);
            s.push_str(&aa);
            s.push_str("  ");
             
        }
        //let string=array.slice(s![i,..]).to_string();
        writeln!(file,"{}",s)?;
    }
    Ok(())
}
