use std::string::ToString;

fn adjust_even(a:u64)->bool{
    let b=a.to_string();
    let n=b.len();
    println!("{}",b.as_bytes()[-1]);
    return b==b

}
fn main() {
    let n=10;
    println!("{}",adjust_even(n));
}
