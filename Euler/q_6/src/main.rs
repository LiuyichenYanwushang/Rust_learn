fn main() {
    let n=100;
    let mut a=0;
    let mut b=0;
    for i in 1..n{
        a+=i*i;
        b+=i;
    }
    b=b*b;
    println!("{}",b-a);
}
