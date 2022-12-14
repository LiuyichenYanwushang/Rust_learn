fn main() {
    let n=1000;
    let mut answer=0;
    for i in 1..n{
        if i%3==0 || i%5==0{
            answer+=i
        }
    }
    println!("{}",answer)
}
