fn main() {
    let mut x1=1;
    let mut x2=1;
    let n=10;
    let mut i=0;
    while i<n{
        let x3=x1+x2;
        x1=x2;
        x2=x3;
        i+=1;
        println!("{}",x2);
    }
}
