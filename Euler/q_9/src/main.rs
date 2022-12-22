fn main() {
    for a in 1..333{
        let b0=1000-a;
        for b in a..b0{
            let c=b0-b;
            if c>b0 || a*a+b*b==c*c{
                println!("{}^2+{}^2={}^2",a,b,c);
                println!("{}",a*b*c);
            }
        }
    }
}
