fn main() {
    let mut n=1;
    for i in 2..20{
        if n%i!=0{
            n=n*i/euclidean(n,i)
        }
    }
    println!("{}",n)
}

fn euclidean(a:u64,b:u64)->u64{
    let mut a=a;
    let mut b=b;
    loop{
        if a>b{
            a=a%b;
        }else{
            b=b%a;
        }
        if a==0{
            return b;
        }else if b==0{
            return a;
        }
    }

}
