fn main() {
    let mut i:u128=6000;
//    let mut aa=0;
    loop{
        i+=1;
        let b=gen_trig(i);
        let a=gen_reduce(b).len();
        if a>500{
            println!("{}",b);
            break;
        } 
//        else if a>aa{
//            println!("{},{},{}",i,b,a);
//            aa=a.clone();
//        }
//        println!("{}",i)
    }
}

fn gen_trig(n:u128)->u128{
    let mut a:u128=0;
    a=n*(n+1)/2;
    return a;
}

fn gen_reduce(n:u128)->Vec<u128>{
    let mut a:Vec<u128>=Vec::new();
    for i in 1..n+1{
        if n%i==0{
            a.push(i);
        }
    }
    return a;

}
