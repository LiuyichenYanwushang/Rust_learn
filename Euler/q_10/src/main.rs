fn main() {
    let mut a0=2;
    for i in 3..2000000{
        if judge(i){
            a0+=i;
        }
    }
    println!("{}",a0)
        
}

fn judge(a:u64)-> bool{
    let a0=a as f64;
    let b=a0.sqrt()+1.0;
    let mut i=3.0;
    let mut back=true;
    if a0%2.0==0.0{
        back=false;
        //println!("it's a even number")
    } else if a0%10.0==5.0 { 
        back=false;
        //println!("it can be divisible by 5");
    } else {
        while i<b{
            let c=a0%i;
            if c==0.0{
                back=false;
                //println!("it can be divisible by {}",i);
                break;
            }
            i+=2.0;
            if i%10.0==5.0 {
                i+=2.0;
            }
        }
    }
    return back;
}
