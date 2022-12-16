fn main() {
    let mut n:u64=600851475143;
    let mut result:Vec<u64>=Vec::new();
    let mut i:u64=3;
    loop{
        if judge(i){
            let s=n%i;
            if s==0{
                result.push(i);
                println!("{}",i);
                n=n/i;
            }
            else {
                i+=2;
            }

        } else{
            i+=2;
        }
        if i>n{
            break;
        }

    }

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
