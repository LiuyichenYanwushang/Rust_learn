fn main() {
    let mut n=1;
    let mut i=3;
    let mut i0=0;
    let mut i1=0;
    loop{
        if judge(i){
            n+=1;
        }
        if n==100001{
            println!("{}",i);
            break;
        }
        i+=2;
        i0+=1;
        i1+=1;
        if i0==3{
            i+=2;
            i0=1;
        } else if i1==5{
            i+=2;
            i1=1;
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
    } else if a0%10.0==5.0 { 
        back=false;
        //println!("{},{},{}",a0,a0%10.0,back)
    } else {
        while i<b{
            let c=a0%i;
            if c==0.0{
                back=false;
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
