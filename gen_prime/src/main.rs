pub use ndarray::*;

fn main() {
    let n=133524;
    let mut v:Vec<u64>=Vec::new(); //create a empty vector
    let mut i:u64=2;
    v.push(2)
    while i < n{
        let a0=i as f64;
        let b=a0.sqrt()+1.0;
        if judge(i){
            v.push(i);
            println!("{}",i);
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
