use std::string::ToString;

fn adjust_even(a:u64)->bool{
    if a%2==0{
        return true;
    } else{
        return false;
    }
        

}
fn main() {
    let mut a1:Vec<u64>=Vec::new();
    let mut i=0;
    let mut res=0;
    a1.push(1);
    a1.push(1);
    i+=1;
    loop{
        a1.push(a1[i]+a1[i-1]);
        i+=1;
        if a1[i]>4000000{
            break;
        }
        if adjust_even(a1[i]){
           res+=a1[i];
        }
    }
    println!("{}",res)
}
