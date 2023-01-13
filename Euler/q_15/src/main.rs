fn main() {
    let n:u128=20;
    println!("{}",solve(n,n))
    
}
fn solve(a:u128,b:u128)->u128{
    if a>0 && b>0{
        return solve(a-1,b)+solve(a,b-1)
    }else{
        return 1
    }
}
