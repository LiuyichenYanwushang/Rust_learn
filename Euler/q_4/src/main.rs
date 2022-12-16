fn main() {
    let mut n0=0;
    for i in 100..999{
        for j in 100..999{
            let n=i*j;
            let num1=n.to_string();
            let mut new_char=String::new();
            for c in num1.chars().rev() {
                new_char.push(c)
            }
            if new_char==num1{
                if n>n0{
                    n0=n;
                }
            }
        }
    }
    println!("{}",n0)
    
}
