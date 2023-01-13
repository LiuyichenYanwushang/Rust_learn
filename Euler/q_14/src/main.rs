fn main() {
    let n0=10000000;
    let mut nn=0;
    for i in 2..n0{
        let n=gen_calo(i);
        if nn<n{
            nn=n;
            println!("{},{}",nn,i)
        }
   }
}

fn gen_calo(n:u64)->u64{
    let mut n0=n;
    let mut i=1;
    loop{
        if judge_even(n0){
            n0=n0/2;
        }else{
            n0*=3;
            n0+=1;
        }
        i+=1;
        if n0==1{
            break;
        }
    }
    return i;
}
fn judge_even(n:u64)->bool{
    let n0=n.to_string();
    let leng=n0.len();
    let n_char=&n0[leng-1..leng];
    return n_char=="0"||n_char=="2"||n_char=="4"||n_char=="6"||n_char=="8";
}
