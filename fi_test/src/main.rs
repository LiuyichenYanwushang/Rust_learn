fn main() {
    let x_min:f64=1e-7;
    let x_max:f64=5.0;
    //let y:f64=integrate(x_min,x_max,/* f64 */);
    let pre:f64=1e-8;
    let y=integrate(x_min,x_max,pre);
    println!("{}",y);
}

fn function(x:f64)->f64{
    //return x.powf(x);
    return (x*x.ln()).exp();
}
fn simption(x_min:f64,x_max:f64)->f64{
    let x_mid=(x_min+x_max)/2.0;
    let y1:f64=function(x_min);
    let y2:f64=function(x_max);
    let y0:f64=function(x_mid);
    let y:f64=(x_max-x_min)/6.0*(y1+y2+4.0*y0);
    return y;
}

fn integrate(x_min:f64,x_max:f64,pre:f64)->f64{
    let x_mid=(x_min+x_max)/2.0;
    let y0=simption(x_min,x_max);
    let y1=simption(x_min,x_mid);
    let y2=simption(x_mid,x_max);
    let err=(y1+y2-y0).abs()/15.0;
    if err>pre{
        let a:f64=integrate(x_min,x_mid,pre/2.0);
        let b:f64=integrate(x_mid,x_max,pre/2.0);
        return a+b;
    } else{
        return y0;
    }
}
