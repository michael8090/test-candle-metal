use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{ Module, VarBuilder};
use candle_nn as nn;

use nn::{AdamW, Optimizer, VarMap};

fn main() {

    let device = Device::new_metal(0).unwrap();
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let l1 = nn::linear(1, 1, vs.pp("1")).unwrap();
    let l2 = nn::linear(1, 1, vs.pp("2")).unwrap();

    println!("metal");
    println!("{:?}", l1);
    println!("{:?}", l2);


    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let l1 = nn::linear(1, 1, vs.pp("1")).unwrap();
    let l2 = nn::linear(1, 1, vs.pp("2")).unwrap();
    
    println!("cpu");
    println!("{:?}", l1);
    println!("{:?}", l2);


}
