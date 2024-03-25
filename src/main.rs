use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{ Module, VarBuilder};
use candle_nn as nn;

use nn::{AdamW, Optimizer, VarMap};

const POSITION_SIZE: usize = 1;
const DATA_SIZE: usize = 1;
const POSITIONED_DATA_SIZE: usize = POSITION_SIZE + DATA_SIZE;
const EFFECT_SIZE: usize = 2;
const FIELD_SIZE: usize = 1;
const POSITIONED_FIELD_SIZE: usize = POSITION_SIZE + FIELD_SIZE;
const HIDDEN_SIZE: usize = 64;
const DEPTH: usize = 4;

fn build_net(
    vs: &VarBuilder,
    name: &str,
    input_size: usize,
    output_size: usize,
    hidden_layer_count: usize,
) -> Result<impl Module> {
    let mut net = candle_nn::seq()
        .add(nn::linear(
            input_size,
            HIDDEN_SIZE,
            vs.pp(format!("{name}_input")),
        )?)
        .add_fn(|xs| xs.relu());

    for i in 0..hidden_layer_count {
        net = net.add(nn::linear(
            HIDDEN_SIZE,
            HIDDEN_SIZE,
            vs.pp(format!("{name}_hidden_layer{i}")),
        )?)
        .add_fn(|xs| xs.relu());
    }

    net = net.add(nn::linear(
        HIDDEN_SIZE,
        output_size,
        vs.pp(format!("{name}_output")),
    )?);

    Ok(net)
}

fn field_data_effect(vs: &VarBuilder) -> Result<impl Module> {
    build_net(vs, "field_data_effect", POSITIONED_FIELD_SIZE + POSITIONED_DATA_SIZE, EFFECT_SIZE, DEPTH)
}


fn get_device() -> Device {
    if candle_core::utils::metal_is_available() {
        return Device::new_metal(0).unwrap();
    }

    Device::cuda_if_available(0).unwrap()
}
fn encode_position(x: &Tensor, device: &Device) -> Tensor {
    let shape = x.shape().dims();
    let l = shape.len();

    let mut position_shape = [0, 1];
    position_shape.copy_from_slice(shape);
    position_shape[l-1] = POSITION_SIZE;
    let positions = Tensor::zeros(candle_core::Shape::from_dims(shape), DType::F32, device).unwrap();

    for i in 0..shape[0] {
        positions.get(i).unwrap().clone_from(&Tensor::new(&[i as f32], device).unwrap());
    }

    let y = Tensor::cat(&[&positions, x], (l-1) as usize).unwrap();
    y
}

struct Layout {
    device: Device,
    field_data_effect_net: Box<dyn Module>,
    effect_to_data_net: Box<dyn Module>,
}

impl Layout {
    fn new(device: Device, root: &VarBuilder) -> Self {
        Layout {
            device,

            field_data_effect_net: Box::new(field_data_effect(&(root.pp("field_effect_net"))).unwrap()),
            effect_to_data_net: Box::new(build_net(&(root.pp("effect_to_data_net")), "effect_to_data_net",  EFFECT_SIZE, DATA_SIZE, DEPTH).unwrap()),
        }
    }

    fn layout(&self, x: &Tensor) -> Result<Tensor> {
        let line_length_x = x.dim(0)?;
        let line_length_field = line_length_x;
        {
            let mut initial_field =
                Tensor::zeros(&[line_length_field, POSITIONED_DATA_SIZE], DType::F32, &self.device)?;
            initial_field.clone_from(x);
            let mut edges = Tensor::new(&[0.0f32], &self.device).unwrap();


            for i in 0..line_length_field {
                for j in 0..line_length_x {
                    let c = Tensor::cat(&[initial_field.get(i)?, x.get(j)?], 0).unwrap();
                    edges = Tensor::cat(&[edges, c], 0).unwrap();
                }
            }

            edges = edges.i(1..).unwrap().reshape((line_length_field, line_length_x, initial_field.dim(1)? + x.dim(1)?))?;

            let effects = self.field_data_effect_net.forward(&edges)?;

            let sum_effect = effects.sum([1])?;

            return self.effect_to_data_net.forward(&sum_effect);
        }
    }
}

struct Bot <'a> {
    predictor: Layout,
    // actor: Layout,
    vs: VarBuilder<'a>,
    varmap: VarMap,
    opt: AdamW,
}

impl <'a> Bot<'a> {
    fn new() -> Self {
        let device = get_device();
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let predictor = Layout::new(device, &(vs.pp("predictor")));

        let lr = 1e-3;
        let opt = nn::AdamW::new_lr(varmap.all_vars(), lr).unwrap();

        Bot {
            predictor,
            varmap,
            // actor,
            vs,
            opt,
        }
    }

}

fn running() -> Result<()> {
    let device = get_device();
    let mut bot = Bot::new();

    const BATCH_SIZE: usize = 1;

    let length = 16 * BATCH_SIZE;
    let count = length / BATCH_SIZE;
    let data = (0usize..length).map(|v| v as f32).collect::<Vec<f32>>();

    let x_dataset = Tensor::new(data, &device)?
        .reshape(&[count, BATCH_SIZE, DATA_SIZE])?;

    let mut epoch = 0;
    loop {
        for i in 0..count {
            let x = &x_dataset.get(i).unwrap();
            let p_x = &encode_position(x, &device);
            let pred = bot.predictor.layout(p_x).unwrap();
            let y = ((x * 0.05)?.sin()?*10000.0).unwrap();

            // ISSUE HERE: MSE is not right on metal backend
            let pred_loss = nn::loss::mse(&pred, &y).unwrap();

            println!("===== pred_loss: {}", pred_loss);

            bot.opt.backward_step(&pred_loss).unwrap();

            println!("step: {}    pred_loss: {}", epoch * count + i, pred_loss.to_scalar::<f32>().unwrap());
            println!(
                "pred:\n{}\n y:\n{} \n\n",
                pred.flatten_all()?,
                y.flatten_all()?,
            );
        }
        epoch += 1;
    }
}

fn main() {
    let _ = running();
}
