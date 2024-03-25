use tch::nn::OptimizerConfig;
use tch::{
    nn::{self, Module, Optimizer, Path, VarStore},
    Device, Kind, Tensor,
};
const POSITION_SIZE: i64 = 1;
const DATA_SIZE: i64 = 1;
const POSITIONED_DATA_SIZE: i64 = POSITION_SIZE + DATA_SIZE;
const HIDDEN_SIZE: i64 = 1;
const DEPTH: i64 = 1;

fn build_net(
    vs: &nn::Path,
    name: &str,
    input_size: i64,
    output_size: i64,
    hidden_layer_count: i64,
) -> impl Module {
    let mut net = nn::seq()
        .add(nn::linear(
            vs / format!("{name}_input"),
            input_size,
            HIDDEN_SIZE,
            Default::default(),
        ))
        .add_fn(|xs| xs.relu());

    for i in 0..hidden_layer_count {
        net = net
            .add(nn::linear(
                vs / format!("{name}_hidden_layer{i}"),
                HIDDEN_SIZE,
                HIDDEN_SIZE,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu());
    }

    net = net.add(nn::linear(
        vs / format!("{name}_output"),
        HIDDEN_SIZE,
        output_size,
        Default::default(),
    ));

    net
}

fn get_device() -> Device {
    // ISSUE: the grad is right with CPU device.
    // return Device::Cpu;
    if tch::utils::has_mps() {
        return Device::Mps;
    }

    Device::cuda_if_available()
}

fn encode_position(x: &Tensor, device: Device) -> Tensor {
    let size = x.size();
    let l = size.len();

    let mut position_size = size.clone();
    position_size[l - 1] = POSITION_SIZE;
    let positions = Tensor::zeros(position_size, (Kind::Float, device));

    for i in 0..x.size()[0] {
        positions.get(i).copy_(&Tensor::from_slice(&[i]));
    }

    let y = Tensor::concat(&[&positions, x], (l - 1) as i64);
    y
}

struct Layout {
    dummy_net: Box<dyn Module>,
}

impl Layout {
    fn new(root: &Path) -> Self {
        Layout {
            dummy_net: Box::new(build_net(
                &root,
                "dummy_net",
                POSITIONED_DATA_SIZE,
                DATA_SIZE,
                DEPTH,
            )),
        }
    }

    fn layout1(&self, x: &Tensor) -> Tensor {
        self.dummy_net.forward(x)
    }
}

struct Bot {
    predictor: Layout,
    vs: VarStore,
    opt: Optimizer,
    pred_input: Option<Tensor>,
    device: Device,
}

impl Bot {
    fn new(device: Device) -> Self {
        let vs = nn::VarStore::new(device);
        let predictor = Layout::new(&(vs.root() / "predictor"));
        let lr = 1e-5;
        let opt = nn::Adam::default().build(&vs, lr).unwrap();
        Bot {
            predictor,
            vs,
            opt,
            pred_input: None,
            device,
        }
    }

    fn tick(&mut self, input: &Tensor) -> () {
        let device = self.device;

        // ISSUE HERE: use the line below with metal to see zero gradients...
        let mut pred_loss = Tensor::zeros(&[1], (Kind::Float, device));
        // With the line below, the issue goes away
        // let mut pred_loss;
        if let Some(pred_input) = &self.pred_input {
            pred_loss = input.mse_loss(&pred_input, tch::Reduction::Mean);
            self.opt.backward_step(&pred_loss);
            
            self.vs.variables().iter().for_each(|(name, v)| {
                if v.grad().defined() == false {
                    panic!("{} is not defined", name);
                }
            });
            let gs = self
                .vs
                .variables()
                .iter()
                .map(|(_name, v)| v.grad())
                .filter(|g| g.defined())
                .map(|g| g.sum(Kind::Float))
                .map(|g| f32::try_from(g).unwrap())
                .collect::<Vec<f32>>();
            println!(
                "\n\ninput: {}, \n\npred_input: {}, \n\npred_loss: {}, \n\ngrads: {:?}",
                input.to_device(Device::Cpu),
                pred_input.to_device(Device::Cpu),
                pred_loss.to(Device::Cpu),
                gs
            );
        }

        let input = &input.detach();
        let input = &encode_position(input, device);
        self.pred_input = Some(self.predictor.layout1(input));
    }
}

fn running() {
    let device = get_device();
    let mut bot = Bot::new(device.clone());

    const BATCH_SIZE: i64 = 1;
    let length = 40 * BATCH_SIZE;
    let count = length / BATCH_SIZE;
    let data = (0i64..length).map(|v| v as f32).collect::<Vec<f32>>();

    let x_dataset = Tensor::from_slice(data.as_slice()).to_device(device)
        .reshape(&[count, BATCH_SIZE, DATA_SIZE]);

    for i in 0..count {
        println!("=====");

        let x = &x_dataset.get(i);
        let _ = bot.tick(&x);
    }
}
fn main() {
    running();
}
