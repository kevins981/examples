#include <torch/torch.h>

#include <cmath>
#include <cstdio>
#include <iostream> 

//TODO: do we need to specify the data type for the tensors?
//TODO: use TensorList instead of vector of tensors? What is the difference?

struct DLRM_NetImpl : torch::nn::Module {

  // embedding layers
  //torch::nn::ModuleList emb_l; 
  std::vector<torch::nn::EmbeddingBag> emb_l;
  std::vector<torch::Tensor> v_W_l;
  int m_spa;
  // mlp layers
  torch::nn::Sequential bot_l, top_l;
  
  DLRM_NetImpl(
      std::vector<int> ln_emb, 
      int m_spa,
      std::vector<int> ln_bot, 
      std::vector<int> ln_top,
      int sigmoid_bot, 
      int sigmoid_top)
        : emb_l(create_emb(m_spa, ln_emb)),
        //: emb_l(hardcode_emb(m_spa, ln_emb)),
          m_spa(m_spa),
          bot_l(create_mlp(ln_bot, sigmoid_bot)),
          //bot_l(hardcode_bot_mlp(ln_bot, sigmoid_bot)),
          top_l(create_mlp(ln_top, sigmoid_top))
 {
  // DLRM submodules
  //register_module("emb_l", emb_l);
  register_module("top_l", top_l);
  register_module("bot_l", bot_l);
  
  for (auto i = 0; i < ln_emb.size(); i++) {
    // Asuming weighted_pooling is none for now
    v_W_l.push_back(torch::Tensor());
    //v_W_l.push_back({});
  }
 }

  torch::Tensor forward(torch::Tensor dense_x, torch::Tensor lS_o, std::vector<torch::Tensor> lS_i){
    torch::Tensor p;
    p = apply_mlp(dense_x, this->bot_l);
    //std::cout << "forward pass results:" << p << std::endl;

    std::vector<torch::Tensor> ly;
    ly = apply_emb(lS_o, lS_i);
     
    // TODO: clamp output if needed
    return p;
  }

  torch::Tensor apply_mlp(torch::Tensor dense_x, torch::nn::Sequential layers){
    return layers->forward(dense_x);
  }

  std::vector<torch::Tensor> apply_emb(torch::Tensor lS_o, std::vector<torch::Tensor> lS_i) {
                          //torch::nn::ModuleList emb_l, 
                          //std::vector<torch::Tensor> v_W_l){
    std::vector<torch::Tensor> ly;
    for (auto i = 0; i < lS_i.size(); i++) {
      torch::Tensor sparse_index_group_batch = lS_i[i];
      torch::Tensor sparse_offset_group_batch = lS_o.index({i});
      std::cout << "index: " << sparse_index_group_batch << std::endl;
      std::cout << "offset: " << sparse_offset_group_batch << std::endl;

      torch::nn::EmbeddingBag E = emb_l[i];
      torch::Tensor V = E->forward(sparse_index_group_batch, sparse_offset_group_batch);
      
      std::cout << "embedding result " << V << std::endl;
      ly.push_back(V);
    }
    
    return ly;
  }

  // TODO: need to return both emb_l and v_W_l
  // TODO: assuming weighted_pooling is none for now
  std::vector<torch::nn::EmbeddingBag> create_emb(int m ,std::vector<int> ln){
    // m is the sparse feature size
    // ln is a list of enbedding dimensions. E.g. [4,3,2]
    //torch::nn::ModuleList emb_l;
    std::vector<torch::nn::EmbeddingBag> emb_l;
    for (auto i = 0; i < ln.size(); i++) {
      int n = ln[i];
      torch::nn::EmbeddingBag EE(torch::nn::EmbeddingBagOptions(n,m).sparse(true).mode(torch::kSum));
      // TODO: need to use high, low to initialize randomly
      EE.get()->weight.data() = torch::randn({n, m});
      emb_l.push_back(EE);
    }
    return emb_l;
  }

  // DEBUGing hardcoding weights to test correctness
  std::vector<torch::nn::EmbeddingBag> hardcode_emb(int m ,std::vector<int> ln){
    std::vector<torch::nn::EmbeddingBag> emb_l;
    int i = 0;
    int n = ln[i];
    torch::nn::EmbeddingBag EE(torch::nn::EmbeddingBagOptions(n,m).sparse(true).mode(torch::kSum));
    EE.get()->weight.data() = torch::tensor({{-0.44032, -0.10196},
                                             { 0.238  , -0.31751},
                                             {-0.32455,  0.03155},
                                             { 0.03183,  0.1344 }});
    emb_l.push_back(EE);

    i = 1;
    n = ln[i];
    torch::nn::EmbeddingBag EE1(torch::nn::EmbeddingBagOptions(n,m).sparse(true).mode(torch::kSum));
    EE1.get()->weight.data() = torch::tensor({{ 0.40349,  0.25918},
                                              { 0.1282 ,  0.25686},
                                              {-0.20443, -0.15959}});
    emb_l.push_back(EE1);

    i = 2;
    n = ln[i];
    torch::nn::EmbeddingBag EE2(torch::nn::EmbeddingBagOptions(n,m).sparse(true).mode(torch::kSum));
    EE2.get()->weight.data() = torch::tensor({{-0.38429, -0.29173},
                                              {0.18523, -0.57685}});
    emb_l.push_back(EE2);
    return emb_l;
  }


  // DEBUGing hardcoding weights and biases to test correctness
  torch::nn::Sequential hardcode_bot_mlp(std::vector<int> ln, int sigmoid_layer){
    torch::nn::Sequential layers;
    int i = 0;
    int n = ln[i];
    int m = ln[i+1];
    torch::nn::Linear LL(torch::nn::LinearOptions(n, m).bias(true));
    LL.get()->weight.data() = torch::tensor({{-0.99188, -0.95116, -1.47006, -0.12516},
                                             {-0.37202, -0.94831,  1.26233,  0.0187},
                                             {-0.18422, -0.38755,  0.55569, -0.12921}});
    LL.get()->bias.data() = torch::tensor({-0.06519, -0.9588, 0.00782});
    layers->push_back(LL);
    if (i == sigmoid_layer) {
      layers->push_back(torch::nn::Sigmoid());
    } else {
      layers->push_back(torch::nn::ReLU());
    }

    i = 1;
    n = ln[i];
    m = ln[i+1];
    torch::nn::Linear LL2(torch::nn::LinearOptions(n, m).bias(true));
    LL2.get()->weight.data() = torch::tensor({{ 0.21337, -0.58605,  0.1744 },
                                              { 0.23455,  0.7427 , -1.28533}});
    LL2.get()->bias.data() = torch::tensor({0.4119, -0.50995});
    layers->push_back(LL2);
    if (i == sigmoid_layer) {
      layers->push_back(torch::nn::Sigmoid());
    } else {
      layers->push_back(torch::nn::ReLU());
    }


    return layers;
    
  }

  torch::nn::Sequential create_mlp(std::vector<int> ln, int sigmoid_layer){
    // ln is a list of layer dimensions in the mlp. E.g. [128,128,128,128]
    //torch::nn::ModuleList layers;
    torch::nn::Sequential layers;
    //for (auto i = ln.begin(); i != ln.end(); ++i) {
    for (auto i = 0; i < ln.size()-1; i++) {
      int n = ln[i];
      int m = ln[i+1];
      torch::nn::Linear LL(torch::nn::LinearOptions(n, m).bias(true));
    
      //double mean = 0.0;
      //double std_dev = sqrt(2 / (m + n));
      // TODO: how to use mean and std_dev
      // TODO: require_grad? need this for inf?

      LL.get()->weight.data() = torch::randn({m, n});
      LL.get()->bias.data() = torch::randn({m});
      layers->push_back(LL);

      if (i == sigmoid_layer) {
        layers->push_back(torch::nn::Sigmoid());
      } else {
        layers->push_back(torch::nn::ReLU());
      }
    }
    //return torch::nn::Sequential(*layers);
    return layers;
    //std::vector<torch::Tensor> v_W_l;
  }
};

TORCH_MODULE(DLRM_Net);

int main(int argc, const char* argv[]) {
  torch::manual_seed(1);

  // Create the device we pass around based on whether CUDA is available.
  torch::Device device(torch::kCPU);
  // use --arch-mlp-bot=128-128-128-128 --arch-mlp-top=512-512-512-256-1 as examples
  //std::vector<int> ln_bot = {128, 128, 128, 128};
  //std::vector<int> ln_top = {512, 512, 512, 256, 1};
  std::vector<int> ln_emb = {4,3,2};
  std::vector<int> ln_bot = {4,3,2};
  std::vector<int> ln_top = {8,4,2,1};
  int m_spa = 2;

  DLRM_Net dlrm(ln_emb, m_spa,
                ln_bot, ln_top, -1, ln_top.size()-2);

  for (const auto& pair : dlrm->named_parameters()) {
    std::cout << pair.key() << ": " << pair.value() << std::endl;
  }

  std::cout << "embedding layers: " << c10::str(dlrm->emb_l) << std::endl;
  std::cout << "top mlp layers: " << c10::str(dlrm->top_l) << std::endl;
  std::cout << "bot mlp layers: " << c10::str(dlrm->bot_l) << std::endl;
  
  dlrm->to(device);
  //torch::Tensor dense = torch::tensor({0.6965, 0.2861, 0.2269, 0.5513}, {torch::kFloat64});
  torch::Tensor dense = torch::tensor({0.6965, 0.2861, 0.2269, 0.5513});
  torch::Tensor offset = torch::tensor({{0}, {0}, {0}});
  std::vector<torch::Tensor> indice = {torch::tensor({1,2,3}), torch::tensor({1}), torch::tensor({1})};
  dlrm(dense, offset, indice);


}
