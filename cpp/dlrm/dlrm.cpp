#include <torch/torch.h>

#include <cmath>
#include <cstdio>
#include <iostream> 

struct DLRM_NetImpl : torch::nn::Module {

  // embedding layers
  torch::nn::ModuleList emb_l; 
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
          m_spa(m_spa),
          bot_l(create_mlp(ln_bot, sigmoid_bot)),
          top_l(create_mlp(ln_top, sigmoid_top))
 {
  // DLRM submodules
   register_module("emb_l", emb_l);
   register_module("top_l", top_l);
   register_module("bot_l", bot_l);

   for (auto i = 0; i < ln_emb.size(); i++) {
     // Asuming weighted_pooling is none for now
     v_W_l.push_back(torch::Tensor());
     //v_W_l.push_back({});
   }
 }

 //torch::Tensor forward(torch::Tensor dense_x, lS_o, lS_i) {
 //   torch::Tensor x = apply_mlp(dense_x, this->bot_l);
 //   torch::Tensor ly = apply_emb(lS_o, lS_i, this->emb_l, this->v_W_l);
 //   torch::Tensor z = interact_features(x, ly);
 //   torch::Tensor p = apply_mlp(z, this->top_l);
 //   // TODO: clamp output if needed
 //  return p;
 //}
 torch::Tensor forward(){
    torch::Tensor p;
    // TODO: clamp output if needed
   return p;
 }

  // TODO: need to return both emb_l and v_W_l
  // TODO: assuming weighted_pooling is none for now
  torch::nn::ModuleList create_emb(int m ,std::vector<int> ln){
    // m is the sparse feature size
    // ln is a list of enbedding dimensions. E.g. [4,3,2]
    torch::nn::ModuleList emb_l;
    for (auto i = 0; i < ln.size(); i++) {
      int n = ln[i];
      torch::nn::EmbeddingBag EE(torch::nn::EmbeddingBagOptions(n,m).sparse(true).mode(torch::kSum));
      // TODO: need to use high, low to initialize randomly
      EE.get()->weight.data() = torch::randn({n, m});
      emb_l->push_back(EE);
    }
    return emb_l;
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
    
      double mean = 0.0;
      double std_dev = sqrt(2 / (m + n));
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
  std::vector<int> ln_bot = {2,3};
  std::vector<int> ln_top = {1,2,3,4};
  int m_spa = 10;

  DLRM_Net dlrm(ln_emb, m_spa,
                ln_bot, ln_top, -1, ln_top.size()-2);

  for (const auto& pair : dlrm->named_parameters()) {
    std::cout << pair.key() << ": " << pair.value() << std::endl;
  }

  std::cout << "embedding layers: " << c10::str(dlrm->emb_l) << std::endl;
  std::cout << "top mlp layers: " << c10::str(dlrm->top_l) << std::endl;
  std::cout << "bot mlp layers: " << c10::str(dlrm->bot_l) << std::endl;
  


  dlrm->to(device);


}
