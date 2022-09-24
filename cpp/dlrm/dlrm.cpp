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
        //: emb_l(create_emb(m_spa, ln_emb)),
        : emb_l(hardcode_emb(m_spa, ln_emb)),
          m_spa(m_spa),
          //bot_l(create_mlp(ln_bot, sigmoid_bot)),
          bot_l(hardcode_bot_mlp(ln_bot, sigmoid_bot)),
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
    torch::Tensor x;
    x = apply_mlp(dense_x, this->bot_l);
    //std::cout << "forward pass results:" << p << std::endl;

    std::vector<torch::Tensor> ly;
    ly = apply_emb(lS_o, lS_i);
     
    
    torch::Tensor z;
    z = interact_features(x, ly);
    // TODO: clamp output if needed
    return z;
  }

  torch::Tensor interact_features(torch::Tensor x, std::vector<torch::Tensor> ly){
    // x is the output of bottom mlp, ly the output of embedding layers
    // need to concat them into a single matrix
    // only dot interaction op for now
    //std::cout << "x  " << x << std::endl;
    //std::cout << "ly  " << ly << std::endl;
    //std::vector<uint64_t> shapes = x.sizes();
    auto shapes = x.sizes();  // type IntArrayRef
    int batch_size, d; // d is the dimension of the bot mlp output vector
    if (shapes.size() == 1) {
      // if x is 1D
      batch_size = 1;
      d = shapes[0];
    } else if (shapes.size() == 2) {
      batch_size = shapes[0];
      d = shapes[1];
    } else {
      std::cout << "Assertion fail: shapes should contain 1 or 2 elements." << std::endl;
    }
    //std::cout << "batch size  " << batch_size << std::endl;
    //std::cout << "d  " << d << std::endl;
    
    // create a TensorList of x and ly
    // first create the corresponding vector of tensors and cast to TensorList
    //std::vector<torch::Tensor> temp_vec;
    //temp_vec.push_back(x);
    //std::vector<torch::Tensor> temp_vec_concat;
    //temp_vec.insert(temp_vec.end(), ly.begin(), ly.end());
    //std::cout << "after concating x and ly  " << temp_vec << std::endl;

    //torch::TensorList temp_list = torch::TensorList(temp_vec);
    //std::cout << "tensorlist " << temp_list << std::endl;

    //T = torch::cat(temp_list, 1); //dim = 1
    // first cat the ly tensors
    torch::Tensor temp_cat;
    // temp_cat is a 1D vector
    temp_cat = torch::cat(ly, 1); //dim = 0
    //std::cout << "temp cat  " << temp_cat << std::endl;
    //std::cout << "x  " << x << std::endl;
    // then cat the result with x
    torch::Tensor T;
    T = torch::cat({x, temp_cat}, 1); 
    //std::cout << "T cat  " << T << std::endl;
    T = T.view({batch_size, -1, d}); 
    //std::cout << "T " << T << std::endl;
    
    torch::Tensor Z;
    Z = torch::bmm(T, torch::transpose(T, 1, 2));
    //std::cout << "Z " << Z << std::endl;
    
    
    auto Zshapes = Z.sizes();
    int ni = Zshapes[1];
    int nj = Zshapes[2];

    // Corresponding python code: 
    // li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
    // lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
    std::vector<long> li_tmp, lj_tmp;
    for ( int i  = 0; i < ni; i++) {
      for ( int j  = 0; j < i; j++) {
        li_tmp.push_back(i);
        lj_tmp.push_back(j);
      }
    }
    //std::cout << "li vec " << li_tmp << std::endl;
    auto opts = torch::TensorOptions().dtype(torch::kInt64);
    torch::Tensor li = torch::from_blob(li_tmp.data(), {li_tmp.size()}, opts).clone();
    torch::Tensor lj = torch::from_blob(lj_tmp.data(), {lj_tmp.size()}, opts).clone();
    //std::cout << "li tensor " << li << std::endl;
    //std::cout << "lj tensor " << lj << std::endl;

    // tensor indexing guide: https://pytorch.org/cppdocs/notes/tensor_indexing.html
    torch::Tensor Zflat = Z.index({torch::indexing::Slice(), li, lj});
    //std::cout << "zflat " << Zflat << std::endl;
    
    torch::Tensor R = torch::cat({x, Zflat}, 1);
    //std::cout << "R " << R << std::endl;
    return R;
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
      //std::cout << "index: " << sparse_index_group_batch << std::endl;
      //std::cout << "offset: " << sparse_offset_group_batch << std::endl;

      torch::nn::EmbeddingBag E = emb_l[i];
      torch::Tensor V = E->forward(sparse_index_group_batch, sparse_offset_group_batch);
      
      //std::cout << "embedding result " << V << std::endl;
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
    EE.get()->weight.data() = torch::tensor({{-0.46095, -0.33017},
 { 0.37814, -0.40165},
 {-0.07889,  0.45789},
 { 0.03317,  0.19188}});


    emb_l.push_back(EE);

    i = 1;
    n = ln[i];
    torch::nn::EmbeddingBag EE1(torch::nn::EmbeddingBagOptions(n,m).sparse(true).mode(torch::kSum));

    EE1.get()->weight.data() = torch::tensor(
{{-0.21302,  0.21535},
 { 0.38639, -0.55623},
 { 0.28884,  0.56449}}
    );
    emb_l.push_back(EE1);

    i = 2;
    n = ln[i];
    torch::nn::EmbeddingBag EE2(torch::nn::EmbeddingBagOptions(n,m).sparse(true).mode(torch::kSum));
    EE2.get()->weight.data() = torch::tensor(
    {{ 0.35096, -0.3105 },
     { 0.4091 , -0.56112}}
    );
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
    LL.get()->weight.data() = torch::tensor(
      {{ 0.46686, -0.05954, -0.55486, -0.53959},
       {-0.56566,  0.3508 , -0.0334 , -0.92935},
       { 0.05514, -0.33229,  0.14738, -0.58299}}
    );
    LL.get()->bias.data() = torch::tensor( {-0.35218,  0.17691,  0.97678} );
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



    LL2.get()->weight.data() = torch::tensor(
      {{-0.47305, -0.36733, -0.07005},
       { 1.29149,  0.28304,  0.43221}}
    );
    LL2.get()->bias.data() = torch::tensor({0.01618, 0.60616});
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
  //torch::Tensor dense = torch::tensor({{0.6965, 0.2861, 0.2269, 0.5513},
  //                                     {0.7195, 0.4231, 0.9808, 0.6848}});
  torch::Tensor dense = torch::tensor(
    {{4.1702e-01, 7.2032e-01, 1.1437e-04, 3.0233e-01},
     {1.4676e-01, 9.2339e-02, 1.8626e-01, 3.4556e-01}}
  );
  torch::Tensor offset = torch::tensor({{0, 2}, {0, 1}, {0, 2}});
  std::vector<torch::Tensor> indice = {torch::tensor({1,2,0,1,3}), torch::tensor({1, 0}), torch::tensor({0,1,1})};
  dlrm(dense, offset, indice);


}
