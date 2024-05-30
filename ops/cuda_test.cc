#include <torch/torch.h>
#include <torch/script.h>

at::Tensor flash_gat(
    at::Tensor x, at::Tensor out, at::Tensor rotate,
    at::Tensor ce_0_w, at::Tensor ce_0_b,
    at::Tensor ce_1_w, at::Tensor ce_1_b,
    at::Tensor ce_3_w, at::Tensor ce_3_b,
    at::Tensor ce_4_w, at::Tensor ce_4_b,
    at::Tensor ce_6_w, at::Tensor ce_6_b,
    at::Tensor ce_7_w, at::Tensor ce_7_b,
    at::Tensor bos_mask, at::Tensor bos_token,
    at::Tensor edge_begins, at::Tensor edge_index,
    at::Tensor n1_w, at::Tensor n1_b,
    at::Tensor edge_attr,
    at::Tensor nbr_0_w, at::Tensor nbr_0_b,
    at::Tensor nbr_1_w, at::Tensor nbr_1_b,
    at::Tensor nbr_3_w, at::Tensor nbr_3_b,
    at::Tensor ea_0_w, at::Tensor ea_0_b,
    at::Tensor ea_1_w, at::Tensor ea_1_b,
    at::Tensor ea_3_w, at::Tensor ea_3_b,
    at::Tensor aggr_0_w, at::Tensor aggr_0_b,
    at::Tensor aggr_2_w, at::Tensor aggr_2_b,
    at::Tensor aggr_3_w, at::Tensor aggr_3_b,
    at::Tensor q_w, at::Tensor q_b,
    at::Tensor k_w, at::Tensor k_b,
    at::Tensor v_w, at::Tensor v_b,
    at::Tensor ih_w, at::Tensor ih_b,
    at::Tensor hh_w, at::Tensor hh_b,
    at::Tensor self_w, at::Tensor self_b,
    at::Tensor out_w, at::Tensor out_b,
    at::Tensor n2_w, at::Tensor n2_b,
    at::Tensor mlp_0_w, at::Tensor mlp_0_b,
    at::Tensor mlp_3_w, at::Tensor mlp_3_b);

int main(int argc, char** argv) {
    torch::jit::script::Module module = torch::jit::load("../../gat_test.pt");
    std::vector<torch::Tensor> inputs;
    for (const auto& param : module.named_parameters())
        inputs.push_back(param.value);
    at::Tensor my_out = torch::zeros_like(inputs[1]);
    my_out = flash_gat(inputs[0], my_out, inputs[2], inputs[3], inputs[4], inputs[5], inputs[6],
              inputs[7], inputs[8], inputs[9], inputs[10], inputs[11], inputs[12], inputs[13],
              inputs[14], inputs[15], inputs[16], inputs[17], inputs[18], inputs[19], inputs[20],
              inputs[21], inputs[22], inputs[23], inputs[24], inputs[25], inputs[26], inputs[27],
              inputs[28], inputs[29], inputs[30], inputs[31], inputs[32], inputs[33], inputs[34],
              inputs[35], inputs[36], inputs[37], inputs[38], inputs[39], inputs[40], inputs[41],
              inputs[42], inputs[43], inputs[44], inputs[45], inputs[46], inputs[47], inputs[48],
              inputs[49], inputs[50], inputs[51], inputs[52], inputs[53], inputs[54], inputs[55],
              inputs[56], inputs[57], inputs[58], inputs[59]);
    printf("finished\n");
    std::cout << my_out.sizes() << std::endl;
    for (int i = 0; i < 4 * 3; i++) {
        if (torch::isnan(my_out[i]).any().item<bool>()) {
            printf("nan detected on node %d.\n", i);
            at::Tensor my_out_cpu = my_out[i].to(at::kCPU);
            at::Tensor inputs_cpu = inputs[1][i].to(at::kCPU);
            std::cout << my_out_cpu.sizes() << std::endl;
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++)
                    printf("%f vs. %f, ", my_out_cpu[i*8+j].item<float>(), inputs_cpu[i*8+j].item<float>());
                printf("\n");
            }
        }
    }
    return 0;
}
