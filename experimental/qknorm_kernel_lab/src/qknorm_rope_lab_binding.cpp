#include <torch/extension.h>

namespace py = pybind11;

void fused_qk_norm_rope_improve(torch::Tensor& qkv, int64_t num_heads_q,
                                int64_t num_heads_k, int64_t num_heads_v,
                                int64_t head_dim, double eps,
                                torch::Tensor& q_weight,
                                torch::Tensor& k_weight,
                                torch::Tensor& cos_sin_cache, bool is_neox,
                                torch::Tensor& position_ids,
                                int64_t block_size);

void fused_qk_norm_rope_improve_2_token_heads(
    torch::Tensor& qkv, int64_t num_heads_q, int64_t num_heads_k,
    int64_t num_heads_v, int64_t head_dim, double eps, torch::Tensor& q_weight,
    torch::Tensor& k_weight, torch::Tensor& cos_sin_cache, bool is_neox,
    torch::Tensor& position_ids, int64_t block_size,
    int64_t token_heads_per_warp);

void fused_qk_norm_rope_improve_force_hpw(
    torch::Tensor& qkv, int64_t num_heads_q, int64_t num_heads_k,
    int64_t num_heads_v, int64_t head_dim, double eps, torch::Tensor& q_weight,
    torch::Tensor& k_weight, torch::Tensor& cos_sin_cache, bool is_neox,
    torch::Tensor& position_ids, int64_t block_size,
    int64_t token_heads_per_warp);

void fused_qk_norm_rope_no_cossin_async(
    torch::Tensor& qkv, int64_t num_heads_q, int64_t num_heads_k,
    int64_t num_heads_v, int64_t head_dim, double eps, torch::Tensor& q_weight,
    torch::Tensor& k_weight, torch::Tensor& cos_sin_cache, bool is_neox,
    torch::Tensor& position_ids, int64_t block_size,
    int64_t token_heads_per_warp);

PYBIND11_MODULE(qknorm_rope_lab, m) {
  m.doc() = "Isolated qk_norm_rope kernel optimization lab";

  m.def("fused_qk_norm_rope_improve", &fused_qk_norm_rope_improve,
        py::arg("qkv"), py::arg("num_heads_q"), py::arg("num_heads_k"),
        py::arg("num_heads_v"), py::arg("head_dim"), py::arg("eps"),
        py::arg("q_weight"), py::arg("k_weight"), py::arg("cos_sin_cache"),
        py::arg("is_neox"), py::arg("position_ids"),
        py::arg("block_size") = 256);

  m.def("fused_qk_norm_rope_improve_2_token_heads",
        &fused_qk_norm_rope_improve_2_token_heads, py::arg("qkv"),
        py::arg("num_heads_q"), py::arg("num_heads_k"), py::arg("num_heads_v"),
        py::arg("head_dim"), py::arg("eps"), py::arg("q_weight"),
        py::arg("k_weight"), py::arg("cos_sin_cache"), py::arg("is_neox"),
        py::arg("position_ids"), py::arg("block_size") = 256,
        py::arg("token_heads_per_warp") = 2);

  m.def("fused_qk_norm_rope_improve_force_hpw",
        &fused_qk_norm_rope_improve_force_hpw, py::arg("qkv"),
        py::arg("num_heads_q"), py::arg("num_heads_k"), py::arg("num_heads_v"),
        py::arg("head_dim"), py::arg("eps"), py::arg("q_weight"),
        py::arg("k_weight"), py::arg("cos_sin_cache"), py::arg("is_neox"),
        py::arg("position_ids"), py::arg("block_size") = 256,
        py::arg("token_heads_per_warp") = 1);

  m.def("fused_qk_norm_rope_no_cossin_async",
        &fused_qk_norm_rope_no_cossin_async, py::arg("qkv"),
        py::arg("num_heads_q"), py::arg("num_heads_k"), py::arg("num_heads_v"),
        py::arg("head_dim"), py::arg("eps"), py::arg("q_weight"),
        py::arg("k_weight"), py::arg("cos_sin_cache"), py::arg("is_neox"),
        py::arg("position_ids"), py::arg("block_size") = 256,
        py::arg("token_heads_per_warp") = 4);
}
