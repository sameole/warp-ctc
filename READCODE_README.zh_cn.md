![Baidu Logo](/doc/baidu-research-logo-small.png)

[In English](README.md)

# warp-ctc

###

'''
compute_ctc_loss函数是关键(src/ctc_entrypoint.cpp)

'''

'''
include\detail\cpu_ctc.h关键的函数部分
cost_and_grad函数是外面调用的关键点
在训练的时候调用
template<typename ProbT>
ctcStatus_t
CpuCTC<ProbT>::cost_and_grad(const ProbT* const activations,
                             ProbT *grads,
                             ProbT *costs,
                             const int* const flat_labels,
                             const int* const label_lengths,
                             const int* const input_lengths)

在前向时调用
template<typename ProbT>
ctcStatus_t CpuCTC<ProbT>::score_forward(const ProbT* const activations,
                                         ProbT* costs,
                                         const int* const flat_labels,
                                         const int* const label_lengths,
                                         const int* const input_lengths) 


'''
