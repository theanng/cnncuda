#include <vector>
#include <memory>
#include <cmath>
#include <cstdlib>
#include <cstring>

#ifndef LAYER_H
#define LAYER_H

const static float dt = 1.0E-01f;
const static float threshold = 1.0E-02f;

class Layer {
    public:
    int M, N, O;

    float *output;
    float *preact;

    float *bias;
    float *weight;

    float *d_output;
    float *d_preact;
    float *d_weight;

    Layer(int M, int N, int O);

    ~Layer();

    void setOutput(float *data);
    void clear();
    void bp_clear();
};

// Utility functions
float step_function(float v);
void apply_step_function(float *input, float *output, const int N);
void makeError(float *err, float *output, unsigned int Y, const int N);
void apply_grad(float *output, float *grad, const int N);

// Forward propagation functions
void fp_preact_c1(float input[28][28], float preact[6][24][24], float weight[6][5][5]);
void fp_bias_c1(float preact[6][24][24], float bias[6]);
void fp_preact_s1(float input[6][24][24], float preact[6][6][6], float weight[1][4][4]);
void fp_bias_s1(float preact[6][6][6], float bias[1]);
void fp_preact_f(float input[6][6][6], float preact[10], float weight[10][6][6][6]);
void fp_bias_f(float preact[10], float bias[10]);

// Back propagation functions
void bp_weight_f(float d_weight[10][6][6][6], float d_preact[10], float p_output[6][6][6]);
void bp_bias_f(float bias[10], float d_preact[10]);
void bp_output_s1(float d_output[6][6][6], float n_weight[10][6][6][6], float nd_preact[10]);
void bp_preact_s1(float d_preact[6][6][6], float d_output[6][6][6], float preact[6][6][6]);
void bp_weight_s1(float d_weight[1][4][4], float d_preact[6][6][6], float p_output[6][24][24]);
void bp_bias_s1(float bias[1], float d_preact[6][6][6]);
void bp_output_c1(float d_output[6][24][24], float n_weight[1][4][4], float nd_preact[6][6][6]);
void bp_preact_c1(float d_preact[6][24][24], float d_output[6][24][24], float preact[6][24][24]);
void bp_weight_c1(float d_weight[6][5][5], float d_preact[6][24][24], float p_output[28][28]);
void bp_bias_c1(float bias[6], float d_preact[6][24][24]);

#endif