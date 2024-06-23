#include "layer.h"

// Constructor
Layer::Layer(int M, int N, int O)
{
    this->M = M;
    this->N = N;
    this->O = O;

    float h_bias[N];
    float h_weight[N][M];

    output = new float[O];
    preact = new float[O];
    bias   = new float[N];
    weight = new float[M * N];

    for (int i = 0; i < N; ++i) {
        h_bias[i] = 0.5f - float(rand()) / float(RAND_MAX);
        for (int j = 0; j < M; ++j) {
            h_weight[i][j] = 0.5f - float(rand()) / float(RAND_MAX);
        }
    }

    std::memcpy(bias, h_bias, sizeof(float) * N);
    std::memcpy(weight, h_weight, sizeof(float) * M * N);

    d_output = new float[O];
    d_preact = new float[O];
    d_weight = new float[M * N];
}

// Destructor
Layer::~Layer()
{
    delete[] output;
    delete[] preact;
    delete[] bias;
    delete[] weight;
    delete[] d_output;
    delete[] d_preact;
    delete[] d_weight;
}

// Send data one row from dataset to the CPU
void Layer::setOutput(float *data)
{
    std::memcpy(output, data, sizeof(float) * O);
}

// Reset CPU memory between iterations
void Layer::clear()
{
    std::memset(output, 0x00, sizeof(float) * O);
    std::memset(preact, 0x00, sizeof(float) * O);
}

void Layer::bp_clear()
{
    std::memset(d_output, 0x00, sizeof(float) * O);
    std::memset(d_preact, 0x00, sizeof(float) * O);
    std::memset(d_weight, 0x00, sizeof(float) * M * N);
}

float step_function(float v)
{
    return 1 / (1 + exp(-v));
}

void apply_step_function(float *input, float *output, const int N)
{
    for (int idx = 0; idx < N; ++idx) {
        output[idx] = step_function(input[idx]);
    }
}

void makeError(float *err, float *output, unsigned int Y, const int N)
{
    for (int idx = 0; idx < N; ++idx) {
        err[idx] = ((Y == idx ? 1.0f : 0.0f) - output[idx]);
    }
}

void apply_grad(float *output, float *grad, const int N)
{
    for (int idx = 0; idx < N; ++idx) {
        output[idx] += dt * grad[idx];
    }
}

void fp_preact_c1(float input[28][28], float preact[6][24][24], float weight[6][5][5])
{
    for (int i3 = 0; i3 < 6; ++i3) {
        for (int i4 = 0; i4 < 24; ++i4) {
            for (int i5 = 0; i5 < 24; ++i5) {
                preact[i3][i4][i5] = 0;
                for (int i1 = 0; i1 < 5; ++i1) {
                    for (int i2 = 0; i2 < 5; ++i2) {
                        preact[i3][i4][i5] += weight[i3][i1][i2] * input[i4 + i1][i5 + i2];
                    }
                }
            }
        }
    }
}

void fp_bias_c1(float preact[6][24][24], float bias[6])
{
    for (int i1 = 0; i1 < 6; ++i1) {
        for (int i2 = 0; i2 < 24; ++i2) {
            for (int i3 = 0; i3 < 24; ++i3) {
                preact[i1][i2][i3] += bias[i1];
            }
        }
    }
}

void fp_preact_s1(float input[6][24][24], float preact[6][6][6], float weight[1][4][4])
{
    for (int i3 = 0; i3 < 6; ++i3) {
        for (int i4 = 0; i4 < 6; ++i4) {
            for (int i5 = 0; i5 < 6; ++i5) {
                preact[i3][i4][i5] = 0;
                for (int i1 = 0; i1 < 4; ++i1) {
                    for (int i2 = 0; i2 < 4; ++i2) {
                        preact[i3][i4][i5] += weight[0][i1][i2] * input[i3][i4 * 4 + i1][i5 * 4 + i2];
                    }
                }
            }
        }
    }
}

void fp_bias_s1(float preact[6][6][6], float bias[1])
{
    for (int i1 = 0; i1 < 6; ++i1) {
        for (int i2 = 0; i2 < 6; ++i2) {
            for (int i3 = 0; i3 < 6; ++i3) {
                preact[i1][i2][i3] += bias[0];
            }
        }
    }
}

void fp_preact_f(float input[6][6][6], float preact[10], float weight[10][6][6][6])
{
    for (int i1 = 0; i1 < 10; ++i1) {
        preact[i1] = 0;
        for (int i2 = 0; i2 < 6; ++i2) {
            for (int i3 = 0; i3 < 6; ++i3) {
                for (int i4 = 0; i4 < 6; ++i4) {
                    preact[i1] += weight[i1][i2][i3][i4] * input[i2][i3][i4];
                }
            }
        }
    }
}

void fp_bias_f(float preact[10], float bias[10])
{
    for (int idx = 0; idx < 10; ++idx) {
        preact[idx] += bias[idx];
    }
}

void bp_weight_f(float d_weight[10][6][6][6], float d_preact[10], float p_output[6][6][6])
{
    for (int i1 = 0; i1 < 10; ++i1) {
        for (int i2 = 0; i2 < 6; ++i2) {
            for (int i3 = 0; i3 < 6; ++i3) {
                for (int i4 = 0; i4 < 6; ++i4) {
                    d_weight[i1][i2][i3][i4] = d_preact[i1] * p_output[i2][i3][i4];
                }
            }
        }
    }
}

void bp_bias_f(float bias[10], float d_preact[10])
{
    for (int idx = 0; idx < 10; ++idx) {
        bias[idx] += dt * d_preact[idx];
    }
}

void bp_output_s1(float d_output[6][6][6], float n_weight[10][6][6][6], float nd_preact[10])
{
    //std::memset(d_output, 0, sizeof(float) * 6 * 6 * 6);
    for (int i1 = 0; i1 < 10; ++i1) {
        for (int i2 = 0; i2 < 6; ++i2) {
            for (int i3 = 0; i3 < 6; ++i3) {
                for (int i4 = 0; i4 < 6; ++i4) {
                    d_output[i2][i3][i4] += n_weight[i1][i2][i3][i4] * nd_preact[i1];
                }
            }
        }
    }
}

void bp_preact_s1(float d_preact[6][6][6], float d_output[6][6][6], float preact[6][6][6])
{
    for (int i1 = 0; i1 < 6; ++i1) {
        for (int i2 = 0; i2 < 6; ++i2) {
            for (int i3 = 0; i3 < 6; ++i3) {
                const float o = step_function(preact[i1][i2][i3]);
                d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
            }
        }
    }
}

void bp_weight_s1(float d_weight[1][4][4], float d_preact[6][6][6], float p_output[6][24][24])
{
    //std::memset(d_weight, 0, sizeof(float) * 1 * 4 * 4);
    for (int i4 = 0; i4 < 6; ++i4) {
        for (int i5 = 0; i5 < 6; ++i5) {
            for (int i6 = 0; i6 < 6; ++i6) {
                for (int i1 = 0; i1 < 4; ++i1) {
                    for (int i2 = 0; i2 < 4; ++i2) {
                        d_weight[0][i1][i2] += d_preact[i4][i5][i6] * p_output[i4][i5 * 4 + i1][i6 * 4 + i2];
                    }
                }
            }
        }
    }
}

void bp_bias_s1(float bias[1], float d_preact[6][6][6])
{
    for (int i1 = 0; i1 < 6; ++i1) {
        for (int i2 = 0; i2 < 6; ++i2) {
            for (int i3 = 0; i3 < 6; ++i3) {
                bias[0] += dt * d_preact[i1][i2][i3] / (6 * 6 * 6);
            }
        }
    }
}

void bp_output_c1(float d_output[6][24][24], float n_weight[1][4][4], float nd_preact[6][6][6])
{
    //std::memset(d_output, 0, sizeof(float) * 6 * 24 * 24);
    for (int i4 = 0; i4 < 6; ++i4) {
        for (int i5 = 0; i5 < 6; ++i5) {
            for (int i1 = 0; i1 < 4; ++i1) {
                for (int i2 = 0; i2 < 4; ++i2) {
                    for (int i3 = 0; i3 < 6; ++i3) {
                        d_output[i4][i5 * 4 + i1][i3 * 4 + i2] += n_weight[0][i1][i2] * nd_preact[i4][i5][i3];
                    }
                }
            }
        }
    }
}



void bp_preact_c1(float d_preact[6][24][24], float d_output[6][24][24], float preact[6][24][24])
{
    for (int i1 = 0; i1 < 6; ++i1) {
        for (int i2 = 0; i2 < 24; ++i2) {
            for (int i3 = 0; i3 < 24; ++i3) {
                const float o = step_function(preact[i1][i2][i3]);
                d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
            }
        }
    }
}

void bp_weight_c1(float d_weight[6][5][5], float d_preact[6][24][24], float p_output[28][28])
{
    //std::memset(d_weight, 0, sizeof(float) * 6 * 5 * 5);
    for (int i1 = 0; i1 < 6; ++i1) {
        for (int i2 = 0; i2 < 5; ++i2) {
            for (int i3 = 0; i3 < 5; ++i3) {
                for (int i4 = 0; i4 < 24; ++i4) {
                    for (int i5 = 0; i5 < 24; ++i5) {
                        d_weight[i1][i2][i3] += d_preact[i1][i4][i5] * p_output[i4 + i2][i5 + i3] / (24 * 24);
                    }
                }
            }
        }
    }
}

void bp_bias_c1(float bias[6], float d_preact[6][24][24])
{
    for (int i1 = 0; i1 < 6; ++i1) {
        for (int i2 = 0; i2 < 24; ++i2) {
            for (int i3 = 0; i3 < 24; ++i3) {
                bias[i1] += dt * d_preact[i1][i2][i3] / (24 * 24);
            }
        }
    }
}