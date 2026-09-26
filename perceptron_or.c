//
// Created by marcus on 9/26/26.
//
#include <stdio.h>

int step(float x) {
    return x >= 0 ? 1:0;
}

int main() {
    float X[4][2] = {
        {0, 0},
        {1, 0},
        {0, 1},
        {1, 1}
    };
    int Y[4] = {0, 1, 1, 1};

    int epochs = 100;
    int interactions = 4;
    float w1 = 0.0, w2 = 0.0, bias = 0.0;
    float learning_rate = 0.01;

    for (int epoch = 0; epoch < epochs; epoch++) {
        printf("epoch %d/%d\n", epoch, epochs);
        for (int i = 0; i < interactions; i++) {
            float x1 = X[i][0];
            float x2 = X[i][1];

            float z = X[i][0] * w1 + X[i][1] * w2 + bias;
            int prediction = step(z);

            int error = Y[i] - prediction;

            w1 += learning_rate * error * x1;
            w2 += learning_rate * w1 * x2;
            bias += learning_rate * error;
        }

        printf(
                   "weights: w1=%.2f w2=%.2f bias=%.2f\n\n",
                   w1, w2, bias
               );
    }

    printf("Truth table Or:\n");
    for (int i = 0; i < interactions; i++) {
        float z = X[i][0] * w1 + X[i][1] * w2 + bias;
        int prediction = step(z);
        printf("%d %d = %d\n", (int)X[i][0], (int)X[i][1], prediction);
    }
}
