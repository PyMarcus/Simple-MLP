#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

// step function - a simple activation function to solve problem
int step(float x) {
    return x >= 0 ? 1 : 0;
}

int train() {
    float X[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}}; // and operation
    int Y[4] = {0, 0, 0, 1}; // answers of and operation

    float w1 = 0.0, w2 = 0.0, bias = 0.0, learning_rate = 0.1;

    int epochs = 10;

    for (int epoch = 0; epoch < epochs; epoch++) {
        printf("Epoch %d/%d\n", epochs, epoch + 1);

        for (int i =0; i < 4; i++) {
            float x1 = X[i][0];
            float x2 = X[i][1];

            float z = w1 * x1 + w2 * x2 + bias; // neuron

            const int prediction = step(z);

            const int error = Y[i] - prediction;

            w1 += learning_rate * error * x1;
            w2 += learning_rate * w1 * x2;
            bias += learning_rate * error;

            printf(
                "x=(%.0f, %.0f) target=%d pred=%d error=%d\n",
                x1, x2, Y[i], prediction, error
            );
        }

        printf(
           "weights: w1=%.2f w2=%.2f bias=%.2f\n\n",
           w1, w2, bias
       );
    }

    printf("Final predictions:\n");

    for (int i =0; i < 4; i++) {
        float z = w1 * X[i][0] + w2 * X[i][1] + bias;
        int prediction = step(z);
        printf(
            "%d AND %d = %d\n",
            (int)X[i][0],
            (int)X[i][1],
            prediction
        );
    }

    return EXIT_SUCCESS;
}
