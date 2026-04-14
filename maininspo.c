// Tell Microsoft's C runtime not to warn about older functions like fopen.
#define _CRT_SECURE_NO_WARNINGS

// Pull in basic type aliases, macros, and utility stuff.
// This probably defines types like u32, u64, f32, b32, b8, i64, true, false, MIN, MAX, GiB, MiB, etc.
#include "base.h"

// Pull in the memory arena interface.
// A memory arena is a big chunk of memory that we carve pieces out of, usually very fast.
#include "arena.h"

// Pull in the pseudo-random number generator interface.
#include "prng.h"

// Include the implementation file for the arena directly into this translation unit.
// That means this .c file will compile the arena code too.
#include "arena.c"

// Include the implementation file for the PRNG directly into this translation unit.
#include "prng.c"

// Define a matrix type.
// A matrix is just a 2D grid of numbers.
typedef struct {
    // Number of rows in the matrix.
    u32 rows, cols;

    // Data is stored in row-major order.
    // Row-major means: all of row 0, then all of row 1, then all of row 2, etc.
    f32* data;
} matrix;

// Declare a function that allocates a new matrix from an arena.
matrix* mat_create(mem_arena* arena, u32 rows, u32 cols);

// Declare a function that allocates a matrix and loads binary float data from a file into it.
matrix* mat_load(mem_arena* arena, u32 rows, u32 cols, const char* filename);

// Declare a function that copies one matrix into another.
// Returns a boolean-like value saying whether it worked.
b32 mat_copy(matrix* dst, matrix* src);

// Declare a function that sets all matrix elements to zero.
void mat_clear(matrix* mat);

// Declare a function that sets all matrix elements to the same number x.
void mat_fill(matrix* mat, f32 x);

// Declare a function that fills a matrix with random floats in [lower, upper).
void mat_fill_rand(matrix* mat, f32 lower, f32 upper);

// Declare a function that multiplies every matrix entry by a scale number.
void mat_scale(matrix* mat, f32 scale);

// Declare a function that sums all numbers in a matrix.
f32 mat_sum(matrix* mat);

// Declare a function that returns the index of the largest value in the matrix data array.
u64 mat_argmax(matrix* mat);

// Declare elementwise matrix addition: out = a + b.
b32 mat_add(matrix* out, const matrix* a, const matrix* b);

// Declare elementwise matrix subtraction: out = a - b.
b32 mat_sub(matrix* out, const matrix* a, const matrix* b);

// Declare matrix multiplication with options:
// - zero_out: clear out first or accumulate into it
// - transpose_a: treat a as transposed
// - transpose_b: treat b as transposed
b32 mat_mul(
    matrix* out, const matrix* a, const matrix* b,
    b8 zero_out, b8 transpose_a, b8 transpose_b
);

// Declare ReLU activation: out[i] = max(0, in[i]).
b32 mat_relu(matrix* out, const matrix* in);

// Declare softmax activation.
// Softmax turns a vector of raw scores into probabilities that sum to 1.
b32 mat_softmax(matrix* out, const matrix* in);

// Declare elementwise cross-entropy terms between p and q.
// Here p is usually the desired distribution and q is the predicted distribution.
b32 mat_cross_entropy(matrix* out, const matrix* p, const matrix* q);

// Declare backward-pass helper for ReLU.
// It adds gradient contributions into out based on in and grad.
b32 mat_relu_add_grad(matrix* out, const matrix* in, const matrix* grad);

// Declare backward-pass helper for softmax.
// It adds gradient contributions into out using softmax_out and incoming grad.
b32 mat_softmax_add_grad(
    matrix* out, const matrix* softmax_out, const matrix* grad
);

// Declare backward-pass helper for cross-entropy.
// It can compute gradient contributions for p and/or q.
b32 mat_cross_entropy_add_grad(
    matrix* p_grad, matrix* q_grad,
    const matrix* p, const matrix* q, const matrix* grad
);

// Define bit flags describing what a model variable is.
typedef enum {
    // No flags set.
    MV_FLAG_NONE = 0,

    // This variable should store and propagate gradients.
    MV_FLAG_REQUIRES_GRAD  = (1 << 0),

    // This variable is a trainable parameter, like weights or biases.
    MV_FLAG_PARAMETER      = (1 << 1),

    // This variable is the model input.
    MV_FLAG_INPUT          = (1 << 2),

    // This variable is the model output.
    MV_FLAG_OUTPUT         = (1 << 3),

    // This variable is the desired output / target / label.
    MV_FLAG_DESIRED_OUTPUT = (1 << 4),

    // This variable is the cost / loss node.
    MV_FLAG_COST           = (1 << 5),
} model_var_flags;

// Define the kinds of operations a model variable can represent.
typedef enum {
    // No operation.
    MV_OP_NULL = 0,

    // A directly created variable, not computed from inputs.
    MV_OP_CREATE,

    // Marker: unary ops start after this.
    _MV_OP_UNARY_START,

    // ReLU activation.
    MV_OP_RELU,

    // Softmax activation.
    MV_OP_SOFTMAX,

    // Marker: binary ops start after this.
    _MV_OP_BINARY_START,

    // Addition.
    MV_OP_ADD,

    // Subtraction.
    MV_OP_SUB,

    // Matrix multiplication.
    MV_OP_MATMUL,

    // Cross entropy.
    MV_OP_CROSS_ENTROPY,
} model_var_op;

// A node can have at most 2 input nodes in this system.
#define MODEL_VAR_MAX_INPUTS 2

// Figure out how many inputs an operation has.
// - before unary marker: 0 inputs
// - between unary and binary markers: 1 input
// - after binary marker: 2 inputs
#define MV_NUM_INPUTS(op) ((op) < _MV_OP_UNARY_START ? 0 : ((op) < _MV_OP_BINARY_START ? 1 : 2))

// A model variable is one node in the computation graph.
typedef struct model_var {
    // Unique index of this node inside the model.
    u32 index;

    // Bit flags describing the role of this node.
    u32 flags;

    // The actual forward value for this node.
    matrix* val;

    // The gradient of the loss with respect to this node's value.
    // Only allocated if the node requires gradients.
    matrix* grad;

    // What operation produced this node.
    model_var_op op;

    // Pointers to the input nodes for this operation.
    struct model_var* inputs[MODEL_VAR_MAX_INPUTS];
} model_var;

// A model program is an ordered list of nodes to compute.
// Think: "execution plan".
typedef struct {
    // Array of node pointers in execution order.
    model_var** vars;

    // Number of nodes in the program.
    u32 size;
} model_program;

// The main model object that stores all important graph entry points and compiled programs.
typedef struct {
    // Total number of variables created in this model.
    u32 num_vars;

    // Pointer to the input node.
    model_var* input;

    // Pointer to the output node.
    model_var* output;

    // Pointer to the target-label node.
    model_var* desired_output;

    // Pointer to the cost / loss node.
    model_var* cost;

    // Program for computing forward output.
    model_program forward_prog;

    // Program for computing loss and anything needed for training.
    model_program cost_prog;
} model_context;

// Training data and hyperparameters.
typedef struct {
    // Training input images.
    matrix* train_images;

    // Training labels, usually one-hot vectors.
    matrix* train_labels;

    // Test input images.
    matrix* test_images;

    // Test labels.
    matrix* test_labels;

    // Number of full passes over the training set.
    u32 epochs;

    // Number of examples per gradient update.
    u32 batch_size;

    // Step size for gradient descent.
    f32 learning_rate;
} model_training_desc;

// Declare a function to create a plain model variable.
model_var* mv_create(
    mem_arena* arena, model_context* model,
    u32 rows, u32 cols, u32 flags
);

// Declare a function to create a ReLU node.
model_var* mv_relu(
    mem_arena* arena, model_context* model,
    model_var* input, u32 flags
);

// Declare a function to create a softmax node.
model_var* mv_softmax(
    mem_arena* arena, model_context* model,
    model_var* input, u32 flags
);

// Declare a function to create an addition node.
model_var* mv_add(
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b, u32 flags
);

// Declare a function to create a subtraction node.
model_var* mv_sub(
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b, u32 flags
);

// Declare a function to create a matrix-multiply node.
model_var* mv_matmul(
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b, u32 flags
);

// Declare a function to create a cross-entropy node.
model_var* mv_cross_entropy(
    mem_arena* arena, model_context* model,
    model_var* p, model_var* q, u32 flags
);

// Declare a function that walks backward from an output node and builds an execution program.
model_program model_prog_create(
    mem_arena* arena, model_context* model, model_var* out_var
);

// Declare a function that executes forward computations for all nodes in a program.
void model_prog_compute(model_program* prog);

// Declare a function that executes backward gradient propagation for a program.
void model_prog_compute_grads(model_program* prog);

// Declare a function to create an empty model context.
model_context* model_create(mem_arena* arena);

// Declare a function to compile the model into executable forward and cost programs.
void model_compile(mem_arena* arena, model_context* model);

// Declare a function to run forward inference only.
void model_feedforward(model_context* model);

// Declare a function to train the model.
void model_train(
    model_context* model,
    const model_training_desc* training_desc
);

// Declare a helper that prints one MNIST digit using terminal colors.
void draw_mnist_digit(f32* data);

// Declare a helper that builds the specific MNIST neural network graph.
void create_mnist_model(mem_arena* arena, model_context* model);

// Program entry point.
int main(void) {
    // Create one permanent arena with 1 GiB total size and 1 MiB commit granularity.
    // This is the big memory pool for almost everything in this program.
    mem_arena* perm_arena = arena_create(GiB(1), MiB(1));

    // Load training images from file.
    // Shape: 60000 rows, 784 columns.
    // That means: 60000 examples, each example is 784 pixels.
    matrix* train_images = mat_load(perm_arena, 60000, 784, "train_images.mat");

    // Load test images from file.
    // Shape: 10000 examples, each with 784 pixels.
    matrix* test_images = mat_load(perm_arena, 10000, 784, "test_images.mat");

    // Create training-label matrix with shape 60000 x 10.
    // This will hold one-hot encoded labels.
    matrix* train_labels = mat_create(perm_arena, 60000, 10);

    // Create test-label matrix with shape 10000 x 10.
    matrix* test_labels = mat_create(perm_arena, 10000, 10);

    // Start a local scope just to keep temporary label-file variables limited to this block.
    {
        // Load raw training labels from file.
        // Shape is 60000 x 1, so each row is just one class number.
        matrix* train_labels_file = mat_load(perm_arena, 60000, 1, "train_labels.mat");

        // Load raw test labels from file.
        matrix* test_labels_file = mat_load(perm_arena, 10000, 1, "test_labels.mat");

        // IMPORTANT:
        // This code assumes train_labels starts as all zeros.
        // If mat_create / PUSH_ARRAY does NOT zero memory, then the non-target entries may contain garbage.
        // In that case you would need mat_clear(train_labels) before this loop.
        for (u32 i = 0; i < 60000; i++) {
            // Read the class number for training example i.
            // Even though the source storage is f32, this is being interpreted as an integer label 0..9.
            u32 num = train_labels_file->data[i];

            // Set the correct class slot to 1.0 to make a one-hot vector.
            // Index math:
            // row i starts at i * 10
            // then + num picks the correct class column
            train_labels->data[i * 10 + num] = 1.0f;
        }

        // Same logic for the test set.
        // Again, this assumes test_labels starts as zeros.
        for (u32 i = 0; i < 10000; i++) {
            // Read the raw class number for test example i.
            u32 num = test_labels_file->data[i];

            // Set that class position to 1.0 in the one-hot vector.
            test_labels->data[i * 10 + num] = 1.0f;
        }
    }

    // Draw the very first test image as a 28x28 grayscale block in the terminal.
    draw_mnist_digit(test_images->data);

    // Print the first test label vector so we can see which digit it is.
    for (u32 i = 0; i < 10; i++) {
        // Print each one-hot label entry as a rounded integer-like float.
        printf("%.0f ", test_labels->data[i]);
    }

    // Print a blank line after the label vector.
    printf("\n\n");

    // Create the model context object.
    model_context* model = model_create(perm_arena);

    // Build the MNIST neural network graph inside the model.
    create_mnist_model(perm_arena, model);

    // Compile execution programs from the graph.
    model_compile(perm_arena, model);

    // Copy the first test image into the model's input node.
    // sizeof(f32) * 784 bytes are copied.
    memcpy(model->input->val->data, test_images->data, sizeof(f32) * 784);

    // Run the forward pass once before training.
    model_feedforward(model);

    // Print header text for the pre-training output.
    printf("pre-training output: ");

    // Print the 10 output probabilities/logits after softmax.
    for (u32 i = 0; i < 10; i++) {
        // Print with 2 decimal places.
        printf("%.2f ", model->output->val->data[i]);
    }

    // End the line.
    printf("\n");

    // Build the training configuration object using designated initializers.
    model_training_desc training_desc = {
        // Pointer to training images.
        .train_images = train_images,

        // Pointer to training labels.
        .train_labels = train_labels,

        // Pointer to test images.
        .test_images = test_images,

        // Pointer to test labels.
        .test_labels = test_labels,

        // Train for 10 epochs.
        .epochs = 10,

        // Use mini-batches of 50 examples.
        .batch_size = 50,

        // Use learning rate 0.01.
        .learning_rate = 0.01f
    };

    // Train the model using the config above.
    model_train(model, &training_desc);
    
    // After training, copy the same first test image into the input again.
    memcpy(model->input->val->data, test_images->data, sizeof(f32) * 784);

    // Run another forward pass after training.
    model_feedforward(model);

    // Print header text for the post-training output.
    printf("post-training output: ");

    // Print the 10 output values after training.
    for (u32 i = 0; i < 10; i++) {
        // Print full float format.
        printf("%f ", model->output->val->data[i]);
    }

    // Print two newlines after the output.
    printf("\n\n");

    // Destroy the permanent arena and free all memory it owns.
    arena_destroy(perm_arena);

    // Return success from main.
    return 0;
}

// Draw one MNIST image stored as 784 floats.
void draw_mnist_digit(f32* data) {
    // Loop over each y row from 0 to 27.
    for (u32 y = 0; y < 28; y++) {
        // Loop over each x column from 0 to 27.
        for (u32 x = 0; x < 28; x++) {
            // Read pixel value at (x, y).
            // Row-major 2D to 1D index: x + y * 28.
            f32 num = data[x + y * 28];

            // Map pixel intensity to a terminal color code.
            // 232..255 is a grayscale ramp in 256-color ANSI terminals.
            u32 col = 232 + (u32)(num * 23);

            // Print two colored spaces using that background color.
            printf("\x1b[48;5;%dm  ", col);
        }

        // Move to next terminal line after one row is printed.
        printf("\n");
    }

    // Reset terminal formatting so later text is normal again.
    printf("\x1b[0m");
}

// Build the neural network graph for MNIST classification.
void create_mnist_model(mem_arena* arena, model_context* model) {
    // Create the input node.
    // Shape is 784 x 1, meaning one column vector with 784 entries.
    model_var* input = mv_create(arena, model, 784, 1, MV_FLAG_INPUT);

    // Create first weight matrix W0.
    // Shape 16 x 784 means it maps 784 inputs to 16 hidden units.
    model_var* W0 = mv_create(arena, model, 16, 784, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

    // Create second weight matrix W1.
    // Shape 16 x 16 means hidden layer to hidden layer.
    model_var* W1 = mv_create(arena, model, 16, 16, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

    // Create third weight matrix W2.
    // Shape 10 x 16 means hidden layer to 10 output classes.
    model_var* W2 = mv_create(arena, model, 10, 16, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

    // Compute Xavier/Glorot-style initialization bound for W0.
    // Formula sqrt(6 / (fan_in + fan_out)).
    f32 bound0 = sqrtf(6.0f / (784 + 16));

    // Compute Xavier/Glorot bound for W1.
    f32 bound1 = sqrtf(6.0f / (16 + 16));

    // Compute Xavier/Glorot bound for W2.
    f32 bound2 = sqrtf(6.0f / (16 + 10));

    // Fill W0 with random values in [-bound0, bound0].
    mat_fill_rand(W0->val, -bound0, bound0);

    // Fill W1 with random values in [-bound1, bound1].
    mat_fill_rand(W1->val, -bound1, bound1);

    // Fill W2 with random values in [-bound2, bound2].
    mat_fill_rand(W2->val, -bound2, bound2);

    // Create bias vector b0 for the first hidden layer.
    // Shape 16 x 1.
    // IMPORTANT: this relies on new memory being zeroed, because there is no explicit initialization here.
    model_var* b0 = mv_create(arena, model, 16, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

    // Create bias vector b1 for the second hidden layer.
    // Same zero-init caveat.
    model_var* b1 = mv_create(arena, model, 16, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

    // Create bias vector b2 for the output layer.
    // Same zero-init caveat.
    model_var* b2 = mv_create(arena, model, 10, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

    // Compute z0_a = W0 * input.
    model_var* z0_a = mv_matmul(arena, model, W0, input, 0);

    // Compute z0_b = z0_a + b0.
    model_var* z0_b = mv_add(arena, model, z0_a, b0, 0);

    // Compute a0 = ReLU(z0_b).
    model_var* a0 = mv_relu(arena, model, z0_b, 0);

    // Compute z1_a = W1 * a0.
    model_var* z1_a = mv_matmul(arena, model, W1, a0, 0);

    // Compute z1_b = z1_a + b1.
    model_var* z1_b = mv_add(arena, model, z1_a, b1, 0);

    // Compute z1_c = ReLU(z1_b).
    model_var* z1_c = mv_relu(arena, model, z1_b, 0);

    // Compute a1 = a0 + z1_c.
    // This is a residual / skip-style connection.
    model_var* a1 = mv_add(arena, model, a0, z1_c, 0);

    // Compute z2_a = W2 * a1.
    model_var* z2_a = mv_matmul(arena, model, W2, a1, 0);

    // Compute z2_b = z2_a + b2.
    model_var* z2_b = mv_add(arena, model, z2_a, b2, 0);

    // Compute output = softmax(z2_b).
    // Mark this node as the model output.
    model_var* output = mv_softmax(arena, model, z2_b, MV_FLAG_OUTPUT);

    // Create the desired-output / target vector node.
    // Shape 10 x 1.
    model_var* y = mv_create(arena, model, 10, 1, MV_FLAG_DESIRED_OUTPUT);

    // Compute elementwise cross-entropy between target y and predicted output.
    // Mark this node as the cost node.
    model_var* cost = mv_cross_entropy(arena, model, y, output, MV_FLAG_COST);

    // Silence "unused variable" warnings if the compiler warns about local names not referenced later.
    // In plain C this is optional and not present in the original code, so we keep behavior unchanged by doing nothing.
    // The local pointers exist only to construct the graph.
}

// Allocate a matrix object and its data from the arena.
matrix* mat_create(mem_arena* arena, u32 rows, u32 cols) {
    // Allocate space for the matrix struct itself.
    matrix* mat = PUSH_STRUCT(arena, matrix);

    // Store number of rows.
    mat->rows = rows;

    // Store number of columns.
    mat->cols = cols;

    // Allocate enough float slots for rows * cols elements.
    mat->data = PUSH_ARRAY(arena, f32, (u64)rows * cols);

    // Return the new matrix.
    return mat;
}

// Allocate a matrix and load raw bytes from a file into its data buffer.
matrix* mat_load(mem_arena* arena, u32 rows, u32 cols, const char* filename) {
    // First create the destination matrix.
    matrix* mat = mat_create(arena, rows, cols);

    // Open the file in binary read mode.
    FILE* f = fopen(filename, "rb");

    // Move file cursor to end so ftell can measure total file size.
    fseek(f, 0, SEEK_END);

    // Read current position, which is now the file size in bytes.
    u64 size = ftell(f);

    // Move file cursor back to the start so we can read the file from the beginning.
    fseek(f, 0, SEEK_SET);

    // Clamp the number of bytes we will read so we do not overflow the matrix buffer.
    size = MIN(size, sizeof(f32) * rows * cols);

    // Read size bytes from the file into mat->data.
    // Because mat->data is float storage, this assumes the file layout matches raw f32 bytes.
    fread(mat->data, 1, size, f);

    // Close the file handle.
    fclose(f);

    // Return the loaded matrix.
    return mat;
}

// Copy all data from src to dst if they have the same shape.
b32 mat_copy(matrix* dst, matrix* src) {
    // If shape does not match, copying element-for-element is invalid.
    if (dst->rows != src->rows || dst->cols != src->cols) {
        // Report failure.
        return false;
    }

    // Copy all float bytes from src data into dst data.
    memcpy(dst->data, src->data, sizeof(f32) * (u64)dst->rows * dst->cols);

    // Report success.
    return true;
}

// Set all matrix entries to zero bytes.
void mat_clear(matrix* mat) {
    // Zero out the whole data buffer.
    memset(mat->data, 0, sizeof(f32) * (u64)mat->rows * mat->cols);
}

// Fill every element with the same float x.
void mat_fill(matrix* mat, f32 x) {
    // Compute total element count.
    u64 size = (u64)mat->rows * mat->cols;

    // Visit every element.
    for (u64 i = 0; i < size; i++) {
        // Write x into this slot.
        mat->data[i] = x;
    }
}

// Fill every element with a random float in [lower, upper).
void mat_fill_rand(matrix* mat, f32 lower, f32 upper) {
    // Compute total element count.
    u64 size = (u64)mat->rows * mat->cols;

    // Visit every element.
    for (u64 i = 0; i < size; i++) {
        // prng_randf() is expected to give a float in [0, 1).
        // Multiply by range width, then shift by lower.
        mat->data[i] = prng_randf() * (upper - lower) + lower;
    }

}

// Multiply every element by a scalar.
void mat_scale(matrix* mat, f32 scale) {
    // Compute total element count.
    u64 size = (u64)mat->rows * mat->cols;

    // Visit every element.
    for (u64 i = 0; i < size; i++) {
        // Scale this element in place.
        mat->data[i] *= scale;
    }
}

// Add up all elements in the matrix and return the total.
f32 mat_sum(matrix* mat) {
    // Compute total element count.
    u64 size = (u64)mat->rows * mat->cols;

    // Start the running sum at zero.
    f32 sum = 0.0f;

    // Visit every element.
    for (u64 i = 0; i < size; i++) {
        // Add this element to the running total.
        sum += mat->data[i];
    }

    // Return the finished sum.
    return sum;
}

// Return the flat index of the largest matrix element.
u64 mat_argmax(matrix* mat) {
    // Compute total element count.
    u64 size = (u64)mat->rows * mat->cols;

    // Assume the first element is the maximum until proven otherwise.
    u64 max_i = 0;

    // Visit every element.
    for (u64 i = 0; i < size; i++) {
        // If this element is bigger than the current best, remember its index.
        if (mat->data[i] > mat->data[max_i]) {
            max_i = i;
        }
    }

    // Return the winning index.
    return max_i;
}

// Elementwise add matrices a and b into out.
b32 mat_add(matrix* out, const matrix* a, const matrix* b) {
    // Shapes of a and b must match.
    if (a->rows != b->rows || a->cols != b->cols) {
        return false;
    }

    // out must have the same shape too.
    if (out->rows != a->rows || out->cols != a->cols) {
        return false;
    }

    // Compute total element count.
    u64 size = (u64)out->rows * out->cols;

    // Visit every element.
    for (u64 i = 0; i < size; i++) {
        // Add corresponding entries.
        out->data[i] = a->data[i] + b->data[i];
    }

    // IMPORTANT:
    // This returns false even though the operation succeeded.
    // That is almost certainly a bug in the original code.
    return false;
}

// Elementwise subtract matrix b from a into out.
b32 mat_sub(matrix* out, const matrix* a, const matrix* b) {
    // Shapes of a and b must match.
    if (a->rows != b->rows || a->cols != b->cols) {
        return false;
    }

    // out must have the same shape too.
    if (out->rows != a->rows || out->cols != a->cols) {
        return false;
    }

    // Compute total element count.
    u64 size = (u64)out->rows * out->cols;

    // Visit every element.
    for (u64 i = 0; i < size; i++) {
        // Subtract corresponding entries.
        out->data[i] = a->data[i] - b->data[i];
    }

    // IMPORTANT:
    // This also returns false even on success.
    // That is almost certainly a bug in the original code.
    return false;
}

// Multiply matrices with no transposes: out += a * b.
void _mat_mul_nn(matrix* out, const matrix* a, const matrix* b) {
    // Loop over output rows i.
    for (u64 i = 0; i < out->rows; i++) {
        // Loop over shared dimension k.
        for (u64 k = 0; k < a->cols; k++) {
            // Loop over output columns j.
            for (u64 j = 0; j < out->cols; j++) {
                // Accumulate one multiply-add into out(i, j).
                // Flat index formula for row-major: col + row * row_width.
                out->data[j + i * out->cols] += 
                    a->data[k + i * a->cols] * 
                    b->data[j + k * b->cols];
            }
        }
    }
}

// Multiply where a is normal and b is treated as transposed: out += a * b^T.
void _mat_mul_nt(matrix* out, const matrix* a, const matrix* b) {
    // Loop over output rows i.
    for (u64 i = 0; i < out->rows; i++) {
        // Loop over output columns j.
        for (u64 j = 0; j < out->cols; j++) {
            // Loop over shared dimension k.
            for (u64 k = 0; k < a->cols; k++) {
                // Accumulate one multiply-add into out(i, j).
                out->data[j + i * out->cols] += 
                    a->data[k + i * a->cols] * 
                    b->data[k + j * b->cols];
            }
        }
    }
}

// Multiply where a is treated as transposed and b is normal: out += a^T * b.
void _mat_mul_tn(matrix* out, const matrix* a, const matrix* b) {
    // Loop over shared dimension k as rows of original a.
    for (u64 k = 0; k < a->rows; k++) {
        // Loop over output rows i.
        for (u64 i = 0; i < out->rows; i++) {
            // Loop over output columns j.
            for (u64 j = 0; j < out->cols; j++) {
                // Accumulate one multiply-add into out(i, j).
                out->data[j + i * out->cols] += 
                    a->data[i + k * a->cols] * 
                    b->data[j + k * b->cols];
            }
        }
    }
}

// Multiply where both a and b are treated as transposed: out += a^T * b^T.
void _mat_mul_tt(matrix* out, const matrix* a, const matrix* b) {
    // Loop over output rows i.
    for (u64 i = 0; i < out->rows; i++) {
        // Loop over output columns j.
        for (u64 j = 0; j < out->cols; j++) {
            // Loop over shared dimension k.
            for (u64 k = 0; k < a->rows; k++) {
                // Accumulate one multiply-add into out(i, j).
                out->data[j + i * out->cols] += 
                    a->data[i + k * a->cols] * 
                    b->data[k + j * b->cols];
            }
        }
    }
}

// Generic matrix multiplication wrapper with transpose and zeroing options.
b32 mat_mul(
    matrix* out, const matrix* a, const matrix* b,
    b8 zero_out, b8 transpose_a, b8 transpose_b
) {
    // Effective rows of a after optional transpose.
    u32 a_rows = transpose_a ? a->cols : a->rows;

    // Effective columns of a after optional transpose.
    u32 a_cols = transpose_a ? a->rows : a->cols;

    // Effective rows of b after optional transpose.
    u32 b_rows = transpose_b ? b->cols : b->rows;

    // Effective columns of b after optional transpose.
    u32 b_cols = transpose_b ? b->rows : b->cols;

    // Matrix multiplication requires inner dimensions to match.
    if (a_cols != b_rows) { return false; }

    // Output matrix must match resulting shape.
    if (out->rows != a_rows || out->cols != b_cols) { return false; }

    // If requested, clear output first so the result starts from zero.
    if (zero_out) {
        mat_clear(out);
    }

    // Pack the two transpose flags into one 2-bit number:
    // bit 1 = transpose_a
    // bit 0 = transpose_b
    u32 transpose = (transpose_a << 1) | transpose_b;

    // Pick the correct specialized multiplication kernel.
    switch (transpose) {
        // Neither input transposed.
        case 0b00: { _mat_mul_nn(out, a, b); } break;

        // Only b transposed.
        case 0b01: { _mat_mul_nt(out, a, b); } break;

        // Only a transposed.
        case 0b10: { _mat_mul_tn(out, a, b); } break;

        // Both transposed.
        case 0b11: { _mat_mul_tt(out, a, b); } break;
    }

    // Report success.
    return true;
}

// Apply ReLU elementwise.
b32 mat_relu(matrix* out, const matrix* in) {
    // Shapes must match.
    if (out->rows != in->rows || out->cols != in->cols) {
        return false;
    }

    // Compute total element count.
    u64 size = (u64)out->rows * out->cols;

    // Visit every element.
    for (u64 i = 0; i < size; i++) {
        // ReLU keeps positive values and clamps negative ones to zero.
        out->data[i] = MAX(0, in->data[i]);
    }

    // Report success.
    return true;
}

// Apply softmax to all entries in the input.
b32 mat_softmax(matrix* out, const matrix* in) {
    // Shapes must match.
    if (out->rows != in->rows || out->cols != in->cols) {
        return false;
    }

    // Compute total element count.
    u64 size = (u64)out->rows * out->cols;

    // Start sum of exponentials at zero.
    f32 sum = 0.0f;

    // First pass: exponentiate each input and accumulate the sum.
    for (u64 i = 0; i < size; i++) {
        // Compute e^(input_i).
        // IMPORTANT:
        // This implementation does NOT subtract the max input first,
        // so it is numerically less stable and can overflow for large inputs.
        out->data[i] = expf(in->data[i]);

        // Add to total.
        sum += out->data[i];
    }

    // Second pass: divide all exponentials by the sum so they add to 1.
    mat_scale(out, 1.0f / sum);

    // Report success.
    return true;
}

// Compute elementwise cross-entropy terms out[i] = p[i] * -log(q[i]), except where p[i] is 0.
b32 mat_cross_entropy(matrix* out, const matrix* p, const matrix* q) {
    // p and q must have the same shape.
    if (p->rows != q->rows || p->cols != q->cols) { return false; }

    // out must have that same shape too.
    if (out->rows != p->rows || out->cols != p->cols) { return false; }

    // Compute total element count.
    u64 size = (u64)out->rows * out->cols;

    // Visit every element.
    for (u64 i = 0; i < size; i++) {
        // If p[i] is zero, its contribution is zero.
        // Otherwise compute p[i] * -log(q[i]).
        // IMPORTANT:
        // If q[i] is zero or extremely tiny, logf(q[i]) can blow up to -inf / large magnitude.
        out->data[i] = p->data[i] == 0.0f ?
            0.0f : p->data[i] * -logf(q->data[i]);
    }

    // Report success.
    return true;
}

// Add gradient contribution through a ReLU.
b32 mat_relu_add_grad(matrix* out, const matrix* in, const matrix* grad) {
    // out and in must match shape.
    if (out->rows != in->rows || out->cols != in->cols) {
        return false;
    }

    // out and grad must match shape.
    if (out->rows != grad->rows || out->cols != grad->cols) {
        return false;
    }

    // Compute total element count.
    u64 size = (u64)out->rows * out->cols;

    // Visit every element.
    for (u64 i = 0; i < size; i++) {
        // ReLU derivative is:
        // 1 if input > 0
        // 0 otherwise
        // So pass gradient through only where input was positive.
        out->data[i] += in->data[i] > 0.0f ? grad->data[i] : 0.0f;
    }

    // Report success.
    return true;
}

// Add gradient contribution through a softmax.
b32 mat_softmax_add_grad(
    matrix* out, const matrix* softmax_out, const matrix* grad
) {
    // Require softmax_out to be a vector shape:
    // either 1 x N or N x 1.
    // If both dimensions are not 1, then it is not a vector.
    if (softmax_out->rows != 1 && softmax_out->cols != 1) {
        return false;
    }

    // Get a temporary scratch arena for short-lived allocations.
    mem_arena_temp scratch = arena_scratch_get(NULL, 0);

    // Size of the vector is the larger of rows and cols.
    u32 size = MAX(softmax_out->rows, softmax_out->cols);

    // Allocate a Jacobian matrix of shape size x size.
    // The Jacobian stores all partial derivatives of the softmax output with respect to its inputs.
    matrix* jacobian = mat_create(scratch.arena, size, size);

    // Fill the Jacobian.
    for (u32 i = 0; i < size; i++) {
        for (u32 j = 0; j < size; j++) {
            // Softmax Jacobian entry:
            // s_i * (delta_ij - s_j)
            jacobian->data[j + i * size] =
                softmax_out->data[i] * ((i == j) - softmax_out->data[j]);
        }
    }

    // Multiply Jacobian by incoming gradient to get gradient wrt softmax input.
    // zero_out = 0 means "accumulate into out".
    mat_mul(out, jacobian, grad, 0, 0, 0);

    // Release temporary scratch allocations.
    arena_scratch_release(scratch);

    // Report success.
    return true;
}

// Add gradient contributions through the cross-entropy operation.
b32 mat_cross_entropy_add_grad(
    matrix* p_grad, matrix* q_grad,
    const matrix* p, const matrix* q, const matrix* grad
) {
    // p and q must have the same shape.
    if (p->rows != q->rows || p->cols != q->cols) { return false; }

    // Compute total number of elements.
    u64 size = (u64)p->rows * p->cols;

    // If caller wants gradient wrt p, compute it.
    if (p_grad != NULL) {
        // p_grad must have the same shape as p.
        if (p_grad->rows != p->rows || p_grad->cols != p->cols) {
            return false; 
        }

        // For each element:
        // d/dp [p * -log(q)] = -log(q)
        // then multiply by incoming gradient and accumulate.
        for (u64 i = 0; i < size; i++) {
            p_grad->data[i] += -logf(q->data[i]) * grad->data[i];
        }
    }

    // If caller wants gradient wrt q, compute it.
    if (q_grad != NULL) {
        // q_grad must have the same shape as q.
        if (q_grad->rows != q->rows || q_grad->cols != q->cols) {
            return false; 
        }

        // For each element:
        // d/dq [p * -log(q)] = -p / q
        // then multiply by incoming gradient and accumulate.
        for (u64 i = 0; i < size; i++) {
            q_grad->data[i] += -p->data[i] / q->data[i] * grad->data[i];
        }
    }

    // Report success.
    return true;
}

// Create a raw model variable node.
model_var* mv_create(
    mem_arena* arena, model_context* model,
    u32 rows, u32 cols, u32 flags
) {
    // Allocate the node struct.
    model_var* out = PUSH_STRUCT(arena, model_var);

    // Assign this node a unique index, then increment model->num_vars.
    out->index = model->num_vars++;

    // Store flag bits.
    out->flags = flags;

    // Mark this node as a directly created node by default.
    out->op = MV_OP_CREATE;

    // Allocate its forward value matrix.
    out->val = mat_create(arena, rows, cols);

    // If gradients are required, allocate a gradient matrix too.
    if (flags & MV_FLAG_REQUIRES_GRAD) {
        out->grad = mat_create(arena, rows, cols);
    }

    // If this node is marked as the input, remember it in the model.
    if (flags & MV_FLAG_INPUT) { model->input = out; }

    // If this node is marked as the output, remember it in the model.
    if (flags & MV_FLAG_OUTPUT) { model->output = out; }

    // If this node is marked as the desired output, remember it in the model.
    if (flags & MV_FLAG_DESIRED_OUTPUT) { model->desired_output = out; }

    // If this node is marked as the cost, remember it in the model.
    if (flags & MV_FLAG_COST) { model->cost = out; }

    // Return the new node.
    return out;
}

// Shared helper for building unary-operation nodes.
model_var* _mv_unary_impl(
    mem_arena* arena, model_context* model,
    model_var* input, u32 rows, u32 cols,
    u32 flags, model_var_op op
) {
    // If the input requires gradients, then the output should also require gradients.
    if (input->flags & MV_FLAG_REQUIRES_GRAD) {
        flags |= MV_FLAG_REQUIRES_GRAD;
    }

    // Create the output node.
    model_var* out = mv_create(arena, model, rows, cols, flags);

    // Record which operation this node performs.
    out->op = op;

    // Record the input node pointer.
    out->inputs[0] = input;

    // Return the new op node.
    return out;
}

// Shared helper for building binary-operation nodes.
model_var* _mv_binary_impl(
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b,
    u32 rows, u32 cols,
    u32 flags, model_var_op op
) {
    // If either input requires gradients, the output should require gradients too.
    if (
        (a->flags & MV_FLAG_REQUIRES_GRAD) ||
        (b->flags & MV_FLAG_REQUIRES_GRAD)
    ) {
        flags |= MV_FLAG_REQUIRES_GRAD;
    }

    // Create the output node.
    model_var* out = mv_create(arena, model, rows, cols, flags);

    // Record which operation produced this node.
    out->op = op;

    // Store left input pointer.
    out->inputs[0] = a;

    // Store right input pointer.
    out->inputs[1] = b;

    // Return the new op node.
    return out;
}

// Create a ReLU node.
model_var* mv_relu(
    mem_arena* arena, model_context* model,
    model_var* input, u32 flags
) {
    // Unary op output has the same shape as the input.
    return _mv_unary_impl(
        arena, model, input,
        input->val->rows, input->val->cols,
        flags, MV_OP_RELU
    );
}

// Create a softmax node.
model_var* mv_softmax(
    mem_arena* arena, model_context* model,
    model_var* input, u32 flags
) {
    // Softmax output has the same shape as the input.
    return _mv_unary_impl(
        arena, model, input,
        input->val->rows, input->val->cols,
        flags, MV_OP_SOFTMAX
    );
}

// Create an addition node.
model_var* mv_add(
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b, u32 flags
) {
    // Addition requires identical shapes.
    if (a->val->rows != b->val->rows || a->val->cols != b->val->cols) {
        return NULL;
    }

    // Output shape is the same as the input shapes.
    return _mv_binary_impl(
        arena, model, a, b,
        a->val->rows, a->val->cols,
        flags, MV_OP_ADD
    );
}

// Create a subtraction node.
model_var* mv_sub(
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b, u32 flags
) {
    // Subtraction requires identical shapes.
    if (a->val->rows != b->val->rows || a->val->cols != b->val->cols) {
        return NULL;
    }

    // Output shape is the same as the input shapes.
    return _mv_binary_impl(
        arena, model, a, b,
        a->val->rows, a->val->cols,
        flags, MV_OP_SUB
    );
}

// Create a matrix-multiply node.
model_var* mv_matmul(
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b, u32 flags
) {
    // Matrix multiplication requires a.cols == b.rows.
    if (a->val->cols != b->val->rows) {
        return NULL;
    }

    // Output shape is a.rows x b.cols.
    return _mv_binary_impl(
        arena, model, a, b,
        a->val->rows, b->val->cols,
        flags, MV_OP_MATMUL
    );
}

// Create a cross-entropy node.
model_var* mv_cross_entropy(
    mem_arena* arena, model_context* model,
    model_var* p, model_var* q, u32 flags
) {
    // Cross-entropy here is elementwise, so p and q must have the same shape.
    if (p->val->rows != q->val->rows || p->val->cols != q->val->cols) {
        return NULL;
    }

    // Output shape matches p and q.
    return _mv_binary_impl(
        arena, model, p, q,
        p->val->rows, p->val->cols,
        flags, MV_OP_CROSS_ENTROPY
    );
}

// Build an execution program by walking backward from an output variable.
model_program model_prog_create(
    mem_arena* arena, model_context* model, model_var* out_var
) {
    // Get a temporary scratch arena.
    // Passing &arena, 1 probably means "avoid overlapping this arena with scratch selection".
    mem_arena_temp scratch = arena_scratch_get(&arena, 1);

    // Allocate a visited array to track which nodes we have seen.
    b8* visited = PUSH_ARRAY(scratch.arena, b8, model->num_vars);

    // Current number of items in the DFS-like stack.
    u32 stack_size = 0;

    // Current number of nodes in the final output ordering.
    u32 out_size = 0;

    // Allocate stack storage for graph traversal.
    model_var** stack = PUSH_ARRAY(scratch.arena, model_var*, model->num_vars);

    // Allocate temporary output storage for sorted nodes.
    model_var** out = PUSH_ARRAY(scratch.arena, model_var*, model->num_vars);

    // Start traversal from the requested output node.
    stack[stack_size++] = out_var;

    // Continue until there are no more nodes to process.
    while (stack_size > 0) {
        // Pop one node from the stack.
        model_var* cur = stack[--stack_size];

        // Safety check: skip impossible/bad indices.
        if (cur->index >= model->num_vars) { continue; }

        // If we have already visited this node before...
        if (visited[cur->index]) {
            // ...then now we can append it to the output ordering,
            // because its inputs should already have been handled.
            if (out_size < model->num_vars) {
                out[out_size++] = cur;
            }

            // Move on to the next stacked node.
            continue;
        }

        // Mark this node as seen for the first time.
        visited[cur->index] = true;

        // Push the current node back onto the stack.
        // This is the classic "visit later after children" trick.
        if (stack_size < model->num_vars) {
            stack[stack_size++] = cur;
        }

        // Find out how many inputs this operation has.
        u32 num_inputs = MV_NUM_INPUTS(cur->op);

        // Iterate through each input node.
        for (u32 i = 0; i < num_inputs; i++) {
            // Get pointer to this input.
            model_var* input = cur->inputs[i];

            // Skip invalid or already-visited inputs.
            if (input->index >= model->num_vars || visited[input->index]) {
                continue;
            }

            // Remove duplicate occurrences of this input already in the stack.
            for (u32 j = 0; j < stack_size; j++) {
                if (stack[j] == input) {
                    for (u32 k = j; k < stack_size-1; k++) {
                        stack[k] = stack[k+1];
                    }
                    stack_size--;
                }
            }

            // Push this input so it gets processed before cur is finally emitted.
            if (stack_size < model->num_vars) {
                stack[stack_size++] = input;
            }
        }
    }

    // Allocate the final program object in permanent arena memory.
    model_program prog = {
        // Store how many nodes made it into the program.
        .size = out_size,

        // Allocate exact-size array for the final node list.
        .vars = PUSH_ARRAY_NZ(arena, model_var*, out_size)
    };

    // Copy the sorted node pointers from scratch memory into permanent memory.
    memcpy(prog.vars, out, sizeof(model_var*) * out_size);

    // Release scratch allocations.
    arena_scratch_release(scratch);

    // Return the compiled program.
    return prog;
}

// Execute forward computation for all nodes in a compiled program.
void model_prog_compute(model_program* prog) {
    // Process nodes in program order.
    for (u32 i = 0; i < prog->size; i++) {
        // Current node.
        model_var* cur = prog->vars[i];

        // First input pointer, if any.
        model_var* a = cur->inputs[0];

        // Second input pointer, if any.
        model_var* b = cur->inputs[1];

        // Dispatch based on operation type.
        switch (cur->op) {
            // Null node: do nothing.
            case MV_OP_NULL:

            // Created variable node: value already exists, so do nothing.
            case MV_OP_CREATE: break;

            // Marker enum value: do nothing.
            case _MV_OP_UNARY_START: break;

            // Compute ReLU.
            case MV_OP_RELU: { mat_relu(cur->val, a->val); } break;

            // Compute softmax.
            case MV_OP_SOFTMAX: { mat_softmax(cur->val, a->val); } break;

            // Marker enum value: do nothing.
            case _MV_OP_BINARY_START: break;

            // Compute addition.
            case MV_OP_ADD: { mat_add(cur->val, a->val, b->val); } break;

            // Compute subtraction.
            case MV_OP_SUB: { mat_sub(cur->val, a->val, b->val); } break;

            // Compute matrix multiplication.
            case MV_OP_MATMUL: {
                // zero_out = 1 means clear cur->val before accumulating result.
                // transpose flags are both 0.
                mat_mul(cur->val, a->val, b->val, 1, 0, 0); 
            } break;

            // Compute elementwise cross-entropy terms.
            case MV_OP_CROSS_ENTROPY: {
                mat_cross_entropy(cur->val, a->val, b->val);
            } break;
        }
    }
}

// Execute backward gradient propagation through a compiled program.
void model_prog_compute_grads(model_program* prog) {
    // First pass: clear gradients for non-parameter nodes that require grads.
    // Parameter grads are intentionally NOT cleared here because they accumulate over a batch.
    for (u32 i = 0; i < prog->size; i++) {
        // Current node.
        model_var* cur = prog->vars[i];

        // Skip nodes that do not require gradients.
        if ((cur->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD) {
            continue;
        }

        // Skip parameters because batch accumulation wants to keep them.
        if (cur->flags & MV_FLAG_PARAMETER) {
            continue;
        }

        // Clear this non-parameter gradient buffer.
        mat_clear(cur->grad);
    }

    // Seed the gradient at the final output of the program with all ones.
    // Because the cost node stores elementwise loss terms, this means d(sum(cost))/d(cost_i) = 1.
    mat_fill(prog->vars[prog->size-1]->grad, 1.0f);

    // Walk backward through the program.
    for (i64 i = (i64)prog->size - 1; i >= 0; i--) {
        // Current node.
        model_var* cur = prog->vars[i];

        // Skip nodes that do not participate in gradient flow.
        if ((cur->flags & MV_FLAG_REQUIRES_GRAD) == 0) {
            continue;
        }

        // First input pointer.
        model_var* a = cur->inputs[0];

        // Second input pointer.
        model_var* b = cur->inputs[1];

        // Number of inputs to this op.
        u32 num_inputs = MV_NUM_INPUTS(cur->op);

        // If unary op input does not require gradient, skip it.
        if (
            num_inputs == 1 &&
            (a->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD
        ) {
            continue;
        }

        // If binary op and neither input requires gradient, skip it.
        if (
            num_inputs == 2 &&
            (a->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD && 
            (b->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD
        ) {
            continue;
        }

        // Dispatch based on operation type.
        switch (cur->op) {
            // Null node: nothing to backprop through.
            case MV_OP_NULL:

            // Created variable: nothing to backprop through here.
            case MV_OP_CREATE: break;

            // Marker enum value: do nothing.
            case _MV_OP_UNARY_START: break;

            // Backprop through ReLU.
            case MV_OP_RELU: {
                mat_relu_add_grad(a->grad, a->val, cur->grad);
            } break;

            // Backprop through softmax.
            case MV_OP_SOFTMAX: {
                mat_softmax_add_grad(a->grad, cur->val, cur->grad);
            } break;

            // Marker enum value: do nothing.
            case _MV_OP_BINARY_START: break;

            // Backprop through addition.
            case MV_OP_ADD: {
                // d(a + b)/da = 1, so add current gradient into a.
                if (a->flags & MV_FLAG_REQUIRES_GRAD) {
                    mat_add(a->grad, a->grad, cur->grad);
                }

                // d(a + b)/db = 1, so add current gradient into b.
                if (b->flags & MV_FLAG_REQUIRES_GRAD) {
                    mat_add(b->grad, b->grad, cur->grad);
                }
            } break;

            // Backprop through subtraction.
            case MV_OP_SUB: {
                // d(a - b)/da = 1, so add current gradient into a.
                if (a->flags & MV_FLAG_REQUIRES_GRAD) {
                    mat_add(a->grad, a->grad, cur->grad);
                }

                // d(a - b)/db = -1, so subtract current gradient into b.
                if (b->flags & MV_FLAG_REQUIRES_GRAD) {
                    mat_sub(b->grad, b->grad, cur->grad);
                }
            } break;

            // Backprop through matrix multiplication.
            case MV_OP_MATMUL: {
                // If cur = a * b, then:
                // dL/da += dL/dcur * b^T
                if (a->flags & MV_FLAG_REQUIRES_GRAD) {
                    mat_mul(a->grad, cur->grad, b->val, 0, 0, 1);
                }

                // And:
                // dL/db += a^T * dL/dcur
                if (b->flags & MV_FLAG_REQUIRES_GRAD) {
                    mat_mul(b->grad, a->val, cur->grad, 0, 1, 0);
                }
            } break;

            // Backprop through cross-entropy.
            case MV_OP_CROSS_ENTROPY: {
                // Rename inputs for clarity: p is target, q is prediction.
                model_var* p = a;
                model_var* q = b;

                // Add gradient contributions into p->grad and q->grad.
                mat_cross_entropy_add_grad(
                    p->grad, q->grad, p->val, q->val, cur->grad
                );
            } break;
        }
    }
}

// Create an empty model context.
model_context* model_create(mem_arena* arena) {
    // Allocate the model context struct.
    model_context* model = PUSH_STRUCT(arena, model_context);

    // IMPORTANT:
    // This function does not explicitly zero fields.
    // It relies on PUSH_STRUCT / arena behavior, or later code, for sensible defaults.
    return model;
}

// Compile the model into forward and cost programs.
void model_compile(mem_arena* arena, model_context* model) {
    // If the model has an output node, compile a forward program ending there.
    if (model->output != NULL) {
        model->forward_prog = model_prog_create(arena, model, model->output);
    }

    // If the model has a cost node, compile a cost program ending there.
    if (model->cost != NULL) {
        model->cost_prog = model_prog_create(arena, model, model->cost);
    }
}

// Run forward inference through the forward program.
void model_feedforward(model_context* model) {
    // Execute the forward program.
    model_prog_compute(&model->forward_prog);
}

// Train the model using mini-batch gradient descent.
void model_train(
    model_context* model,
    const model_training_desc* training_desc
) {
    // Pull training image pointer into a local variable for shorter code.
    matrix* train_images = training_desc->train_images;

    // Pull training labels pointer into a local variable.
    matrix* train_labels = training_desc->train_labels;

    // Pull test image pointer into a local variable.
    matrix* test_images = training_desc->test_images;

    // Pull test labels pointer into a local variable.
    matrix* test_labels = training_desc->test_labels;

    // Number of training examples = number of rows in the training image matrix.
    u32 num_examples = train_images->rows;

    // Number of input features per example = number of columns in training images.
    u32 input_size = train_images->cols;

    // Number of output classes = number of columns in label matrix.
    u32 output_size = train_labels->cols;

    // Number of test examples.
    u32 num_tests = test_images->rows;

    // Number of full mini-batches we can make.
    // This drops any remainder if num_examples is not divisible by batch_size.
    u32 num_batches = num_examples / training_desc->batch_size;

    // Get scratch memory for temporary training-order array.
    mem_arena_temp scratch = arena_scratch_get(NULL, 0);

    // Allocate an array of training indices.
    u32* training_order = PUSH_ARRAY_NZ(scratch.arena, u32, num_examples);

    // Initialize training_order to 0, 1, 2, ..., num_examples-1.
    for (u32 i = 0; i < num_examples; i++) {
        training_order[i] = i;
    }

    // Outer loop over epochs.
    for (u32 epoch = 0; epoch < training_desc->epochs; epoch++) {
        // Shuffle training_order by doing many random swaps.
        // This is not the clean Fisher-Yates algorithm, but it does randomize the order somewhat.
        for (u32 i = 0; i < num_examples; i++) {
            // Pick random index a.
            u32 a = prng_rand() % num_examples;

            // Pick random index b.
            u32 b = prng_rand() % num_examples;

            // Swap training_order[a] and training_order[b].
            u32 tmp = training_order[b];
            training_order[b] = training_order[a];
            training_order[a] = tmp;
        }

        // Loop over mini-batches.
        for (u32 batch = 0; batch < num_batches; batch++) {
            // Before processing this batch, clear all parameter gradients.
            // Those gradients will then accumulate contributions from each example in the batch.
            for (u32 i = 0; i < model->cost_prog.size; i++) {
                // Current node in the cost program.
                model_var* cur = model->cost_prog.vars[i];

                // If this node is a trainable parameter...
                if (cur->flags & MV_FLAG_PARAMETER) {
                    // ...clear its gradient buffer.
                    mat_clear(cur->grad);
                }
            }

            // Running average cost for this batch.
            f32 avg_cost = 0.0f;

            // Loop over examples inside the current batch.
            for (u32 i = 0; i < training_desc->batch_size; i++) {
                // Convert batch-local index into global shuffled-order index.
                u32 order_index = batch * training_desc->batch_size + i;

                // Convert that into the actual dataset example index.
                u32 index = training_order[order_index];

                // Copy one training image into the model input vector.
                memcpy(
                    model->input->val->data,
                    train_images->data + index * input_size,
                    sizeof(f32) * input_size
                );

                // Copy the corresponding one-hot label into the desired-output vector.
                memcpy(
                    model->desired_output->val->data,
                    train_labels->data + index * output_size,
                    sizeof(f32) * output_size
                );

                // Run forward computation all the way to the cost node.
                model_prog_compute(&model->cost_prog);

                // Run backward computation to accumulate gradients.
                model_prog_compute_grads(&model->cost_prog);

                // Add the scalarized loss for this example.
                // Because cost is stored as elementwise terms, mat_sum turns it into total loss.
                avg_cost += mat_sum(model->cost->val);
            }

            // Divide by batch size to get average batch cost.
            avg_cost /= (f32)training_desc->batch_size;

            // Apply gradient descent update to each parameter.
            for (u32 i = 0; i < model->cost_prog.size; i++) {
                // Current node.
                model_var* cur = model->cost_prog.vars[i];

                // Skip non-parameter nodes.
                if ((cur->flags & MV_FLAG_PARAMETER) != MV_FLAG_PARAMETER) {
                    continue;
                }

                // Scale accumulated gradient by learning_rate / batch_size.
                // This converts sum-of-gradients over batch into average-gradient step size.
                mat_scale(
                    cur->grad,
                    training_desc->learning_rate /
                    training_desc->batch_size
                );

                // Update parameter in place:
                // parameter = parameter - scaled_gradient
                mat_sub(cur->val, cur->val, cur->grad);
            }

            // Print progress on one line with carriage return.
            // \r returns cursor to start of the line so next print overwrites it.
            printf(
                "Epoch %2d / %2d, Batch %4d / %4d, Average Cost: %.4f\r",
                epoch + 1, training_desc->epochs,
                batch + 1, num_batches, avg_cost
            );

            // Force the line to appear immediately.
            fflush(stdout);
        }

        // After all batches in this epoch, print a newline so progress line is finalized.
        printf("\n");

        // Count correct predictions on the test set.
        u32 num_correct = 0;

        // Running average test cost.
        f32 avg_cost = 0;

        // Loop over all test examples.
        for (u32 i = 0; i < num_tests; i++) {
            // Copy test image i into the model input.
            memcpy(
                model->input->val->data,
                test_images->data + i * input_size,
                sizeof(f32) * input_size
            );

            // Copy test label i into desired output.
            memcpy(
                model->desired_output->val->data,
                test_labels->data + i * output_size,
                sizeof(f32) * output_size
            );

            // Run forward computation to the cost node.
            model_prog_compute(&model->cost_prog);

            // Add this example's total loss to running test cost.
            avg_cost += mat_sum(model->cost->val);

            // Compare predicted class index with true class index.
            // If they match, add 1 to num_correct.
            num_correct +=
                mat_argmax(model->output->val) ==
                mat_argmax(model->desired_output->val);
        }

        // Convert total test cost to average test cost.
        avg_cost /= (f32)num_tests;

        // Print epoch test summary.
        printf(
            "Test Completed. Accuracy: %5d / %5d (%.1f%%), Average Cost: %.4f\n",
            num_correct, num_tests, (f32)num_correct / num_tests * 100.0f,
            avg_cost
        );
    }

    // Release scratch memory used for training_order.
    arena_scratch_release(scratch);
}