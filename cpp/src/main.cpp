#include "simple_ai/simple_ai.hpp"

#include <iostream>
#include <string>

int main() {
    simple_ai::TinyTransitionModel model;
    const std::string training_text =
        "Simple AI is built from scratch. "
        "This is a tiny offline model prototype. "
        "We learn one character transition at a time. ";

    model.train(training_text);
    std::cout << model.generate("Simple AI", 100, 7) << '\n';
    return 0;
}
